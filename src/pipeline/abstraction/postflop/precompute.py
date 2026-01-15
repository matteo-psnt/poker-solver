"""
Precomputation pipeline for combo-level abstraction.

This module handles the computation-heavy task of:
1. Enumerating all canonical boards
2. For each canonical board, enumerating all canonical hand combos
3. Computing equity for each (canonical_board, canonical_hand) pair
4. Clustering combos into buckets using K-means on equity values

The result is a sparse lookup table:
    street -> canonical_board_id -> canonical_hand_id -> bucket_id
"""

import json
import multiprocessing as mp
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.core.game.state import Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.board_clustering import BoardClusterer
from src.pipeline.abstraction.postflop.board_enumeration import (
    CanonicalBoardEnumerator,
    CanonicalBoardInfo,
)
from src.pipeline.abstraction.postflop.hand_bucketing import (
    PostflopBucketer,
    get_all_canonical_hands,
    get_representative_hand,
)
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    canonicalize_board,
    get_canonical_board_id,
)
from src.pipeline.abstraction.utils.equity import RangeEquityEngine


def _worker_compute_cluster_equities(args) -> list[tuple[int, int, int, float]]:
    """
    Worker that computes equities for one representative board, tagged with cluster_id.

    One exact range-vs-range pass covers every combo on the board; canonical
    combos are then looked up from that shared table.
    """
    board_info, cluster_id, flop_runouts, seed = args

    engine = RangeEquityEngine(max_runouts=flop_runouts, seed=seed)
    table = engine.board_equities(board_info.representative)

    _, suit_mapping = canonicalize_board(board_info.representative)
    results = []
    for combo in get_all_canonical_hands(board_info.representative):
        rep_hand = get_representative_hand(combo.hand, suit_mapping)
        results.append((cluster_id, board_info.board_id, combo.hand_id, table.equity(rep_hand)))

    return results


class PostflopPrecomputer:
    """
    Precomputes combo-level abstraction data using board clustering.

    This is the main entry point for generating abstraction tables.
    Uses public-state (board) clustering to make precomputation tractable.
    """

    def __init__(self, config: PrecomputeConfig):
        """
        Initialize precomputer.

        Args:
            config: Precomputation configuration
        """
        self.config = config

        # Initialize board clusterer
        self.board_clusterer = BoardClusterer(config=self.config)

        # Storage for computed equities: street -> cluster_id -> hand_id -> equity
        self._equities: dict[Street, dict[int, dict[int, float]]] = {
            Street.FLOP: {},
            Street.TURN: {},
            Street.RIVER: {},
        }

        # Final abstraction object
        self.abstraction = PostflopBucketer()

        # Attach board clusterer to abstraction for runtime use
        self.abstraction.set_board_clusterer(self.board_clusterer)

    def precompute_street(
        self,
        street: Street,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Precompute abstraction for a single street using board clustering.

        Pipeline:
        1. Enumerate all canonical boards
        2. Cluster boards by texture
        3. Compute equity only for representative boards
        4. Cluster hands into buckets per cluster
        5. Store by cluster_id (not board_id)

        Args:
            street: Which street to precompute
            progress_callback: Optional callback(current, total) for progress
        """
        print(f"Starting precomputation for {street.name}")

        # Step 1: Enumerate all canonical boards
        print("Step 1: Enumerating canonical boards...")
        enumerator = CanonicalBoardEnumerator(street)
        enumerator.enumerate()

        board_infos = list(enumerator.iterate())
        total_boards = len(board_infos)
        print(f"Found {total_boards} canonical boards for {street.name}")

        # Step 2: Cluster boards by texture
        print(
            f"Step 2: Clustering boards into {self.config.num_board_clusters[street]} clusters..."
        )
        canonical_boards = [info.canonical_board for info in board_infos]
        representative_boards = [info.representative for info in board_infos]

        self.board_clusterer.fit(
            canonical_boards=canonical_boards,
            representative_boards=representative_boards,
            street=street,
        )

        clusters = self.board_clusterer.get_all_clusters(street)
        print(f"Created {len(clusters)} clusters")

        # Step 3: Compute equities only for representative boards
        print(
            f"Step 3: Computing equities for representatives ({self.config.representatives_per_cluster} per cluster)..."
        )

        num_workers = self.config.num_workers or mp.cpu_count()

        # Collect all representative boards from all clusters
        work_items = []
        for cluster in clusters:
            for rep_board in cluster.representative_boards:
                canonical_rep, _ = canonicalize_board(rep_board)
                board_info = CanonicalBoardInfo(
                    canonical_board=canonical_rep,
                    board_id=get_canonical_board_id(canonical_rep),
                    raw_count=1,
                    representative=rep_board,
                )
                work_items.append(
                    (board_info, cluster.cluster_id, self.config.flop_runouts, self.config.seed)
                )

        total_work = len(work_items)
        print(
            f"Computing equity for {total_work} representative boards (vs {total_boards} without clustering)"
        )

        # Process in parallel
        all_equities: list[
            tuple[int, int, int, float]
        ] = []  # (cluster_id, board_id, hand_id, equity)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_worker_compute_cluster_equities, item): i
                for i, item in enumerate(work_items)
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Computing {street.name} equities",
            ):
                board_results = future.result()
                all_equities.extend(board_results)

                if progress_callback:
                    progress_callback(len(all_equities) // 500, total_work)

        # Step 4: Organize equities by cluster (aggregate across representatives)
        print("Step 4: Aggregating equities by cluster...")

        # Aggregate: for each (cluster_id, hand_id), average equity across representatives
        cluster_hand_equities: dict[int, dict[int, list[float]]] = {}

        for cluster_id, board_id, hand_id, equity in all_equities:
            if cluster_id not in cluster_hand_equities:
                cluster_hand_equities[cluster_id] = {}
            if hand_id not in cluster_hand_equities[cluster_id]:
                cluster_hand_equities[cluster_id][hand_id] = []

            cluster_hand_equities[cluster_id][hand_id].append(equity)

        # Average equities for each (cluster, hand)
        for cluster_id, hands in cluster_hand_equities.items():
            if cluster_id not in self._equities[street]:
                self._equities[street][cluster_id] = {}

            for hand_id, equity_list in hands.items():
                self._equities[street][cluster_id][hand_id] = float(np.mean(equity_list))

        # Step 5: Cluster into buckets using K-means
        print(f"Step 5: Clustering into {self.config.num_buckets[street]} buckets...")
        self._cluster_street(street)

        print(f"Completed precomputation for {street.name}")

    def _cluster_street(self, street: Street) -> None:
        """
        Cluster equity values into buckets using K-means.

        Uses global clustering across all clusters for consistency.
        Now operates on (cluster_id, hand_id) pairs instead of (board_id, hand_id).
        """
        num_buckets = self.config.num_buckets[street]

        # Collect all equity values with their (cluster_id, hand_id) keys
        all_data = []
        for cluster_id, hands in self._equities[street].items():
            for hand_id, equity in hands.items():
                all_data.append((cluster_id, hand_id, equity))

        if not all_data:
            print(f"Warning: No data to cluster for {street.name}")
            return

        # Extract just the equity values for clustering
        equities = np.array([x[2] for x in all_data]).reshape(-1, 1)

        # Fit K-means
        kmeans = KMeans(
            n_clusters=min(num_buckets, len(all_data)),
            max_iter=self.config.kmeans_max_iter,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.seed,
        )
        labels = kmeans.fit_predict(equities)

        # Sort cluster centers and remap labels so bucket 0 = lowest equity
        center_order = np.argsort(kmeans.cluster_centers_.flatten())
        label_map = {old: new for new, old in enumerate(center_order)}

        # Assign buckets to abstraction (indexed by cluster_id, not board_id)
        for (cluster_id, hand_id, equity), label in zip(all_data, labels):
            bucket = label_map[label]
            self.abstraction.assign_bucket(street, cluster_id, hand_id, bucket)

        self.abstraction.set_num_buckets(street, len(set(labels)))

        print(f"Clustered {len(all_data)} combos into {len(set(labels))} buckets for {street.name}")

    def precompute_all(
        self,
        streets: list[Street] | None = None,
    ) -> PostflopBucketer:
        """
        Precompute abstraction for all streets.

        Args:
            streets: Which streets to precompute (default: all postflop)

        Returns:
            Fitted PostflopBucketer object
        """
        if streets is None:
            streets = [Street.FLOP, Street.TURN, Street.RIVER]

        for street in streets:
            self.precompute_street(street)

        return self.abstraction

    def save(self, path: Path) -> None:
        """
        Save precomputed abstraction to disk.

        Creates:
        - combo_abstraction.pkl: The PostflopBucketer object (includes board_clusterer)
        - metadata.json: Configuration and statistics
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save abstraction (includes board_clusterer)
        with open(path / "combo_abstraction.pkl", "wb") as f:
            pickle.dump(self.abstraction, f)

        # Save metadata
        statistics: dict[str, dict[str, int]] = {}
        metadata = {
            "config": self.config.model_dump(),
            "config_hash": self.config.get_config_hash(),
            "statistics": statistics,
        }

        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            if self._equities.get(street):
                num_clusters = len(self._equities[street])
                num_combos = sum(len(h) for h in self._equities[street].values())
                statistics[street.name] = {
                    "num_clusters": num_clusters,
                    "num_combos": num_combos,
                    "num_buckets": self.abstraction.num_buckets(street),
                }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved abstraction to {path}")

    @classmethod
    def load(cls, path: Path) -> PostflopBucketer:
        """
        Load precomputed abstraction from disk.

        Args:
            path: Directory containing saved abstraction

        Returns:
            Loaded PostflopBucketer object
        """
        path = Path(path)

        with open(path / "combo_abstraction.pkl", "rb") as f:
            try:
                return pickle.load(f)
            except (ModuleNotFoundError, AttributeError) as exc:
                raise RuntimeError(
                    f"Stale abstraction at {path}: it was pickled against an older "
                    f"code layout ({exc}). Regenerate it via 'Precompute Combo "
                    f"Abstraction' from the CLI."
                ) from exc
