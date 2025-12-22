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
import logging
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.abstraction.isomorphism.board_clustering import BoardClusterer
from src.abstraction.isomorphism.canonical_boards import (
    CanonicalBoardEnumerator,
)
from src.abstraction.isomorphism.combo_abstraction import (
    CanonicalCombo,
    ComboAbstraction,
    get_all_canonical_combos,
    get_representative_hand,
)
from src.abstraction.isomorphism.suit_canonicalization import (
    CanonicalCard,
    canonicalize_board,
    get_canonical_board_id,
    get_canonical_hand_id,
)
from src.abstraction.utils import EquityCalculator
from src.game.state import Card, Street

logger = logging.getLogger(__name__)


@dataclass
class PrecomputeConfig:
    """Configuration for precomputation."""

    # Board clustering (public-state abstraction)
    num_board_clusters: Dict[Street, int]

    # Representatives per cluster (for equity computation)
    representatives_per_cluster: int

    # Number of buckets per street (hand abstraction)
    num_buckets: Dict[Street, int]

    # Number of Monte Carlo samples for equity calculation
    equity_samples: int = 1000

    # Number of parallel workers (None = CPU count)
    num_workers: Optional[int] = None

    # Random seed for reproducibility
    seed: int = 42

    # K-means settings
    kmeans_max_iter: int = 300
    kmeans_n_init: int = 10

    # Config name (for matching during training)
    config_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_name: str) -> "PrecomputeConfig":
        """Load configuration from YAML file.

        Args:
            config_name: Name of config file (without .yaml extension)
                        e.g., 'fast_test', 'default', 'production'

        Returns:
            PrecomputeConfig instance
        """
        from pathlib import Path

        import yaml

        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "abstraction"
            / f"{config_name}.yaml"
        )

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Parse board clusters
        board_clusters_data = data.get("board_clusters", {})
        num_board_clusters = {
            Street.FLOP: board_clusters_data.get("flop"),
            Street.TURN: board_clusters_data.get("turn"),
            Street.RIVER: board_clusters_data.get("river"),
        }

        # Parse buckets
        buckets_data = data.get("buckets", {})
        num_buckets = {
            Street.FLOP: buckets_data.get("flop"),
            Street.TURN: buckets_data.get("turn"),
            Street.RIVER: buckets_data.get("river"),
        }

        return cls(
            num_board_clusters=num_board_clusters,
            representatives_per_cluster=data.get("representatives_per_cluster", 1),
            num_buckets=num_buckets,
            equity_samples=data.get("equity_samples", 1000),
            num_workers=data.get("num_workers"),
            seed=data.get("seed", 42),
            kmeans_max_iter=data.get("kmeans_max_iter", 300),
            kmeans_n_init=data.get("kmeans_n_init", 10),
            config_name=config_name,
        )

    @classmethod
    def default(cls) -> "PrecomputeConfig":
        """Load the default configuration."""
        return cls.from_yaml("default")

    @classmethod
    def fast_test(cls) -> "PrecomputeConfig":
        """Load the fast_test configuration for quick testing."""
        return cls.from_yaml("fast_test")


@dataclass
class ComboEquity:
    """Equity information for a canonical combo."""

    combo: CanonicalCombo
    equity: float

    # For debugging/verification
    representative_board: Tuple[Card, ...]
    representative_hand: Tuple[Card, Card]


def compute_equity_for_combo(
    canonical_board: Tuple[CanonicalCard, ...],
    canonical_hand: Tuple[CanonicalCard, CanonicalCard],
    representative_board: Tuple[Card, ...],
    equity_samples: int,
    seed: int,
) -> Tuple[int, int, float]:
    """
    Compute equity for a single combo. Designed for parallel execution.

    Returns:
        (board_id, hand_id, equity)
    """
    # Get representative hand from canonical form
    _, suit_mapping = canonicalize_board(representative_board)
    rep_hand = get_representative_hand(canonical_hand, suit_mapping)

    # Check for card conflicts
    board_set = set(representative_board)
    if rep_hand[0] in board_set or rep_hand[1] in board_set:
        # This shouldn't happen if generation is correct
        return (
            get_canonical_board_id(canonical_board),
            get_canonical_hand_id(canonical_hand),
            -1.0,  # Error marker
        )

    # Determine street from board length
    if len(representative_board) == 3:
        street = Street.FLOP
    elif len(representative_board) == 4:
        street = Street.TURN
    else:
        street = Street.RIVER

    # Calculate equity
    calculator = EquityCalculator(num_samples=equity_samples, seed=seed)
    equity = calculator.calculate_equity(rep_hand, representative_board, street)

    return (
        get_canonical_board_id(canonical_board),
        get_canonical_hand_id(canonical_hand),
        equity,
    )


def _worker_compute_board_equities(args) -> List[Tuple[int, int, float]]:
    """
    Worker function to compute equities for all combos on a single board.

    This is the unit of parallelism - one board per worker.
    """
    board_info, equity_samples, seed = args

    results = []

    # Get all canonical combos for this board
    for i, combo in enumerate(get_all_canonical_combos(board_info.representative)):
        board_id, hand_id, equity = compute_equity_for_combo(
            canonical_board=combo.board,
            canonical_hand=combo.hand,
            representative_board=board_info.representative,
            equity_samples=equity_samples,
            seed=seed + i,  # Vary seed per combo for variety
        )
        results.append((board_id, hand_id, equity))

    return results


def _worker_compute_cluster_equities(args) -> List[Tuple[int, int, int, float]]:
    """Worker that computes equities for representative boards and tags them with cluster_id."""
    board_info, cluster_id, equity_samples, seed = args
    results = []

    for i, combo in enumerate(get_all_canonical_combos(board_info.representative)):
        board_id, hand_id, equity = compute_equity_for_combo(
            canonical_board=combo.board,
            canonical_hand=combo.hand,
            representative_board=board_info.representative,
            equity_samples=equity_samples,
            seed=seed + i,
        )
        results.append((cluster_id, board_id, hand_id, equity))

    return results


class ComboPrecomputer:
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
        self.board_clusterer = BoardClusterer(config.num_board_clusters)

        # Storage for computed equities: street -> cluster_id -> hand_id -> equity
        self._equities: Dict[Street, Dict[int, Dict[int, float]]] = {
            Street.FLOP: {},
            Street.TURN: {},
            Street.RIVER: {},
        }

        # Final abstraction object
        self.abstraction = ComboAbstraction()

        # Attach board clusterer to abstraction for runtime use
        self.abstraction._board_clusterer = self.board_clusterer

    def precompute_street(
        self,
        street: Street,
        progress_callback: Optional[Callable[[int, int], None]] = None,
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
        logger.info(f"Starting precomputation for {street.name}")

        # Step 1: Enumerate all canonical boards
        logger.info("Step 1: Enumerating canonical boards...")
        enumerator = CanonicalBoardEnumerator(street)
        enumerator.enumerate()

        board_infos = list(enumerator.iterate())
        total_boards = len(board_infos)
        logger.info(f"Found {total_boards} canonical boards for {street.name}")

        # Step 2: Cluster boards by texture
        logger.info(
            f"Step 2: Clustering boards into {self.config.num_board_clusters[street]} clusters..."
        )
        canonical_boards = [info.canonical_board for info in board_infos]
        representative_boards = [info.representative for info in board_infos]

        self.board_clusterer.fit(
            canonical_boards=canonical_boards,
            representative_boards=representative_boards,
            street=street,
            representatives_per_cluster=self.config.representatives_per_cluster,
        )

        clusters = self.board_clusterer.get_all_clusters(street)
        logger.info(f"Created {len(clusters)} clusters")

        # Step 3: Compute equities only for representative boards
        logger.info(
            f"Step 3: Computing equities for representatives ({self.config.representatives_per_cluster} per cluster)..."
        )

        num_workers = self.config.num_workers or mp.cpu_count()

        # Collect all representative boards from all clusters
        work_items = []
        for cluster in clusters:
            for rep_board in cluster.representative_boards:
                # Create a minimal board info for the worker
                from src.abstraction.isomorphism.canonical_boards import CanonicalBoardInfo

                canonical_rep, _ = canonicalize_board(rep_board)
                board_info = CanonicalBoardInfo(
                    canonical_board=canonical_rep,
                    board_id=get_canonical_board_id(canonical_rep),
                    raw_count=1,
                    representative=rep_board,
                )
                work_items.append(
                    (board_info, cluster.cluster_id, self.config.equity_samples, self.config.seed)
                )

        total_work = len(work_items)
        logger.info(
            f"Computing equity for {total_work} representative boards (vs {total_boards} without clustering)"
        )

        # Process in parallel
        all_equities: List[
            Tuple[int, int, int, float]
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
        logger.info("Step 4: Aggregating equities by cluster...")

        # Aggregate: for each (cluster_id, hand_id), average equity across representatives
        cluster_hand_equities: Dict[int, Dict[int, List[float]]] = {}

        for cluster_id, board_id, hand_id, equity in all_equities:
            if equity < 0:
                continue

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
        logger.info(f"Step 5: Clustering into {self.config.num_buckets[street]} buckets...")
        self._cluster_street(street)

        logger.info(f"Completed precomputation for {street.name}")

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
            logger.warning(f"No data to cluster for {street.name}")
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

            if cluster_id not in self.abstraction._buckets[street]:
                self.abstraction._buckets[street][cluster_id] = {}
            self.abstraction._buckets[street][cluster_id][hand_id] = bucket

        self.abstraction.set_num_buckets(street, len(set(labels)))

        logger.info(
            f"Clustered {len(all_data)} combos into {len(set(labels))} buckets for {street.name}"
        )

    def precompute_all(
        self,
        streets: Optional[List[Street]] = None,
    ) -> ComboAbstraction:
        """
        Precompute abstraction for all streets.

        Args:
            streets: Which streets to precompute (default: all postflop)

        Returns:
            Fitted ComboAbstraction object
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
        - combo_abstraction.pkl: The ComboAbstraction object (includes board_clusterer)
        - metadata.json: Configuration and statistics
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save abstraction (includes board_clusterer)
        with open(path / "combo_abstraction.pkl", "wb") as f:
            pickle.dump(self.abstraction, f)

        # Save metadata
        metadata = {
            "config": {
                "config_name": self.config.config_name,
                "num_board_clusters": {
                    s.name: n for s, n in self.config.num_board_clusters.items()
                },
                "representatives_per_cluster": self.config.representatives_per_cluster,
                "num_buckets": {s.name: n for s, n in self.config.num_buckets.items()},
                "equity_samples": self.config.equity_samples,
                "seed": self.config.seed,
            },
            "statistics": {},
        }

        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            if street in self._equities and self._equities[street]:
                num_clusters = len(self._equities[street])
                num_combos = sum(len(h) for h in self._equities[street].values())
                metadata["statistics"][street.name] = {
                    "num_clusters": num_clusters,
                    "num_combos": num_combos,
                    "num_buckets": self.abstraction.num_buckets(street),
                }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved abstraction to {path}")

    @classmethod
    def load(cls, path: Path) -> ComboAbstraction:
        """
        Load precomputed abstraction from disk.

        Args:
            path: Directory containing saved abstraction

        Returns:
            Loaded ComboAbstraction object
        """
        path = Path(path)

        with open(path / "combo_abstraction.pkl", "rb") as f:
            return pickle.load(f)


def estimate_precompute_time(
    config: PrecomputeConfig,
    sample_size: int = 100,
) -> Dict[Street, float]:
    """
    Estimate precomputation time by running a small sample.

    Returns estimated hours per street.
    """
    import time

    estimates = {}

    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        enumerator = CanonicalBoardEnumerator(street)
        enumerator.enumerate()

        total_boards = len(list(enumerator.iterate()))

        # Time a few boards
        start = time.time()
        sample_boards = list(enumerator.iterate())[:sample_size]

        for board_info in sample_boards:
            for combo in get_all_canonical_combos(board_info.representative):
                # Just enumerate, don't compute equity for estimate
                pass

        elapsed = time.time() - start

        # Estimate total time
        # Equity calculation is ~100x slower than enumeration
        estimated_seconds = (elapsed / sample_size) * total_boards * 100
        estimates[street] = estimated_seconds / 3600  # Convert to hours

    return estimates
