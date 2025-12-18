"""
Equity-based K-means bucketing for postflop card abstraction.

Clusters (hand, board) pairs into buckets based on equity values.
This is the final component that ties together:
- PreflopHandMapper (169 hands)
- EquityCalculator (Monte Carlo equity)
- BoardClusterer (board texture clustering)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from src.abstraction.board_clustering import BoardClusterer
from src.abstraction.constants import (
    DEFAULT_EQUITY_SAMPLES,
    DEFAULT_FLOP_BOARD_CLUSTERS,
    DEFAULT_FLOP_BUCKETS,
    DEFAULT_RIVER_BOARD_CLUSTERS,
    DEFAULT_RIVER_BUCKETS,
    DEFAULT_TURN_BOARD_CLUSTERS,
    DEFAULT_TURN_BUCKETS,
    KMEANS_MAX_ITER,
    KMEANS_RANDOM_STATE,
    NUM_PREFLOP_HANDS,
)
from src.abstraction.equity_calculator import EquityCalculator
from src.abstraction.preflop_hands import PreflopHandMapper
from src.game.state import Card, Street

# Setup logger
logger = logging.getLogger(__name__)


class EquityBucketing:
    """
    Equity-based card abstraction using K-means clustering.

    For each street:
    - Precompute equity for all 169 hands × board clusters
    - Cluster (hand, board) pairs into buckets based on equity
    - Bucket counts: 50 (flop), 100 (turn), 200 (river)

    Storage requirements:
    - Flop: 169 × 200 × 1 byte = 34 KB
    - Turn: 169 × 500 × 1 byte = 85 KB
    - River: 169 × 1000 × 1 byte = 169 KB
    Total: ~288 KB (very compact!)
    """

    def __init__(
        self,
        num_buckets: Optional[Dict[Street, int]] = None,
        num_board_clusters: Optional[Dict[Street, int]] = None,
        equity_calculator: Optional[EquityCalculator] = None,
        board_clusterer: Optional[BoardClusterer] = None,
    ):
        """
        Initialize equity bucketing.

        Args:
            num_buckets: Dict mapping Street → num_buckets
                        Default: {FLOP: 50, TURN: 100, RIVER: 200}
            num_board_clusters: Dict mapping Street → num_board_clusters
                               Default: {FLOP: 200, TURN: 500, RIVER: 1000}
                               For small tests, use smaller values (e.g., 5-10)
            equity_calculator: EquityCalculator instance (created if None)
            board_clusterer: BoardClusterer instance (created if None)
        """
        if num_buckets is None:
            num_buckets = {
                Street.FLOP: DEFAULT_FLOP_BUCKETS,
                Street.TURN: DEFAULT_TURN_BUCKETS,
                Street.RIVER: DEFAULT_RIVER_BUCKETS,
            }
        self.num_buckets = num_buckets
        self.equity_calculator = equity_calculator or EquityCalculator(
            num_samples=DEFAULT_EQUITY_SAMPLES
        )

        # Create board clusterer with appropriate cluster counts
        if board_clusterer is None:
            if num_board_clusters is None:
                # Use default board cluster counts
                num_board_clusters = {
                    Street.FLOP: DEFAULT_FLOP_BOARD_CLUSTERS,
                    Street.TURN: DEFAULT_TURN_BOARD_CLUSTERS,
                    Street.RIVER: DEFAULT_RIVER_BOARD_CLUSTERS,
                }
            self.board_clusterer = BoardClusterer(num_clusters=num_board_clusters)
        else:
            self.board_clusterer = board_clusterer
        self.hand_mapper = PreflopHandMapper()

        # Bucket assignments: bucket_assignments[street][hand_idx][board_cluster] = bucket_id
        self.bucket_assignments: Dict[Street, np.ndarray] = {}

        # K-means models: clusterers[street] = KMeans model
        self.clusterers: Dict[Street, KMeans] = {}

        # Track if fitted
        self.fitted = False

    def fit(
        self,
        sample_boards: Dict[Street, list],
        num_samples_per_cluster: int = 10,
    ):
        """
        Fit bucketing system on sample boards.

        This is the training/precomputation step that:
        1. Fits board clusterer on sample boards
        2. Computes equity for all (hand, board_cluster) pairs
        3. Runs K-means to assign buckets

        Args:
            sample_boards: Dict mapping Street → list of board tuples
                          e.g., {Street.FLOP: [(As, Ks, Qs), ...]}
            num_samples_per_cluster: How many boards to sample per cluster
                                    for equity calculation
        """
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            if street not in sample_boards:
                continue

            logger.info(f"Fitting {street.name}...")

            # 1. Fit board clusterer
            boards = sample_boards[street]
            self.board_clusterer.fit(boards, street)
            num_board_clusters = self.board_clusterer.num_clusters[street]

            # 2. Sample representative boards for each cluster
            cluster_representatives = self._sample_cluster_representatives(
                boards, street, num_samples_per_cluster
            )

            # 3. Compute equity matrix: [169 hands × num_board_clusters]
            equity_matrix = self._compute_equity_matrix(cluster_representatives, street)

            # 4. Run K-means clustering on (hand, board_cluster) pairs
            bucket_assignments = self._cluster_equities(equity_matrix, street)

            self.bucket_assignments[street] = bucket_assignments

            logger.info(
                f"{street.name}: {len(boards)} boards → "
                f"{num_board_clusters} clusters → "
                f"{self.num_buckets[street]} buckets"
            )

        self.fitted = True

    def _sample_cluster_representatives(
        self,
        boards: list,
        street: Street,
        num_samples_per_cluster: int,
    ) -> Dict[int, list]:
        """
        Sample representative boards for each cluster.

        Returns:
            Dict mapping cluster_id → list of board tuples
        """
        num_clusters = self.board_clusterer.num_clusters[street]
        cluster_boards = {i: [] for i in range(num_clusters)}

        # Assign each board to cluster
        for board in boards:
            cluster_id = self.board_clusterer.get_cluster(board)
            cluster_boards[cluster_id].append(board)

        # Sample from each cluster
        representatives = {}
        for cluster_id in range(num_clusters):
            boards_in_cluster = cluster_boards[cluster_id]

            if len(boards_in_cluster) == 0:
                # Cluster has no boards - skip it
                representatives[cluster_id] = []
            elif len(boards_in_cluster) <= num_samples_per_cluster:
                # Use all boards
                representatives[cluster_id] = boards_in_cluster
            else:
                # Sample without replacement
                indices = np.random.choice(
                    len(boards_in_cluster),
                    size=num_samples_per_cluster,
                    replace=False,
                )
                representatives[cluster_id] = [boards_in_cluster[i] for i in indices]

        return representatives

    def _compute_equity_matrix(
        self,
        cluster_representatives: Dict[int, list],
        street: Street,
    ) -> np.ndarray:
        """
        Compute equity for all (hand, board_cluster) pairs.

        Returns:
            equity_matrix: shape [169 hands, num_board_clusters]
        """
        num_clusters = self.board_clusterer.num_clusters[street]
        equity_matrix = np.zeros((169, num_clusters))

        # Get all 169 hand strings in order
        all_hands = PreflopHandMapper.get_all_hands()

        # Track statistics for validation
        empty_clusters = 0
        conflict_defaults = 0
        total_calculations = 0

        # For each preflop hand
        for hand_idx in range(NUM_PREFLOP_HANDS):
            hand_string = all_hands[hand_idx]

            # Get a concrete example of this hand
            hole_cards = self._get_example_hand(hand_string)

            # For each board cluster
            for cluster_id in range(num_clusters):
                boards = cluster_representatives[cluster_id]

                if len(boards) == 0:
                    # No boards in this cluster - use default equity
                    equity_matrix[hand_idx, cluster_id] = 0.5
                    empty_clusters += 1
                    logger.warning(
                        f"{street.name}: Empty cluster {cluster_id} for hand {hand_string}. "
                        f"Using default equity 0.5. This may indicate insufficient board samples."
                    )
                    continue

                # Compute average equity across representative boards
                equities = []
                conflicts = 0
                for board in boards:
                    # Skip if hole cards conflict with board
                    if self._cards_conflict(hole_cards, board):
                        conflicts += 1
                        continue

                    equity = self.equity_calculator.calculate_equity(hole_cards, board, street)
                    equities.append(equity)

                if len(equities) == 0:
                    # All boards conflicted - use default
                    equity_matrix[hand_idx, cluster_id] = 0.5
                    conflict_defaults += 1
                    logger.warning(
                        f"{street.name}: All {len(boards)} boards in cluster {cluster_id} "
                        f"conflicted with hand {hand_string} ({conflicts} conflicts). "
                        f"Using default equity 0.5. This should be rare."
                    )
                else:
                    equity_matrix[hand_idx, cluster_id] = np.mean(equities)
                    if conflicts > 0:
                        logger.debug(
                            f"{street.name}: Hand {hand_string}, cluster {cluster_id}: "
                            f"Skipped {conflicts}/{len(boards)} boards due to conflicts"
                        )

                total_calculations += 1

        # Log summary statistics
        total_cells = NUM_PREFLOP_HANDS * num_clusters
        logger.info(
            f"{street.name} equity matrix computed: "
            f"{total_cells} cells, {empty_clusters} empty clusters ({100*empty_clusters/total_cells:.2f}%), "
            f"{conflict_defaults} conflict defaults ({100*conflict_defaults/total_cells:.2f}%)"
        )

        if empty_clusters > 0 or conflict_defaults > 0:
            logger.warning(
                f"{street.name}: {empty_clusters + conflict_defaults} cells used default equity 0.5. "
                f"Consider increasing board samples or cluster representatives."
            )

        return equity_matrix

    def _cluster_equities(
        self,
        equity_matrix: np.ndarray,
        street: Street,
    ) -> np.ndarray:
        """
        Cluster (hand, board_cluster) pairs into buckets using K-means.

        Args:
            equity_matrix: shape [169 hands, num_board_clusters]

        Returns:
            bucket_assignments: shape [169, num_board_clusters]
                               Values are bucket IDs (0 to num_buckets-1)
        """
        # Flatten into (hand, board_cluster) pairs
        num_hands, num_board_clusters = equity_matrix.shape

        # Feature matrix: each row is [equity] for a (hand, board_cluster) pair
        # Shape: [169 × num_board_clusters, 1]
        features = equity_matrix.reshape(-1, 1)

        # Run K-means
        n_clusters = self.num_buckets[street]
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=KMEANS_RANDOM_STATE,
            max_iter=KMEANS_MAX_ITER,
            n_init=10,
        )
        labels = kmeans.fit_predict(features)

        # Reshape back to [169, num_board_clusters]
        bucket_assignments = labels.reshape(num_hands, num_board_clusters)

        # Store clusterer
        self.clusterers[street] = kmeans

        return bucket_assignments.astype(np.uint8)  # Save memory

    def get_bucket(
        self,
        hole_cards: Tuple[Card, Card],
        board: Tuple[Card, ...],
        street: Street,
    ) -> int:
        """
        Get bucket ID for a (hole_cards, board) pair.

        Args:
            hole_cards: Player's hole cards
            board: Community cards
            street: Current street

        Returns:
            Bucket ID (0 to num_buckets-1)

        Raises:
            ValueError: If not fitted for this street
        """
        if street not in self.bucket_assignments:
            raise ValueError(f"Bucketing not fitted for {street}")

        # Get hand index
        hand_idx = self.hand_mapper.get_hand_index(hole_cards)

        # Get board cluster
        board_cluster = self.board_clusterer.get_cluster(board)

        # Lookup bucket
        bucket = self.bucket_assignments[street][hand_idx, board_cluster]

        return int(bucket)

    def _get_example_hand(self, hand_string: str) -> Tuple[Card, Card]:
        """
        Get a concrete example of a hand string.

        e.g., "AKs" → (As, Ks)
              "72o" → (7h, 2d)
              "TT" → (Th, Td)
        """
        if len(hand_string) == 2:
            # Pair
            rank = hand_string[0]
            return (Card.new(f"{rank}h"), Card.new(f"{rank}d"))
        else:
            # Suited or offsuit
            high_rank = hand_string[0]
            low_rank = hand_string[1]
            suited = hand_string[2] == "s"

            if suited:
                return (Card.new(f"{high_rank}s"), Card.new(f"{low_rank}s"))
            else:
                return (Card.new(f"{high_rank}h"), Card.new(f"{low_rank}d"))

    def _cards_conflict(
        self,
        hole_cards: Tuple[Card, Card],
        board: Tuple[Card, ...],
    ) -> bool:
        """
        Check if hole cards conflict with board cards.

        Args:
            hole_cards: Two hole cards
            board: Board cards

        Returns:
            True if any cards are duplicated
        """
        all_cards = set(hole_cards) | set(board)
        return len(all_cards) < len(hole_cards) + len(board)

    def save(self, filepath: Path):
        """
        Save bucketing to disk.

        Saves:
        - Bucket assignments
        - K-means models
        - Configuration
        - Board clusterer state
        """
        data = {
            "num_buckets": self.num_buckets,
            "bucket_assignments": self.bucket_assignments,
            "clusterers": self.clusterers,
            "board_clusterer": self.board_clusterer,
            "fitted": self.fitted,
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved bucketing to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "EquityBucketing":
        """
        Load bucketing from disk.

        Args:
            filepath: Path to saved bucketing file

        Returns:
            EquityBucketing instance
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        bucketing = cls(
            num_buckets=data["num_buckets"],
            board_clusterer=data["board_clusterer"],
        )
        bucketing.bucket_assignments = data["bucket_assignments"]
        bucketing.clusterers = data["clusterers"]
        bucketing.fitted = data["fitted"]

        logger.info(f"Loaded bucketing from {filepath}")
        return bucketing

    def get_num_buckets(self, street: Street) -> int:
        """Get number of buckets for a street."""
        return self.num_buckets[street]

    def __str__(self) -> str:
        """String representation."""
        buckets_str = ", ".join([f"{s.name}: {n}" for s, n in self.num_buckets.items()])
        fitted_str = "fitted" if self.fitted else "not fitted"
        return f"EquityBucketing({buckets_str}) - {fitted_str}"
