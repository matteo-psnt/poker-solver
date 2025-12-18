"""
Board clustering for efficient abstraction precomputation.

Groups similar board textures together to reduce the state space from millions
of boards to a manageable number of clusters (~200-1000 per street).
"""

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from src.abstraction.card_utils import get_rank_value, get_suit
from src.abstraction.constants import (
    DEFAULT_FLOP_BOARD_CLUSTERS,
    DEFAULT_RIVER_BOARD_CLUSTERS,
    DEFAULT_TURN_BOARD_CLUSTERS,
    KMEANS_MAX_ITER,
    KMEANS_RANDOM_STATE,
)
from src.game.state import Card, Street


class BoardClusterer:
    """
    Clusters poker boards by texture similarity.

    Uses K-means clustering on board feature vectors to group similar boards.
    This reduces the precomputation space dramatically:
    - Flop: 19,600 boards → ~200 clusters
    - Turn: 230,300 boards → ~500 clusters
    - River: 2.1M boards → ~1000 clusters
    """

    def __init__(self, num_clusters: dict = None):
        """
        Initialize board clusterer.

        Args:
            num_clusters: Dict mapping Street → num_clusters
                         Default: {FLOP: 200, TURN: 500, RIVER: 1000}
        """
        if num_clusters is None:
            num_clusters = {
                Street.FLOP: DEFAULT_FLOP_BOARD_CLUSTERS,
                Street.TURN: DEFAULT_TURN_BOARD_CLUSTERS,
                Street.RIVER: DEFAULT_RIVER_BOARD_CLUSTERS,
            }

        self.num_clusters = num_clusters
        self.clusterers = {}  # Street → fitted KMeans model
        self.fitted = False

    def extract_features(self, board: Tuple[Card, ...]) -> np.ndarray:
        """
        Extract feature vector from board.

        Features capture board texture:
        - Suit distribution (monotone, two-tone, rainbow)
        - Rank pairing (paired, trips, quads)
        - Connectivity (straight possibilities)
        - High card strength
        - Flush draw possibilities

        Args:
            board: Board cards (3 for flop, 4 for turn, 5 for river)

        Returns:
            Feature vector (numpy array)
        """
        features = []

        # 1. Suit distribution features
        suit_features = self._extract_suit_features(board)
        features.extend(suit_features)

        # 2. Rank pairing features
        rank_features = self._extract_rank_features(board)
        features.extend(rank_features)

        # 3. Connectivity features
        connectivity_features = self._extract_connectivity_features(board)
        features.extend(connectivity_features)

        # 4. High card strength
        high_card_features = self._extract_high_card_features(board)
        features.extend(high_card_features)

        return np.array(features, dtype=np.float32)

    def _extract_suit_features(self, board: Tuple[Card, ...]) -> List[float]:
        """
        Extract suit distribution features.

        Features:
        - Number of suits present (1-4)
        - Max cards of same suit (flush draws/made flushes)
        - Is monotone (all same suit)
        - Is two-tone (exactly 2 suits)

        Returns:
            List of features
        """
        # Count cards per suit
        suit_counts = {}
        for card in board:
            suit = get_suit(card)
            suit_counts[suit] = suit_counts.get(suit, 0) + 1

        num_suits = len(suit_counts)
        max_suit_count = max(suit_counts.values())

        features = [
            num_suits / 4.0,  # Normalized
            max_suit_count / len(board),  # Normalized
            1.0 if num_suits == 1 else 0.0,  # Monotone
            1.0 if num_suits == 2 else 0.0,  # Two-tone
        ]

        return features

    def _extract_rank_features(self, board: Tuple[Card, ...]) -> List[float]:
        """
        Extract rank pairing features.

        Features:
        - Number of unique ranks
        - Max cards of same rank (pairs, trips, quads)
        - Is paired
        - Is trips
        - Is two pair

        Returns:
            List of features
        """
        # Count cards per rank
        rank_counts = {}
        for card in board:
            rank = get_rank_value(card)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        num_ranks = len(rank_counts)
        max_rank_count = max(rank_counts.values())
        count_values = sorted(rank_counts.values(), reverse=True)

        features = [
            num_ranks / len(board),  # Normalized
            max_rank_count / 4.0,  # Normalized (max is quads)
            1.0 if max_rank_count >= 2 else 0.0,  # Paired
            1.0 if max_rank_count >= 3 else 0.0,  # Trips
            1.0
            if len(count_values) >= 2 and count_values[0] >= 2 and count_values[1] >= 2
            else 0.0,  # Two pair
        ]

        return features

    def _extract_connectivity_features(self, board: Tuple[Card, ...]) -> List[float]:
        """
        Extract connectivity features.

        Features:
        - Spread (max rank - min rank)
        - Has straight possibility
        - Number of connected cards
        - Has wheel possibility (A-5)

        Returns:
            List of features
        """
        ranks = [get_rank_value(card) for card in board]
        ranks_sorted = sorted(ranks)

        # Spread
        spread = ranks_sorted[-1] - ranks_sorted[0]

        # Check for straight possibility (any 5-card straight containing board cards)
        has_straight = self._has_straight_possibility(ranks_sorted)

        # Count connected cards (within 1 rank)
        connected_count = 0
        for i in range(len(ranks_sorted) - 1):
            if ranks_sorted[i + 1] - ranks_sorted[i] <= 1:
                connected_count += 1

        # Wheel possibility (contains A and low cards)
        has_wheel = 14 in ranks and min(ranks) <= 5

        features = [
            spread / 12.0,  # Normalized (max spread = 12, e.g., 2 to A)
            1.0 if has_straight else 0.0,
            connected_count / len(board),  # Normalized
            1.0 if has_wheel else 0.0,
        ]

        return features

    def _extract_high_card_features(self, board: Tuple[Card, ...]) -> List[float]:
        """
        Extract high card strength features.

        Features:
        - Highest rank on board
        - Average rank
        - Number of broadway cards (T+)

        Returns:
            List of features
        """
        ranks = [get_rank_value(card) for card in board]

        max_rank = max(ranks)
        avg_rank = np.mean(ranks)
        broadway_count = sum(1 for r in ranks if r >= 10)  # T, J, Q, K, A

        features = [
            max_rank / 14.0,  # Normalized (A = 14)
            avg_rank / 14.0,  # Normalized
            broadway_count / len(board),  # Normalized
        ]

        return features

    def _has_straight_possibility(self, ranks_sorted: List[int]) -> bool:
        """
        Check if board has straight possibility.

        Args:
            ranks_sorted: Sorted rank values

        Returns:
            True if any 5-card straight is possible
        """
        # For each rank, check if we can make a straight centered around it
        all_ranks = set(ranks_sorted)

        # Check wheel (A-2-3-4-5)
        if 14 in all_ranks and {2, 3, 4, 5} & all_ranks:
            return True

        # Check regular straights
        for rank in ranks_sorted:
            # Check if 5 consecutive ranks starting from (rank - 4) to rank
            for r in range(rank - 4, rank + 1):
                if r < 2:  # Below 2
                    continue
                # We only need 3+ cards from the board for straight possibility
                # (players can fill gaps with hole cards)

            # Simplified: if spread is <= 4, straight is possible
            if max(ranks_sorted) - min(ranks_sorted) <= 4:
                return True

        return False

    def fit(self, boards: List[Tuple[Card, ...]], street: Street):
        """
        Fit clusterer on sample boards for a given street.

        Args:
            boards: List of sample boards (should be representative)
            street: Which street these boards are from
        """
        # Extract features from all boards
        features = np.array([self.extract_features(board) for board in boards])

        # Fit K-means
        n_clusters = self.num_clusters[street]
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=KMEANS_RANDOM_STATE,
            max_iter=KMEANS_MAX_ITER,
            n_init=10,
        )
        kmeans.fit(features)

        self.clusterers[street] = kmeans
        self.fitted = True

    def get_cluster(self, board: Tuple[Card, ...]) -> int:
        """
        Get cluster ID for a board.

        Args:
            board: Board cards

        Returns:
            Cluster ID (0 to num_clusters-1)

        Raises:
            ValueError: If clusterer not fitted for this street
        """
        # Determine street from board size
        street = {
            3: Street.FLOP,
            4: Street.TURN,
            5: Street.RIVER,
        }.get(len(board))

        if street is None:
            raise ValueError(f"Invalid board size: {len(board)}")

        if street not in self.clusterers:
            raise ValueError(f"Clusterer not fitted for {street}")

        # Extract features and predict cluster
        features = self.extract_features(board).reshape(1, -1)
        cluster = self.clusterers[street].predict(features)[0]

        return int(cluster)

    def get_num_features(self) -> int:
        """Get number of features in feature vector."""
        # Count features from each extraction method
        # Suit: 4, Rank: 5, Connectivity: 4, High card: 3
        return 4 + 5 + 4 + 3

    def __str__(self) -> str:
        """String representation."""
        clusters_str = ", ".join([f"{s.name}: {n}" for s, n in self.num_clusters.items()])
        return f"BoardClusterer({clusters_str})"
