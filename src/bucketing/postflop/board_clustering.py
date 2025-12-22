"""
Board clustering for public-state abstraction.

Clusters canonical boards by strategic texture to reduce the state space.
This is the critical first step before equity computation - we cluster boards
FIRST, then only compute equity for representative boards.

Key insight: We don't need equity for every possible board. We cluster boards
by texture (connectivity, suits, pairing), then compute equity for a small
number of representatives per cluster.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from treys import Card as TreysCard

from src.bucketing.postflop.suit_isomorphism import CanonicalCard
from src.game.state import Card, Street


@dataclass
class BoardCluster:
    """Information about a board cluster."""

    cluster_id: int
    street: Street
    num_boards: int  # Number of canonical boards in this cluster
    representative_boards: List[Tuple[Card, ...]]  # Small set of representative boards
    canonical_representatives: List[Tuple[CanonicalCard, ...]]  # Canonical forms


class BoardClusterer:
    """
    Clusters canonical boards by strategic texture.

    This is the foundation of tractable abstraction - we cluster the public state
    (boards) first, BEFORE computing equity. This reduces the computation from
    O(all_boards × all_hands) to O(num_clusters × representatives_per_cluster × all_hands).
    """

    def __init__(self, num_clusters: Dict[Street, int]):
        """
        Initialize board clusterer.

        Args:
            num_clusters: Number of clusters per street
                         Example: {FLOP: 50, TURN: 100, RIVER: 200}
        """
        self.num_clusters = num_clusters

        # Fitted KMeans models per street
        self._kmeans: Dict[Street, KMeans] = {}

        # Cluster assignments: canonical_board_id -> cluster_id
        self._board_to_cluster: Dict[Street, Dict[int, int]] = {}

        # Cluster info
        self._clusters: Dict[Street, Dict[int, BoardCluster]] = {}

        self.fitted = False

    def extract_board_features(self, board: Tuple[Card, ...]) -> np.ndarray:
        """
        Extract strategic texture features from a board.

        Features capture:
        - Suit distribution (monotone, two-tone, rainbow, four-flush)
        - Rank pairing (paired, trips, quads, two-pair)
        - Connectivity (straight possibilities, gaps)
        - High card strength

        These features are suit-isomorphic by construction.

        Args:
            board: Board cards (3/4/5 cards)

        Returns:
            Feature vector (numpy array)
        """
        features: list[float] = []

        # === Suit Distribution Features ===
        suit_counts: dict[int, int] = {}
        for card in board:
            suit = TreysCard.get_suit_int(card.card_int)
            suit_counts[suit] = suit_counts.get(suit, 0) + 1

        sorted_suit_counts = sorted(suit_counts.values(), reverse=True)

        # Pad to length 4
        while len(sorted_suit_counts) < 4:
            sorted_suit_counts.append(0)

        features.extend(sorted_suit_counts[:4])  # [max_suit, 2nd_suit, 3rd_suit, 4th_suit]

        # Suit distribution indicators
        features.append(1.0 if len(suit_counts) == 1 else 0.0)  # Monotone
        features.append(1.0 if len(suit_counts) == 2 else 0.0)  # Two-tone
        features.append(
            1.0 if max(suit_counts.values()) >= 3 else 0.0
        )  # Three of suit (flush draw)

        # === Rank Pairing Features ===
        rank_counts: dict[int, int] = {}
        for card in board:
            rank = TreysCard.get_rank_int(card.card_int)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        sorted_rank_counts = sorted(rank_counts.values(), reverse=True)

        # Pad to length 5
        while len(sorted_rank_counts) < 5:
            sorted_rank_counts.append(0)

        features.extend(sorted_rank_counts[:3])  # [max_rank, 2nd_rank, 3rd_rank]

        # Pairing indicators
        features.append(1.0 if max(rank_counts.values()) == 2 else 0.0)  # Paired
        features.append(1.0 if max(rank_counts.values()) == 3 else 0.0)  # Trips
        features.append(1.0 if max(rank_counts.values()) == 4 else 0.0)  # Quads
        features.append(1.0 if list(sorted_rank_counts[:2]) == [2, 2] else 0.0)  # Two pair

        # === Connectivity Features ===
        ranks_sorted = sorted(
            [TreysCard.get_rank_int(card.card_int) for card in board], reverse=True
        )

        # Gap analysis (lower gaps = more connected)
        gaps = []
        for i in range(len(ranks_sorted) - 1):
            gap = ranks_sorted[i] - ranks_sorted[i + 1] - 1
            gaps.append(gap)

        # Pad gaps
        while len(gaps) < 4:
            gaps.append(10)  # Large gap for padding

        features.extend(gaps[:4])  # Gap between each pair of cards

        # Straight draw indicators
        max_gap = max(gaps) if gaps else 10
        features.append(1.0 if max_gap <= 4 else 0.0)  # Connected (potential straight)
        features.append(
            1.0 if max(ranks_sorted) - min(ranks_sorted) <= 4 else 0.0
        )  # Straight possible

        # === High Card Strength ===
        # Normalize ranks to [0, 1] (deuce=0, ace=12)
        normalized_ranks = [r / 12.0 for r in ranks_sorted]

        # Pad to 5 cards
        while len(normalized_ranks) < 5:
            normalized_ranks.append(0.0)

        features.extend(normalized_ranks[:5])  # Top 5 cards (river will have all 5)

        # === Aggregate Strength ===
        features.append(np.mean(normalized_ranks))  # Average board strength
        features.append(np.max(normalized_ranks))  # Highest card

        return np.array(features, dtype=np.float32)

    def fit(
        self,
        canonical_boards: List[Tuple[CanonicalCard, ...]],
        representative_boards: List[Tuple[Card, ...]],
        street: Street,
        representatives_per_cluster: int = 3,
    ) -> None:
        """
        Fit clusterer on canonical boards.

        Args:
            canonical_boards: List of canonical board representations
            representative_boards: Corresponding real board representations (for feature extraction)
            street: Which street
            representatives_per_cluster: How many representative boards to keep per cluster
        """
        if street not in self.num_clusters:
            raise ValueError(f"No cluster count specified for {street.name}")

        # Extract features for all boards
        features = np.array([self.extract_board_features(board) for board in representative_boards])

        # Cluster
        n_clusters = min(self.num_clusters[street], len(canonical_boards))
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(features)

        self._kmeans[street] = kmeans

        # Build cluster info
        from src.bucketing.postflop.suit_isomorphism import get_canonical_board_id

        self._board_to_cluster[street] = {}
        self._clusters[street] = {}

        for cluster_id in range(n_clusters):
            # Find all boards in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            # Select representatives (boards closest to cluster center)
            cluster_features = features[cluster_indices]
            center = kmeans.cluster_centers_[cluster_id]

            # Compute distances to center
            distances = np.linalg.norm(cluster_features - center, axis=1)

            # Select top N closest as representatives
            n_reps = min(representatives_per_cluster, len(cluster_indices))
            closest_indices = cluster_indices[np.argsort(distances)[:n_reps]]

            reps = [representative_boards[i] for i in closest_indices]
            canonical_reps = [canonical_boards[i] for i in closest_indices]

            # Create cluster
            cluster = BoardCluster(
                cluster_id=cluster_id,
                street=street,
                num_boards=len(cluster_indices),
                representative_boards=reps,
                canonical_representatives=canonical_reps,
            )
            self._clusters[street][cluster_id] = cluster

            # Map all boards in cluster
            for idx in cluster_indices:
                board_id = get_canonical_board_id(canonical_boards[idx])
                self._board_to_cluster[street][board_id] = cluster_id

        self.fitted = True

    def predict(self, board: Tuple[Card, ...], street: Street) -> int:
        """
        Predict cluster for ANY board using feature-based inference.

        This is isomorphism-invariant because extract_board_features()
        operates on suit distributions and rank patterns, not absolute suits.
        Works for any board, not just those in the training set.

        Args:
            board: Board cards (tuple of Card objects)
            street: Which street

        Returns:
            Cluster ID

        Raises:
            ValueError: If clusterer not fitted for this street
        """
        if street not in self._kmeans:
            raise ValueError(f"Clusterer not fitted for {street.name}")

        # Extract features and predict using fitted KMeans
        features = self.extract_board_features(board).reshape(1, -1)
        cluster_id = int(self._kmeans[street].predict(features)[0])

        return cluster_id

    def get_cluster(self, board_id: int, street: Street) -> Optional[int]:
        """Get cluster for a canonical board ID."""
        return self._board_to_cluster.get(street, {}).get(board_id)

    def get_cluster_info(self, cluster_id: int, street: Street) -> Optional[BoardCluster]:
        """Get info about a cluster."""
        return self._clusters.get(street, {}).get(cluster_id)

    def get_all_clusters(self, street: Street) -> List[BoardCluster]:
        """Get all clusters for a street."""
        return list(self._clusters.get(street, {}).values())
