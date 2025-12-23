"""Test helpers for poker solver tests."""

from pathlib import Path

import numpy as np

from src.abstraction.core.card_abstraction import CardAbstraction
from src.abstraction.equity.board_clustering import BoardClusterer
from src.abstraction.equity.equity_bucketing import EquityBucketing
from src.game.state import Street


class DummyCardAbstraction(CardAbstraction):
    """
    Minimal card abstraction for testing.

    All hands map to bucket 0 (single bucket per street).
    Used when card abstraction logic isn't being tested.
    """

    def get_bucket(self, hole_cards, board, street):
        """All hands map to bucket 0."""
        return 0

    def num_buckets(self, street):
        """Single bucket per street."""
        return 1


class DummyKMeans:
    """Mock KMeans clusterer for testing."""

    def __init__(self):
        self.cluster_centers_ = np.array([[0.0] * 10])  # Dummy centers
        self.labels_ = np.array([0])

    def predict(self, x):
        return np.zeros(len(x), dtype=np.int32)


def dummy_predict(x):
    """Dummy predict function for BoardClusterer."""
    if hasattr(x, "__len__"):
        return np.zeros(len(x), dtype=np.int32)
    return np.array([0], dtype=np.int32)


def create_minimal_bucketing(save_path: Path = None) -> EquityBucketing:
    """
    Create a minimal EquityBucketing instance for testing.

    Creates a bucketing with minimal clusters and buckets, properly fitted
    so it can be used in tests without errors.

    Args:
        save_path: Optional path to save the bucketing

    Returns:
        Fitted EquityBucketing instance
    """
    # Create bucketing with minimal sizes
    num_buckets = {Street.FLOP: 2, Street.TURN: 2, Street.RIVER: 2}
    num_board_clusters = {Street.FLOP: 1, Street.TURN: 1, Street.RIVER: 1}

    bucketing = EquityBucketing(num_buckets=num_buckets, num_board_clusters=num_board_clusters)

    # Mark as fitted and create dummy data structures
    bucketing.fitted = True

    # Create dummy bucket assignments (169 hands x num_clusters per street)
    bucketing.bucket_assignments = {}
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        n_clusters = num_board_clusters[street]
        # Simple assignments - all hands go to bucket 0 or 1
        bucketing.bucket_assignments[street] = np.zeros((169, n_clusters), dtype=np.int32)

    # Mock board clusterers - simplest possible implementation
    # Just mark them as fitted with minimal state
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        clusterer = bucketing.board_clusterer.clusterers.get(street)
        if clusterer is None:
            clusterer = BoardClusterer(num_clusters={street: num_board_clusters[street]})
            bucketing.board_clusterer.clusterers[street] = clusterer

        # Mock the scikit-learn model with minimal state
        clusterer.fitted = True
        clusterer.model = DummyKMeans()
        # Add predict method directly to clusterer for compatibility
        clusterer.predict = dummy_predict

    if save_path:
        bucketing.save(save_path)

    return bucketing

    if save_path:
        bucketing.save(save_path)

    return bucketing
