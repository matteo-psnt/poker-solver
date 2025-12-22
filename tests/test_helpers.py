"""Test helpers for poker solver tests."""

from src.bucketing.base import BucketingStrategy


class DummyCardAbstraction(BucketingStrategy):
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
