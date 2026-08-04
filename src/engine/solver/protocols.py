"""Solver-facing protocols shared across engine modules."""

from __future__ import annotations

from typing import Protocol

from src.core.game.state import Card, Street


class BucketingStrategy(Protocol):
    """Structural interface for card bucketing used by the solver."""

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        """Map hole cards + board context to an abstract bucket id."""

    def num_buckets(self, street: Street) -> int:
        """Return number of buckets for a specific street."""
