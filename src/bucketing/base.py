"""
Card abstraction interface.

Card abstraction reduces the space of possible hands by grouping similar hands
into buckets. This is essential for making poker tractable.

For the actual implementation, see equity_bucketing.py which provides
equity-based card abstraction using K-means clustering.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from src.game.state import Card, Street


class BucketingStrategy(ABC):
    """
    Abstract interface for card bucketing strategies.

    Implementations must map (hole_cards, board, street) -> bucket_id
    """

    @abstractmethod
    def get_bucket(
        self, hole_cards: Tuple[Card, Card], board: Tuple[Card, ...], street: Street
    ) -> int:
        """
        Get the bucket for a hand.

        Args:
            hole_cards: Player's two hole cards
            board: Community cards (empty for preflop)
            street: Current betting round

        Returns:
            Bucket ID (integer >= 0)
        """
        pass

    @abstractmethod
    def num_buckets(self, street: Street) -> int:
        """
        Get number of buckets for a street.

        Args:
            street: Betting round

        Returns:
            Number of buckets
        """
        pass

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}"
