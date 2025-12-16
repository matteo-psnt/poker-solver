"""
Card abstraction interface and implementations.

Card abstraction reduces the space of possible hands by grouping similar hands
into buckets. This is essential for making poker tractable.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from src.game.evaluator import get_evaluator
from src.game.state import Card, Street


class CardAbstraction(ABC):
    """
    Abstract interface for card bucketing strategies.

    Implementations must map (hole_cards, board, street) -> bucket_id
    """

    @abstractmethod
    def get_bucket(self, hole_cards: Tuple[Card, Card], board: Tuple[Card, ...], street: Street) -> int:
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


class RankBasedBucketing(CardAbstraction):
    """
    Simple rank-based card abstraction for testing.

    This uses hand rank categories (high card, pair, two pair, etc.)
    as buckets. Fast but inaccurate since it ignores card values and
    hand potential.

    Preflop buckets:
    - Premium pairs: AA-JJ
    - Medium pairs: TT-77
    - Small pairs: 66-22
    - High cards: AK, AQ
    - Medium: AJ, KQ
    - Weak: Everything else

    Postflop buckets:
    - Use hand class from evaluator (1-9)
    """

    def __init__(self):
        """Initialize rank-based bucketing."""
        self.evaluator = get_evaluator()

        # Define preflop buckets
        self.preflop_buckets = {
            Street.PREFLOP: 6,  # 6 preflop categories
            Street.FLOP: 9,      # Hand rank classes (1-9)
            Street.TURN: 9,
            Street.RIVER: 9,
        }

    def get_bucket(self, hole_cards: Tuple[Card, Card], board: Tuple[Card, ...], street: Street) -> int:
        """Get bucket for hand using simple rank categories."""
        if street.is_preflop():
            return self._get_preflop_bucket(hole_cards)
        else:
            return self._get_postflop_bucket(hole_cards, board)

    def _get_preflop_bucket(self, hole_cards: Tuple[Card, Card]) -> int:
        """
        Get preflop bucket based on hand strength.

        Buckets:
        0 = Premium pairs (AA-JJ)
        1 = Medium pairs (TT-77)
        2 = Small pairs (66-22)
        3 = High cards (AK, AQ)
        4 = Medium (AJ, KQ, suited connectors)
        5 = Weak (everything else)
        """
        # Convert to treys internal representation
        c1, c2 = hole_cards
        r1 = c1.card_int
        r2 = c2.card_int

        # Extract ranks (treys uses bit manipulation)
        # For simplicity, we'll use string representation
        c1_str = repr(c1)  # e.g., "As"
        c2_str = repr(c2)

        rank1 = c1_str[0]
        rank2 = c2_str[0]
        suit1 = c1_str[1]
        suit2 = c2_str[1]

        is_pair = rank1 == rank2
        is_suited = suit1 == suit2

        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                       '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

        val1 = rank_values.get(rank1, 0)
        val2 = rank_values.get(rank2, 0)

        if is_pair:
            if val1 >= 11:  # JJ+
                return 0  # Premium pairs
            elif val1 >= 7:  # 77-TT
                return 1  # Medium pairs
            else:  # 22-66
                return 2  # Small pairs

        # Not a pair - check high cards
        high = max(val1, val2)
        low = min(val1, val2)

        if high == 14:  # Ace
            if low >= 12:  # AK, AQ
                return 3
            elif low >= 11:  # AJ
                return 4
        elif high == 13 and low == 12:  # KQ
            return 4

        # Check for suited connectors
        if is_suited and abs(val1 - val2) <= 1:
            return 4

        return 5  # Weak

    def _get_postflop_bucket(self, hole_cards: Tuple[Card, Card], board: Tuple[Card, ...]) -> int:
        """
        Get postflop bucket using hand rank class.

        Returns hand class (0-8) where:
        0 = Straight Flush
        1 = Quads
        2 = Full House
        3 = Flush
        4 = Straight
        5 = Trips
        6 = Two Pair
        7 = Pair
        8 = High Card
        """
        if len(board) < 3:
            # Can't evaluate without at least 3 board cards
            return 8  # High card

        rank = self.evaluator.evaluate(hole_cards, board)
        hand_class = self.evaluator.get_rank_class(rank)

        # Hand class is 1-9, convert to 0-8
        return hand_class - 1

    def num_buckets(self, street: Street) -> int:
        """Get number of buckets for street."""
        return self.preflop_buckets[street]

    def __str__(self) -> str:
        return "RankBasedBucketing(preflop=6, postflop=9)"
