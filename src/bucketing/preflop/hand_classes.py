"""
Preflop hand representation and mapping.

Maps hole card pairs to canonical 169-hand notation (e.g., AKs, 72o, TT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.bucketing.utils.card_utils import cards_have_same_suit, get_rank_char

if TYPE_CHECKING:
    from src.game.state import Card


class PreflopHandClasses:
    """
    Maps hole cards to canonical preflop hand representation.

    169 distinct hands:
    - 13 pairs: AA, KK, QQ, JJ, TT, 99, 88, 77, 66, 55, 44, 33, 22
    - 78 suited: AKs, AQs, ..., 32s
    - 78 offsuit: AKo, AQo, ..., 32o
    """

    RANKS = "AKQJT98765432"  # Ordered from high to low

    def __init__(self):
        """Initialize mapper."""
        # Precompute rank values for fast lookup
        self.rank_values = {rank: 14 - idx for idx, rank in enumerate(self.RANKS)}

    def get_hand_string(self, hole_cards: tuple[Card, Card]) -> str:
        """
        Get canonical hand string for hole cards.

        Args:
            hole_cards: Tuple of two cards

        Returns:
            Hand string (e.g., "AKs", "72o", "TT")

        Examples:
            (A♠, K♠) -> "AKs"
            (A♠, K♥) -> "AKo"
            (7♦, 7♣) -> "77"
            (2♥, 7♣) -> "72o"  # Lower rank first for offsuit
        """
        c1, c2 = hole_cards

        # Extract ranks
        r1 = get_rank_char(c1)
        r2 = get_rank_char(c2)

        # Check if suited
        suited = cards_have_same_suit(c1, c2)

        # For pairs, return like "AA", "KK", etc.
        if r1 == r2:
            return f"{r1}{r2}"

        # For non-pairs, put higher rank first
        if self.rank_values[r1] > self.rank_values[r2]:
            high, low = r1, r2
        else:
            high, low = r2, r1

        # Add suited/offsuit suffix
        suffix = "s" if suited else "o"

        return f"{high}{low}{suffix}"

    def get_hand_index(self, hole_cards: tuple[Card, Card]) -> int:
        """
        Get unique index (0-168) for a hand.

        Useful for array indexing in precomputation.

        Args:
            hole_cards: Tuple of two cards

        Returns:
            Index from 0 to 168

        Ordering:
            0-12: Pairs (AA=0, KK=1, ..., 22=12)
            13-90: Suited hands (AKs=13, AQs=14, ..., 32s=90)
            91-168: Offsuit hands (AKo=91, AQo=92, ..., 32o=168)
        """
        hand_str = self.get_hand_string(hole_cards)

        # Pairs
        if len(hand_str) == 2:
            rank = hand_str[0]
            return self.RANKS.index(rank)

        # Suited/offsuit
        high, low, suffix = hand_str[0], hand_str[1], hand_str[2]
        high_idx = self.RANKS.index(high)
        low_idx = self.RANKS.index(low)

        # Calculate position in suited/offsuit block
        # For each high card, there are (13 - high_idx - 1) lower cards
        position = sum(13 - i - 1 for i in range(high_idx)) + (low_idx - high_idx - 1)

        if suffix == "s":
            return 13 + position
        else:
            return 13 + 78 + position

    @staticmethod
    def get_all_hands() -> list[str]:
        """
        Get all 169 canonical hand strings.

        Returns:
            List of hand strings in order

        Example:
            ["AA", "KK", "QQ", ..., "22", "AKs", "AQs", ..., "32s", "AKo", "AQo", ..., "32o"]
        """
        hands = []

        # Pairs
        for rank in PreflopHandClasses.RANKS:
            hands.append(f"{rank}{rank}")

        # Suited
        for i, high in enumerate(PreflopHandClasses.RANKS):
            for low in PreflopHandClasses.RANKS[i + 1 :]:
                hands.append(f"{high}{low}s")

        # Offsuit
        for i, high in enumerate(PreflopHandClasses.RANKS):
            for low in PreflopHandClasses.RANKS[i + 1 :]:
                hands.append(f"{high}{low}o")

        return hands

    def __str__(self) -> str:
        """String representation."""
        return "PreflopHandClasses(169 hands)"


def get_preflop_hand_string(hole_cards: tuple[Card, Card]) -> str:
    """
    Convenience function to get hand string.

    Args:
        hole_cards: Tuple of two cards

    Returns:
        Canonical hand string
    """
    mapper = PreflopHandClasses()
    return mapper.get_hand_string(hole_cards)
