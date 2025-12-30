"""
Hand evaluation using eval7.

This module wraps the eval7 evaluator to provide fast hand strength
calculations for Texas Hold'em.
"""

import eval7
from treys import Card as TreysCard

from src.game.state import Card

_HAND_TYPE_TO_CLASS = {
    "Straight Flush": 1,
    "Four of a Kind": 2,
    "Full House": 3,
    "Flush": 4,
    "Straight": 5,
    "Three of a Kind": 6,
    "Two Pair": 7,
    "One Pair": 8,
    "High Card": 9,
}
_CLASS_TO_HAND_TYPE = {value: key for key, value in _HAND_TYPE_TO_CLASS.items()}
_EVAL7_CARD_CACHE: dict[int, eval7.Card] | None = None


def _get_eval7_card_cache() -> dict[int, eval7.Card]:
    global _EVAL7_CARD_CACHE
    if _EVAL7_CARD_CACHE is None:
        cache: dict[int, eval7.Card] = {}
        for card in Card.get_full_deck():
            card_str = TreysCard.int_to_str(card.card_int)
            cache[card.card_int] = eval7.Card(card_str)
        _EVAL7_CARD_CACHE = cache
    return _EVAL7_CARD_CACHE


class HandEvaluator:
    """
    Fast hand evaluator for Texas Hold'em using eval7.

    eval7 returns higher rank values = stronger hands.
    This wrapper inverts the rank so lower values remain stronger hands.
    """

    def __init__(self):
        self._card_cache = _get_eval7_card_cache()

    @staticmethod
    def _normalize_rank(rank: int) -> int:
        """Convert internal rank to eval7's positive rank for classification."""
        return -rank if rank < 0 else rank

    def evaluate(self, hole_cards: tuple[Card, Card], board: tuple[Card, ...]) -> int:
        """
        Evaluate hand strength.

        Args:
            hole_cards: Player's two hole cards
            board: Community cards (3-5 cards)

        Returns:
            Hand rank where lower values are stronger
        """
        if len(board) < 3:
            raise ValueError("Board must have at least 3 cards for evaluation")
        if len(hole_cards) != 2:
            raise ValueError("Must have exactly 2 hole cards")

        cards = [self._card_cache[card.card_int] for card in board]
        cards.extend(self._card_cache[card.card_int] for card in hole_cards)

        # Invert rank to preserve "lower is better" semantics.
        return -eval7.evaluate(cards)

    def get_rank_class(self, rank: int) -> int:
        """
        Get hand class (1=Straight Flush, 2=Quads, ..., 9=High Card).

        Args:
            rank: Hand rank from evaluate()

        Returns:
            Hand class (1-9)
        """
        hand_type = eval7.handtype(self._normalize_rank(rank))
        try:
            return _HAND_TYPE_TO_CLASS[hand_type]
        except KeyError as exc:
            raise ValueError(f"Unknown hand type: {hand_type}") from exc

    def class_to_string(self, class_int: int) -> str:
        """
        Convert hand class to human-readable string.

        Args:
            class_int: Hand class (1-9)

        Returns:
            Hand class name (e.g., "Straight Flush", "Pair")
        """
        try:
            return _CLASS_TO_HAND_TYPE[class_int]
        except KeyError as exc:
            raise ValueError(f"Unknown hand class: {class_int}") from exc

    def rank_to_string(self, rank: int) -> str:
        """
        Get human-readable hand description.

        Args:
            rank: Hand rank from evaluate()

        Returns:
            Hand description with class
        """
        hand_class = self.get_rank_class(rank)
        class_str = self.class_to_string(hand_class)
        rank_display = self._normalize_rank(rank)
        return f"{class_str} (rank {rank_display})"

    def compare_hands(
        self,
        hole_cards1: tuple[Card, Card],
        hole_cards2: tuple[Card, Card],
        board: tuple[Card, ...],
    ) -> int:
        """
        Compare two hands on the same board.

        Args:
            hole_cards1: Player 1's hole cards
            hole_cards2: Player 2's hole cards
            board: Community cards

        Returns:
            -1 if hand1 wins, 1 if hand2 wins, 0 if tie
        """
        rank1 = self.evaluate(hole_cards1, board)
        rank2 = self.evaluate(hole_cards2, board)

        if rank1 < rank2:  # Lower rank = better
            return -1
        elif rank1 > rank2:
            return 1
        else:
            return 0

    def hand_is_better(
        self,
        hole_cards1: tuple[Card, Card],
        hole_cards2: tuple[Card, Card],
        board: tuple[Card, ...],
    ) -> bool:
        """
        Check if hand1 beats hand2.

        Args:
            hole_cards1: First hand
            hole_cards2: Second hand
            board: Community cards

        Returns:
            True if hand1 wins
        """
        return self.compare_hands(hole_cards1, hole_cards2, board) == -1


# Global evaluator instance for efficiency
_evaluator_instance = None


def get_evaluator() -> HandEvaluator:
    """
    Get the global evaluator instance (singleton pattern).

    Returns:
        Shared HandEvaluator instance
    """
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = HandEvaluator()
    return _evaluator_instance
