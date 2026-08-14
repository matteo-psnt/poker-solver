"""
Hand evaluation using eval7.

This module wraps the eval7 evaluator to provide fast hand strength
calculations for Texas Hold'em.
"""

from functools import lru_cache

import eval7

from src.core.game.state import Card

_HAND_TYPE_TO_CLASS = {
    "Straight Flush": 1,
    "Quads": 2,  # eval7 uses "Quads" instead of "Four of a Kind"
    "Full House": 3,
    "Flush": 4,
    "Straight": 5,
    "Trips": 6,  # eval7 uses "Trips" instead of "Three of a Kind"
    "Two Pair": 7,
    "Pair": 8,  # eval7 uses "Pair" instead of "One Pair"
    "High Card": 9,
}
_CLASS_TO_HAND_TYPE = {value: key for key, value in _HAND_TYPE_TO_CLASS.items()}


class HandEvaluator:
    """
    Fast hand evaluator for Texas Hold'em using eval7.

    eval7 returns rank values where higher is better.
    This wrapper inverts the rank so lower values remain stronger hands.
    """

    # Max possible eval7 rank value (used for inversion)
    _MAX_RANK = 100000000  # Larger than any possible eval7 rank

    @staticmethod
    def _normalize_rank(rank: int) -> int:
        """Convert internal rank back to eval7's rank for classification."""
        # Our internal ranks are inverted (MAX - eval7_rank), convert back to eval7's rank
        return HandEvaluator._MAX_RANK - rank

    def evaluate(self, hole_cards: tuple[Card, Card], board: tuple[Card, ...]) -> int:
        """Hand strength as a rank, where LOWER is stronger."""
        if len(board) < 3:
            raise ValueError("Board must have at least 3 cards for evaluation")
        if len(hole_cards) != 2:
            raise ValueError("Must have exactly 2 hole cards")

        # Cards are already eval7.Card objects internally
        cards = [card.to_eval7() for card in board] + [card.to_eval7() for card in hole_cards]

        # eval7 uses "higher is better" semantics, invert to get "lower is better"
        eval7_rank = eval7.evaluate(cards)
        return self._MAX_RANK - eval7_rank

    def get_rank_class(self, rank: int) -> int:
        """Hand class: 1=Straight Flush, 2=Quads, ..., 9=High Card."""
        hand_type = eval7.handtype(self._normalize_rank(rank))
        try:
            return _HAND_TYPE_TO_CLASS[hand_type]
        except KeyError as exc:
            raise ValueError(f"Unknown hand type: {hand_type}") from exc

    def class_to_string(self, class_int: int) -> str:
        """Hand class as a name -- "Straight Flush", "Pair"."""
        try:
            return _CLASS_TO_HAND_TYPE[class_int]
        except KeyError as exc:
            raise ValueError(f"Unknown hand class: {class_int}") from exc

    def rank_to_string(self, rank: int) -> str:
        """A rank from :meth:`evaluate` as a human-readable description."""
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
        """Compare two hands on one board: -1 if the first wins, 1 if the second, 0 for a tie."""
        rank1 = self.evaluate(hole_cards1, board)
        rank2 = self.evaluate(hole_cards2, board)

        # Lower rank = better hand
        if rank1 == rank2:
            return 0
        return -1 if rank1 < rank2 else 1

    def hand_is_better(
        self,
        hole_cards1: tuple[Card, Card],
        hole_cards2: tuple[Card, Card],
        board: tuple[Card, ...],
    ) -> bool:
        """Whether the first hand beats the second on this board."""
        return self.compare_hands(hole_cards1, hole_cards2, board) == -1


@lru_cache(maxsize=1)
def get_evaluator() -> HandEvaluator:
    """Shared HandEvaluator instance (lazily constructed singleton)."""
    return HandEvaluator()
