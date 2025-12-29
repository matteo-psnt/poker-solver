"""
Hand evaluation using the treys library.

This module wraps the treys poker evaluator to provide fast hand strength
calculations for Texas Hold'em.
"""

from treys import Evaluator as TreysEvaluator

from src.game.state import Card


class HandEvaluator:
    """
    Fast hand evaluator for Texas Hold'em using treys.

    The treys library uses a perfect hash algorithm for fast evaluation.
    Lower rank values = stronger hands (1 = Royal Flush, 7462 = worst hand).
    """

    def __init__(self):
        self.evaluator = TreysEvaluator()

    def evaluate(self, hole_cards: tuple[Card, Card], board: tuple[Card, ...]) -> int:
        """
        Evaluate hand strength.

        Args:
            hole_cards: Player's two hole cards
            board: Community cards (3-5 cards)

        Returns:
            Hand rank (1 = best, 7462 = worst)
        """
        if len(board) < 3:
            raise ValueError("Board must have at least 3 cards for evaluation")
        if len(hole_cards) != 2:
            raise ValueError("Must have exactly 2 hole cards")

        hole_ints = [card.card_int for card in hole_cards]
        board_ints = [card.card_int for card in board]

        return self.evaluator.evaluate(board_ints, hole_ints)

    def get_rank_class(self, rank: int) -> int:
        """
        Get hand class (1=Straight Flush, 2=Quads, ..., 9=High Card).

        Args:
            rank: Hand rank from evaluate()

        Returns:
            Hand class (1-9)
        """
        return self.evaluator.get_rank_class(rank)

    def class_to_string(self, class_int: int) -> str:
        """
        Convert hand class to human-readable string.

        Args:
            class_int: Hand class (1-9)

        Returns:
            Hand class name (e.g., "Straight Flush", "Pair")
        """
        return self.evaluator.class_to_string(class_int)

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
        return f"{class_str} (rank {rank})"

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
