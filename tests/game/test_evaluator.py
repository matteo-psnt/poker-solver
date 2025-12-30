"""
Tests for hand evaluator.

Tests the HandEvaluator class and its methods.
"""

import pytest

from src.game.evaluator import HandEvaluator, get_evaluator
from src.game.state import Card


class TestHandEvaluator:
    """Test hand evaluation functionality."""

    def test_create_evaluator(self):
        """Test creating evaluator."""
        evaluator = HandEvaluator()
        assert evaluator is not None
        assert evaluator._card_cache is not None

    def test_evaluate_high_card(self):
        """Test evaluating a high card hand."""
        evaluator = HandEvaluator()

        hole_cards = (Card.new("As"), Card.new("Kh"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        rank = evaluator.evaluate(hole_cards, board)
        # High card should be high rank number (weak hand)
        assert rank > 1000

    def test_evaluate_pair(self):
        """Test evaluating a pair."""
        evaluator = HandEvaluator()

        hole_cards = (Card.new("Ah"), Card.new("Ac"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        rank = evaluator.evaluate(hole_cards, board)
        # Pair should be better than high card
        assert 100 < rank < 7000

    def test_evaluate_flush(self):
        """Test evaluating a flush."""
        evaluator = HandEvaluator()

        hole_cards = (Card.new("Ah"), Card.new("Kh"))
        board = (Card.new("2h"), Card.new("7h"), Card.new("9h"))

        rank = evaluator.evaluate(hole_cards, board)
        # Flush should be strong (low rank number)
        assert rank < 2000

    def test_evaluate_board_too_small(self):
        """Test error when board has < 3 cards."""
        evaluator = HandEvaluator()

        hole_cards = (Card.new("Ah"), Card.new("Kh"))
        board = (Card.new("2c"), Card.new("7d"))  # Only 2 cards

        with pytest.raises(ValueError, match="at least 3 cards"):
            evaluator.evaluate(hole_cards, board)

    def test_evaluate_wrong_hole_cards_count(self):
        """Test error when hole cards != 2."""
        evaluator = HandEvaluator()

        # Only 1 hole card
        with pytest.raises(ValueError, match="exactly 2 hole cards"):
            evaluator.evaluate((Card.new("Ah"),), (Card.new("2c"), Card.new("7d"), Card.new("9h")))  # type: ignore

    def test_compare_hands_hand1_wins(self):
        """Test comparing hands when hand1 is better."""
        evaluator = HandEvaluator()

        # Pair of aces vs pair of kings
        hole_cards1 = (Card.new("Ah"), Card.new("Ac"))
        hole_cards2 = (Card.new("Kh"), Card.new("Kc"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        result = evaluator.compare_hands(hole_cards1, hole_cards2, board)
        assert result == -1  # Hand1 wins

    def test_compare_hands_hand2_wins(self):
        """Test comparing hands when hand2 is better."""
        evaluator = HandEvaluator()

        # High card vs pair
        hole_cards1 = (Card.new("Ah"), Card.new("Kh"))
        hole_cards2 = (Card.new("2h"), Card.new("2c"))
        board = (Card.new("5c"), Card.new("7d"), Card.new("9h"))

        result = evaluator.compare_hands(hole_cards1, hole_cards2, board)
        assert result == 1  # Hand2 wins

    def test_compare_hands_tie(self):
        """Test comparing hands when they tie."""
        evaluator = HandEvaluator()

        # Both play the board (no pair, same high cards, different suits to avoid flush)
        hole_cards1 = (Card.new("2h"), Card.new("3c"))
        hole_cards2 = (Card.new("2d"), Card.new("3s"))
        board = (Card.new("Ah"), Card.new("Kd"), Card.new("Qh"), Card.new("Jc"), Card.new("9s"))

        result = evaluator.compare_hands(hole_cards1, hole_cards2, board)
        assert result == 0  # Tie

    def test_hand_is_better_true(self):
        """Test hand_is_better returns True for better hand."""
        evaluator = HandEvaluator()

        hole_cards1 = (Card.new("Ah"), Card.new("Ac"))
        hole_cards2 = (Card.new("Kh"), Card.new("Kc"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        assert evaluator.hand_is_better(hole_cards1, hole_cards2, board) is True

    def test_hand_is_better_false(self):
        """Test hand_is_better returns False for worse hand."""
        evaluator = HandEvaluator()

        hole_cards1 = (Card.new("Kh"), Card.new("Kc"))
        hole_cards2 = (Card.new("Ah"), Card.new("Ac"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        assert evaluator.hand_is_better(hole_cards1, hole_cards2, board) is False

    def test_get_rank_class(self):
        """Test getting rank class."""
        evaluator = HandEvaluator()

        hole_cards = (Card.new("Ah"), Card.new("Ac"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        rank = evaluator.evaluate(hole_cards, board)
        rank_class = evaluator.get_rank_class(rank)

        # Should be a valid class (1-9)
        assert 1 <= rank_class <= 9

    def test_class_to_string(self):
        """Test converting class to string."""
        evaluator = HandEvaluator()

        hole_cards = (Card.new("Ah"), Card.new("Ac"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        rank = evaluator.evaluate(hole_cards, board)
        rank_class = evaluator.get_rank_class(rank)
        class_str = evaluator.class_to_string(rank_class)

        # Should get a string description
        assert isinstance(class_str, str)
        assert len(class_str) > 0

    def test_rank_to_string(self):
        """Test converting rank to string."""
        evaluator = HandEvaluator()

        hole_cards = (Card.new("Ah"), Card.new("Ac"))
        board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))

        rank = evaluator.evaluate(hole_cards, board)
        rank_str = evaluator.rank_to_string(rank)

        # Should include rank number in string
        assert str(rank) in rank_str
        assert "(" in rank_str and ")" in rank_str

    def test_get_evaluator_singleton(self):
        """Test that get_evaluator returns same instance."""
        eval1 = get_evaluator()
        eval2 = get_evaluator()

        assert eval1 is eval2  # Should be same object
