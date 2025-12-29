"""Tests for poker actions."""

import pytest

from src.game.actions import (
    Action,
    ActionType,
    all_in,
    bet,
    call,
    check,
    fold,
    raises,
)


class TestActionType:
    """Tests for ActionType enum."""

    def test_is_aggressive(self):
        assert ActionType.BET.is_aggressive()
        assert ActionType.RAISE.is_aggressive()
        assert ActionType.ALL_IN.is_aggressive()
        assert not ActionType.FOLD.is_aggressive()
        assert not ActionType.CHECK.is_aggressive()
        assert not ActionType.CALL.is_aggressive()

    def test_is_passive(self):
        assert ActionType.CHECK.is_passive()
        assert ActionType.CALL.is_passive()
        assert not ActionType.BET.is_passive()
        assert not ActionType.RAISE.is_passive()
        assert not ActionType.FOLD.is_passive()
        assert not ActionType.ALL_IN.is_passive()


class TestAction:
    """Tests for Action dataclass."""

    def test_fold_action(self):
        action = fold()
        assert action.type == ActionType.FOLD
        assert action.amount == 0
        assert not action.is_aggressive()

    def test_check_action(self):
        action = check()
        assert action.type == ActionType.CHECK
        assert action.amount == 0
        assert action.is_passive()

    def test_call_action(self):
        action = call()
        assert action.type == ActionType.CALL
        assert action.amount == 0
        assert action.is_passive()

    def test_bet_action(self):
        action = bet(100)
        assert action.type == ActionType.BET
        assert action.amount == 100
        assert action.is_aggressive()

    def test_raise_action(self):
        action = raises(50)
        assert action.type == ActionType.RAISE
        assert action.amount == 50
        assert action.is_aggressive()

    def test_all_in_action(self):
        action = all_in(200)
        assert action.type == ActionType.ALL_IN
        assert action.amount == 200
        assert action.is_aggressive()

    def test_invalid_amount_negative(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            Action(ActionType.BET, -10)

    def test_fold_with_amount_raises(self):
        with pytest.raises(ValueError, match="must have amount=0"):
            Action(ActionType.FOLD, 10)

    def test_check_with_amount_raises(self):
        with pytest.raises(ValueError, match="must have amount=0"):
            Action(ActionType.CHECK, 10)

    def test_call_with_amount_raises(self):
        with pytest.raises(ValueError, match="must have amount=0"):
            Action(ActionType.CALL, 10)

    def test_bet_zero_amount_raises(self):
        with pytest.raises(ValueError, match="must have positive amount"):
            Action(ActionType.BET, 0)

    def test_raise_zero_amount_raises(self):
        with pytest.raises(ValueError, match="must have positive amount"):
            Action(ActionType.RAISE, 0)

    def test_all_in_zero_amount_raises(self):
        with pytest.raises(ValueError, match="must have positive amount"):
            Action(ActionType.ALL_IN, 0)

    def test_normalize_fold(self):
        assert fold().normalize(100) == "f"

    def test_normalize_check(self):
        assert check().normalize(100) == "x"

    def test_normalize_call(self):
        assert call().normalize(100) == "c"

    def test_normalize_bet(self):
        action = bet(50)
        assert action.normalize(100) == "b0.50"  # 50% of pot

    def test_normalize_raise(self):
        action = raises(75)
        assert action.normalize(100) == "r0.75"  # Raise 75% of pot

    def test_normalize_all_in(self):
        assert all_in(200).normalize(100) == "a"

    def test_action_immutable(self):
        action = bet(50)
        with pytest.raises(AttributeError):
            setattr(action, "amount", 100)

    def test_action_str(self):
        assert str(fold()) == "FOLD"
        assert str(bet(50)) == "BET(50)"
        assert str(all_in(200)) == "ALL_IN(200)"

    def test_action_repr(self):
        action = bet(50)
        assert repr(action) == "Action(type=ActionType.BET, amount=50)"
