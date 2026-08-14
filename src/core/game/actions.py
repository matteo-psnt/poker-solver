"""
Poker action representations and types.

This module defines the action types available in Heads-Up No-Limit Hold'em
and provides data structures for representing player actions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache


class ActionType(Enum):
    """Types of actions available in HUNLHE."""

    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def is_aggressive(self) -> bool:
        """Check if action is aggressive (bet, raise, all-in)."""
        return self in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)

    def is_passive(self) -> bool:
        """Check if action is passive (check, call)."""
        return self in (ActionType.CHECK, ActionType.CALL)


@dataclass(frozen=True)
class Action:
    """Immutable representation of a poker action.

    ``amount`` means a DIFFERENT thing per type, which is the trap:

        FOLD/CHECK/CALL  must be 0 -- a call's size comes from ``state.to_call``
        BET              total chips bet
        RAISE            chips ABOVE the call, so the total added is
                         ``state.to_call + amount``
        ALL_IN           the player's whole remaining stack, an absolute number
                         rather than one relative to any bet

    An ALL_IN at or above ``to_call`` is a call plus whatever raise is left over;
    below it, it is an all-in call for less and the uncalled part is returned. This
    matters for InfoSet key normalization -- a player with 30 facing 50 produces
    ALL_IN(30), not a fold.
    """

    type: ActionType
    amount: int = 0

    def __post_init__(self):
        """Validate action consistency."""
        if self.amount < 0:
            raise ValueError(f"Action amount cannot be negative: {self.amount}")

        # A wrong amount is indistinguishable from a legal action downstream.
        if self.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL) and self.amount != 0:
            raise ValueError(f"{self.type} must have amount=0, got {self.amount}")

        if self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN) and self.amount <= 0:
            raise ValueError(f"{self.type} must have positive amount, got {self.amount}")

    def is_aggressive(self) -> bool:
        """Check if action is aggressive (bet, raise, all-in)."""
        return self.type.is_aggressive()

    def is_passive(self) -> bool:
        """Check if action is passive (check, call)."""
        return self.type.is_passive()

    def normalize(self, pot: int) -> str:
        """This action as an infoset-key string: "f", "c", "b0.75", "r2.5".

        Bet sizes are normalized against ``pot``.
        """
        # Fast path for common non-amount actions (no computation needed)
        if self.type == ActionType.FOLD:
            return "f"
        if self.type == ActionType.CHECK:
            return "x"
        if self.type == ActionType.CALL:
            return "c"
        if self.type == ActionType.ALL_IN:
            return "a"

        # For amount-based actions, use LRU cache
        return self._normalize_amount_action(self.type.value, self.amount, pot)

    @staticmethod
    @lru_cache(maxsize=10000)
    def _normalize_amount_action(action_type_value: int, amount: int, pot: int) -> str:
        """Normalize amount-based actions using a bounded LRU cache."""
        pot_frac = amount / pot if pot > 0 else 0
        if action_type_value == ActionType.BET.value:
            return f"b{pot_frac:.2f}"
        if action_type_value == ActionType.RAISE.value:
            return f"r{pot_frac:.2f}"
        raise ValueError(f"Unknown action type value: {action_type_value}")

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.amount > 0:
            return f"{self.type.name}({self.amount})"
        return self.type.name

    def __repr__(self) -> str:
        return f"Action(type=ActionType.{self.type.name}, amount={self.amount})"


# Interned action constructors. Actions are immutable value objects drawn from
# a small grid, and get_legal_actions() creates them once per node visit — the
# factories return shared instances so construction/validation runs once per
# distinct action instead of millions of times per training run.
_FOLD = Action(ActionType.FOLD, 0)
_CHECK = Action(ActionType.CHECK, 0)
_CALL = Action(ActionType.CALL, 0)


def fold() -> Action:
    """The fold action (interned)."""
    return _FOLD


def check() -> Action:
    """The check action (interned)."""
    return _CHECK


def call() -> Action:
    """The call action (interned)."""
    return _CALL


@lru_cache(maxsize=4096)
def bet(amount: int) -> Action:
    """Create a bet action (interned per amount)."""
    return Action(ActionType.BET, amount)


@lru_cache(maxsize=4096)
def raises(amount: int) -> Action:
    """Create a raise action (interned per amount)."""
    return Action(ActionType.RAISE, amount)


@lru_cache(maxsize=4096)
def all_in(amount: int) -> Action:
    """Create an all-in action (interned per amount)."""
    return Action(ActionType.ALL_IN, amount)
