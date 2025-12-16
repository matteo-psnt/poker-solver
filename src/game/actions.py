"""
Poker action representations and types.

This module defines the action types available in Heads-Up No-Limit Hold'em
and provides data structures for representing player actions.
"""

from dataclasses import dataclass
from enum import Enum, auto


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
    """
    Immutable representation of a poker action.

    Attributes:
        type: The type of action (fold, check, call, bet, raise, all-in)
        amount: Chips to add to pot (0 for fold/check/call)
    """

    type: ActionType
    amount: int = 0

    def __post_init__(self):
        """Validate action consistency."""
        if self.amount < 0:
            raise ValueError(f"Action amount cannot be negative: {self.amount}")

        # Fold, check, call should have 0 amount
        if self.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            if self.amount != 0:
                raise ValueError(f"{self.type} must have amount=0, got {self.amount}")

        # Bet, raise, all-in must have positive amount
        if self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            if self.amount <= 0:
                raise ValueError(f"{self.type} must have positive amount, got {self.amount}")

    def is_aggressive(self) -> bool:
        """Check if action is aggressive (bet, raise, all-in)."""
        return self.type.is_aggressive()

    def is_passive(self) -> bool:
        """Check if action is passive (check, call)."""
        return self.type.is_passive()

    def normalize(self, pot: int) -> str:
        """
        Normalize action to a string representation for infoset keys.

        Args:
            pot: Current pot size for normalizing bet sizes

        Returns:
            Normalized action string (e.g., "f", "c", "b0.75", "r2.5")
        """
        if self.type == ActionType.FOLD:
            return "f"
        elif self.type == ActionType.CHECK:
            return "x"
        elif self.type == ActionType.CALL:
            return "c"
        elif self.type == ActionType.BET:
            pot_frac = self.amount / pot if pot > 0 else 0
            return f"b{pot_frac:.2f}"
        elif self.type == ActionType.RAISE:
            # Normalize as pot-sized unit (e.g., raise to 2.5x pot)
            pot_frac = self.amount / pot if pot > 0 else 0
            return f"r{pot_frac:.2f}"
        elif self.type == ActionType.ALL_IN:
            return "a"
        else:
            raise ValueError(f"Unknown action type: {self.type}")

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.amount > 0:
            return f"{self.type.name}({self.amount})"
        return self.type.name

    def __repr__(self) -> str:
        return f"Action(type=ActionType.{self.type.name}, amount={self.amount})"


# Commonly used action constructors for convenience
def fold() -> Action:
    """Create a fold action."""
    return Action(ActionType.FOLD, 0)


def check() -> Action:
    """Create a check action."""
    return Action(ActionType.CHECK, 0)


def call() -> Action:
    """Create a call action."""
    return Action(ActionType.CALL, 0)


def bet(amount: int) -> Action:
    """Create a bet action."""
    return Action(ActionType.BET, amount)


def raises(amount: int) -> Action:
    """Create a raise action."""
    return Action(ActionType.RAISE, amount)


def all_in(amount: int) -> Action:
    """Create an all-in action."""
    return Action(ActionType.ALL_IN, amount)
