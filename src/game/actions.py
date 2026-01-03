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
    """
    Immutable representation of a poker action.

    Attributes:
        type: The type of action (fold, check, call, bet, raise, all-in)
        amount: Chips involved in the action. Semantics vary by action type:

            - FOLD: Must be 0 (no chips involved)
            - CHECK: Must be 0 (no chips involved)
            - CALL: Must be 0 (actual call amount determined by state.to_call)
            - BET: Total chips being bet (added to pot)
            - RAISE: Additional chips above the call amount (NOT total bet size)
                     Total chips added = state.to_call + amount
            - ALL_IN: Player's remaining stack size (total chips going into pot)
                     This is the ACTUAL stack amount, not relative to any bet.
                     When processing ALL_IN:
                     - If amount >= to_call: treated as call + possible raise
                     - If amount < to_call: treated as all-in call for less (uncalled portion returned)

    ALL_IN Semantics (Important for InfoSet key normalization):
        The ALL_IN action's amount field represents the player's TOTAL remaining stack
        that is being committed. This differs from BET/RAISE where amount is the bet size.

        Example scenarios:
        1. Player has 100 chips, faces 50 to call, goes all-in:
           - Action: ALL_IN(100)
           - Effect: Calls 50, raises 50 more
           - Opponent faces 50 to call

        2. Player has 30 chips, faces 50 to call, goes all-in:
           - Action: ALL_IN(30)
           - Effect: All-in call for less than full amount
           - Goes to showdown, uncalled 20 returned to opponent

        3. Player opens all-in for 100 (no bet facing):
           - Action: ALL_IN(100)
           - Effect: All-in bet of 100
           - Opponent faces 100 to call
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
        # Fast path for common non-amount actions (no computation needed)
        if self.type == ActionType.FOLD:
            return "f"
        elif self.type == ActionType.CHECK:
            return "x"
        elif self.type == ActionType.CALL:
            return "c"
        elif self.type == ActionType.ALL_IN:
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
