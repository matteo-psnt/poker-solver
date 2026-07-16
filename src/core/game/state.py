"""
Game state representation for Heads-Up No-Limit Hold'em.

This module defines the core game state, including cards, streets, and the
complete game state dataclass that tracks all information needed for a poker hand.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import TYPE_CHECKING

import eval7

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.rules import GameRules

from src.core.game.actions import Action, ActionType


class Street(Enum):
    """Betting rounds in Texas Hold'em."""

    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def is_preflop(self) -> bool:
        return self == Street.PREFLOP

    def is_postflop(self) -> bool:
        return self != Street.PREFLOP

    def next_street(self) -> Street | None:
        """Get the next street, or None if this is the river."""
        match self:
            case Street.PREFLOP:
                return Street.FLOP
            case Street.FLOP:
                return Street.TURN
            case Street.TURN:
                return Street.RIVER
            case _:
                return None


class Card:
    """
    Card representation using the eval7 library.

    This wraps eval7.Card to provide a cleaner interface while maintaining
    compatibility with the fast hand evaluation.

    Note: Card objects are cached for performance. Creating the same card
    multiple times returns the same object.
    """

    def __init__(self, eval7_card: eval7.Card):
        """
        Initialize from eval7.Card object.

        Args:
            eval7_card: eval7.Card instance (use Card.new() to create)
        """
        self._card = eval7_card
        self._hash: int | None = None  # Cache hash value for performance

    @property
    def mask(self) -> int:
        """Get the unique integer identifier for this card (eval7's mask)."""
        return self._card.mask

    def to_eval7(self) -> eval7.Card:
        """Return the underlying eval7 card object."""
        return self._card

    def rank_eval7(self) -> int:
        """Return eval7 rank encoding (0=2, ..., 12=A)."""
        return self._card.rank

    def suit_eval7(self) -> int:
        """Return eval7 suit encoding (0=c, 1=d, 2=h, 3=s)."""
        return self._card.suit

    @classmethod
    def new(cls, card_str: str) -> Card:
        """
        Create a card from string representation (e.g., 'As', 'Kh', '2d').

        Cards are cached - calling this multiple times with the same string
        returns the same Card object for performance.

        Args:
            card_str: Two-character string (rank + suit)
                     Ranks: '2'-'9', 'T', 'J', 'Q', 'K', 'A'
                     Suits: 's', 'h', 'd', 'c'

        Returns:
            Card instance (cached)
        """
        return cls._new_cached(card_str)

    @classmethod
    @lru_cache(maxsize=52)
    def _new_cached(cls, card_str: str) -> Card:
        return cls(eval7.Card(card_str))

    @classmethod
    def get_full_deck(cls) -> list[Card]:
        """
        Get a full 52-card deck.

        The deck is cached for performance. Returns a copy to prevent
        accidental mutation of the cached deck.

        Returns:
            List of all 52 cards
        """
        return list(cls._full_deck_cached())

    @classmethod
    @lru_cache(maxsize=1)
    def _full_deck_cached(cls) -> tuple[Card, ...]:
        deck = eval7.Deck()
        return tuple(cls(card) for card in deck.cards)

    def __str__(self) -> str:
        """String representation (e.g., '[ A ♠ ]')."""
        # Create pretty format manually since eval7 doesn't have int_to_pretty_str
        rank_char = str(self._card)[0]
        suit_char = str(self._card)[1]
        suit_symbols = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
        return f"[ {rank_char} {suit_symbols.get(suit_char, suit_char)} ]"

    def __repr__(self) -> str:
        """Compact representation (e.g., 'As')."""
        return str(self._card)

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison based on card mask.

        Optimized with fast type() check for common case.
        """
        # Fast path: direct type check (most common case)
        if type(other) is Card:
            return self._card == other._card

        # Slow path: handle subclasses
        if not isinstance(other, Card):
            return False
        return self._card == other._card

    def __hash__(self) -> int:
        """
        Hash based on card mask for use in sets/dicts.

        Hash is cached for performance (called millions of times).
        """
        if not hasattr(self, "_hash") or self._hash is None:
            self._hash = hash(self._card)
        return self._hash

    def __lt__(self, other: Card) -> bool:
        """Compare cards for sorting (by rank)."""
        return self._card < other._card

    def __reduce__(self):
        """Support pickling by storing the string representation."""
        # Return a tuple (callable, args) to reconstruct the object
        # Use the string representation to recreate the card
        return (self.__class__.new, (str(self._card),))


@dataclass(frozen=True)
class GameState:
    """
    Immutable game state for Heads-Up No-Limit Hold'em.

    This represents a complete snapshot of the game at any point,
    including all public and private information.

    Attributes:
        street: Current betting round
        pot: Total chips in the pot
        stacks: Remaining chips for each player (player0, player1)
        board: Community cards (empty on preflop)
        hole_cards: Private cards for each player
        betting_history: Chronological sequence of actions this hand
        button_position: Which player is on the button (0 or 1)
        current_player: Which player acts next (0 or 1)
        is_terminal: Whether this state is terminal (hand over)
        to_call: Amount current player needs to call (0 if can check)
        last_aggressor: Player who last bet/raised (None if no betting yet)
        _skip_validation: Skip validation (for CFR internal use)
    """

    street: Street
    pot: int
    stacks: tuple[int, int]
    board: tuple[Card, ...]
    hole_cards: tuple[tuple[Card, Card], tuple[Card, Card]]
    betting_history: tuple[Action, ...]
    button_position: int
    current_player: int
    is_terminal: bool
    to_call: int = 0
    last_aggressor: int | None = None
    # Preflop blind gap (big_blind - small_blind); constant for the hand.
    # Seeds pot reconstruction in _normalize_betting_sequence.
    blind_to_call: int = 0
    _skip_validation: bool = False
    _cached_betting_sequence: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        """Validate game state consistency."""
        # Skip validation if requested (for CFR internal states)
        if self._skip_validation:
            return

        # Validate player indices
        if self.button_position not in (0, 1):
            raise ValueError(f"Invalid button_position: {self.button_position}")
        if self.current_player not in (0, 1):
            raise ValueError(f"Invalid current_player: {self.current_player}")
        if self.last_aggressor is not None and self.last_aggressor not in (0, 1):
            raise ValueError(f"Invalid last_aggressor: {self.last_aggressor}")

        # Validate stacks
        if len(self.stacks) != 2:
            raise ValueError(f"Must have exactly 2 stacks, got {len(self.stacks)}")
        if any(stack < 0 for stack in self.stacks):
            raise ValueError(f"Negative stack: {self.stacks}")

        # Validate pot
        if self.pot < 0:
            raise ValueError(f"Negative pot: {self.pot}")

        # Validate to_call
        if self.to_call < 0:
            raise ValueError(f"Negative to_call: {self.to_call}")
        if self.to_call > self.stacks[self.current_player]:
            raise ValueError(
                f"to_call ({self.to_call}) exceeds stack ({self.stacks[self.current_player]})"
            )

        # Validate board cards
        expected_board_size = {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
        }[self.street]
        if len(self.board) != expected_board_size:
            raise ValueError(
                f"Board should have {expected_board_size} cards on {self.street}, "
                f"got {len(self.board)}"
            )

        # Validate hole cards
        if len(self.hole_cards) != 2:
            raise ValueError(f"Must have exactly 2 hands, got {len(self.hole_cards)}")
        for i, hand in enumerate(self.hole_cards):
            if len(hand) != 2:
                raise ValueError(f"Player {i} must have 2 cards, got {len(hand)}")

    def opponent(self, player: int) -> int:
        """Get opponent of given player."""
        return 1 - player

    def is_all_in(self) -> bool:
        """Check if any player is all-in."""
        return any(stack == 0 for stack in self.stacks)

    def effective_stack(self) -> int:
        """Get the smaller of the two stacks (effective stack depth)."""
        return min(self.stacks)

    def can_check(self) -> bool:
        """Check if current player can check (no money to call)."""
        return self.to_call == 0

    def can_bet(self) -> bool:
        """Check if current player can bet (no prior betting this street)."""
        # Can bet if no betting has occurred on this street
        return self.to_call == 0 and self.stacks[self.current_player] > 0

    def can_raise(self) -> bool:
        """Check if current player can raise (facing a bet with chips left)."""
        return self.to_call > 0 and self.stacks[self.current_player] > self.to_call

    def legal_actions(
        self, action_model: ActionModel | None = None, rules: GameRules | None = None
    ) -> list[Action]:
        """Get legal actions for current player using the provided rules engine."""
        if rules is None:
            raise ValueError("rules is required for GameState.legal_actions()")
        return rules.get_legal_actions(self, action_model)

    def apply_action(self, action: Action, rules: GameRules | None = None) -> GameState:
        """Apply an action using the provided rules engine and return the next state."""
        if rules is None:
            raise ValueError("rules is required for GameState.apply_action()")
        return rules.apply_action(self, action)

    def get_payoff(self, player: int, rules: GameRules | None = None) -> float:
        """Compute payoff for a player using the provided rules engine."""
        if rules is None:
            raise ValueError("rules is required for GameState.get_payoff()")
        return rules.get_payoff(self, player)

    def replace(self, *, validate: bool = True, **changes) -> GameState:
        """Copy of this state with ``changes`` applied (the one sanctioned way to
        derive a modified state — never enumerate all fields by hand).

        ``validate=False`` is the explicit escape hatch for mid-transition states
        that legitimately violate invariants (e.g. a street advanced before its
        board is dealt fails the board-size check). The flag is symmetric: the
        derived state's ``_skip_validation`` is set from ``validate``, never
        inherited from the source state.
        """
        changes.setdefault("_skip_validation", not validate)
        return dataclasses.replace(self, **changes)

    @property
    def ended_by_fold(self) -> bool:
        """True when the hand ended on a fold (the alternative terminal is showdown)."""
        return bool(self.betting_history) and self.betting_history[-1].type == ActionType.FOLD

    def normalized_betting_sequence(self) -> str:
        """Return canonical betting-sequence encoding used in infoset keys."""
        return self._normalize_betting_sequence()

    def _normalize_betting_sequence(self) -> str:
        """
        Normalize betting history to a string for the infoset key.

        Each bet/raise is normalized against the pot at the time it was made
        (fold/check/call/all-in are pot-independent), so a sequence normalizes
        identically regardless of which street we observe it from. ``to_call``
        is seeded with ``blind_to_call`` (blind posts are absent from
        ``betting_history``) and the pot level is anchored on the authoritative
        current ``pot``, which every constructor preserves.
        """
        if self._cached_betting_sequence is not None:
            return self._cached_betting_sequence

        if not self.betting_history:
            object.__setattr__(self, "_cached_betting_sequence", "")
            return ""

        # Pass 1: each action's contribution relative to the (as-yet-unknown)
        # starting pot, plus the running pot before each action.
        contributed_before: list[int] = []
        contributed = 0
        to_call = self.blind_to_call
        for action in self.betting_history:
            contributed_before.append(contributed)

            if action.type == ActionType.BET:
                contributed += action.amount
                to_call = action.amount
            elif action.type == ActionType.RAISE:
                # RAISE commits the call plus the raise-over amount.
                contributed += to_call + action.amount
                to_call = action.amount
            elif action.type == ActionType.CALL:
                contributed += to_call
                to_call = 0
            elif action.type == ActionType.ALL_IN:
                # amount is the TOTAL stack committed. Excess over the call
                # becomes the new to_call; an all-in call (possibly for less)
                # closes the action.
                contributed += action.amount
                to_call = action.amount - to_call if action.amount > to_call else 0
            # FOLD and CHECK contribute nothing and leave to_call unchanged.

        # Anchor on the authoritative current pot. initial_pot is the post-blind
        # starting pot (small_blind + big_blind); a negative value can only come
        # from broken contribution accounting.
        initial_pot = self.pot - contributed
        if initial_pot < 0:
            raise ValueError(
                f"Betting-history reconstruction overshoots pot={self.pot} "
                f"(contributions={contributed}); accounting is inconsistent."
            )

        normalized = [
            action.normalize(initial_pot + pot_before)
            for action, pot_before in zip(self.betting_history, contributed_before)
        ]

        sequence = "-".join(normalized)
        object.__setattr__(self, "_cached_betting_sequence", sequence)
        return sequence

    def __str__(self) -> str:
        """Human-readable string representation."""
        player_str = ["P0", "P1"][self.current_player]
        street_str = str(self.street)
        pot_str = f"Pot: {self.pot}"
        stacks_str = f"Stacks: P0={self.stacks[0]}, P1={self.stacks[1]}"
        board_str = f"Board: {[repr(c) for c in self.board]}" if self.board else "Board: []"
        to_call_str = f"ToCall: {self.to_call}" if self.to_call > 0 else ""

        return (
            f"GameState({player_str} to act | {street_str} | {pot_str} | "
            f"{stacks_str} | {board_str} | {to_call_str})"
        )
