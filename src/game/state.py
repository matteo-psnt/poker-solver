"""
Game state representation for Heads-Up No-Limit Hold'em.

This module defines the core game state, including cards, streets, and the
complete game state dataclass that tracks all information needed for a poker hand.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from treys import Card as TreysCard

from src.game.actions import Action

# Import SPR constants only when needed to avoid circular import at module level
# They are imported inside methods that use them


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

    def next_street(self) -> Optional["Street"]:
        """Get the next street, or None if this is the river."""
        if self == Street.PREFLOP:
            return Street.FLOP
        elif self == Street.FLOP:
            return Street.TURN
        elif self == Street.TURN:
            return Street.RIVER
        else:
            return None


# Module-level caches for performance
_CARD_CACHE = {}  # Cache for Card.new()
_FULL_DECK_CACHE = None  # Cache for full deck


class Card:
    """
    Card representation using the treys library.

    This wraps treys.Card to provide a cleaner interface while maintaining
    compatibility with the fast hand evaluation.

    Note: Card objects are cached for performance. Creating the same card
    multiple times returns the same object.
    """

    def __init__(self, card_int: int):
        """
        Initialize from treys card integer.

        Args:
            card_int: Integer representation from treys (use Card.new() to create)
        """
        self.card_int = card_int
        self._hash = None  # Cache hash value for performance

    @classmethod
    def new(cls, card_str: str) -> "Card":
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
        if card_str not in _CARD_CACHE:
            _CARD_CACHE[card_str] = cls(TreysCard.new(card_str))
        return _CARD_CACHE[card_str]

    @classmethod
    def get_full_deck(cls) -> List["Card"]:
        """
        Get a full 52-card deck.

        The deck is cached for performance. Returns a copy to prevent
        accidental mutation of the cached deck.

        Returns:
            List of all 52 cards
        """
        global _FULL_DECK_CACHE

        if _FULL_DECK_CACHE is None:
            ranks = "23456789TJQKA"
            suits = "shdc"
            _FULL_DECK_CACHE = [cls.new(f"{rank}{suit}") for rank in ranks for suit in suits]

        # Return a copy to prevent mutation
        return _FULL_DECK_CACHE.copy()

    def __str__(self) -> str:
        """String representation (e.g., '[ A ♠ ]')."""
        return TreysCard.int_to_pretty_str(self.card_int)

    def __repr__(self) -> str:
        """Compact representation (e.g., 'As')."""
        return TreysCard.int_to_str(self.card_int)

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison based on card integer.

        Optimized with fast type() check for common case.
        """
        # Fast path: direct type check (most common case)
        if type(other) is Card:
            return self.card_int == other.card_int

        # Slow path: handle subclasses
        if not isinstance(other, Card):
            return False
        return self.card_int == other.card_int

    def __hash__(self) -> int:
        """
        Hash based on card integer for use in sets/dicts.

        Hash is cached for performance (called millions of times).
        """
        if not hasattr(self, "_hash") or self._hash is None:
            self._hash = hash(self.card_int)
        return self._hash

    def __lt__(self, other: "Card") -> bool:
        """Compare cards for sorting (by rank)."""
        # In treys, lower values = better ranks
        return self.card_int < other.card_int


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
    stacks: Tuple[int, int]
    board: Tuple[Card, ...]
    hole_cards: Tuple[Tuple[Card, Card], Tuple[Card, Card]]
    betting_history: Tuple[Action, ...]
    button_position: int
    current_player: int
    is_terminal: bool
    to_call: int = 0
    last_aggressor: Optional[int] = None
    _skip_validation: bool = False

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

    def legal_actions(self, action_abstraction=None) -> List[Action]:
        """
        Get legal actions for current player.

        Args:
            action_abstraction: Optional action abstraction

        Returns:
            List of legal actions (delegates to GameRules)
        """
        # Import here to avoid circular dependency
        from src.game.rules import GameRules

        rules = GameRules()
        return rules.get_legal_actions(self, action_abstraction)

    def apply_action(self, action: Action, rules=None) -> "GameState":
        """
        Apply action to get next state.

        Args:
            action: Action to apply
            rules: Optional GameRules instance (creates one if not provided)

        Returns:
            Next game state (delegates to GameRules)
        """
        # Import here to avoid circular dependency
        from src.game.rules import GameRules

        if rules is None:
            rules = GameRules()
        return rules.apply_action(self, action)

    def get_payoff(self, player: int, rules=None) -> int:
        """
        Get payoff for a player (only valid in terminal states).

        Args:
            player: Player index (0 or 1)
            rules: Optional GameRules instance

        Returns:
            Chips won/lost (delegates to GameRules)
        """
        # Import here to avoid circular dependency
        from src.game.rules import GameRules

        if rules is None:
            rules = GameRules()
        return rules.get_payoff(self, player)

    def get_infoset_key(self, player: int, card_abstraction):
        """
        Get information set key for a player.

        Args:
            player: Player index (0 or 1)
            card_abstraction: CardAbstraction instance for bucketing

        Returns:
            InfoSetKey (will be implemented in abstraction module)
        """
        # Imports here to avoid circular dependencies
        from src.abstraction.infoset import InfoSetKey
        from src.abstraction.preflop_hands import PreflopHandMapper

        # Compute SPR bucket
        effective_stack = min(self.stacks)
        spr = effective_stack / self.pot if self.pot > 0 else 0
        spr_bucket = self._get_spr_bucket(spr)

        # Normalize betting history
        betting_sequence = self._normalize_betting_sequence()

        # Hybrid representation: preflop uses hand strings, postflop uses buckets
        if self.street == Street.PREFLOP:
            # Get canonical hand string (e.g., "AKs", "72o", "TT")
            hand_mapper = PreflopHandMapper()
            preflop_hand = hand_mapper.get_hand_string(self.hole_cards[player])

            return InfoSetKey(
                player_position=player,
                street=self.street,
                betting_sequence=betting_sequence,
                preflop_hand=preflop_hand,
                postflop_bucket=None,
                spr_bucket=spr_bucket,
            )
        else:
            # Get bucket for postflop streets
            postflop_bucket = card_abstraction.get_bucket(
                self.hole_cards[player], self.board, self.street
            )

            return InfoSetKey(
                player_position=player,
                street=self.street,
                betting_sequence=betting_sequence,
                preflop_hand=None,
                postflop_bucket=postflop_bucket,
                spr_bucket=spr_bucket,
            )

    def _get_spr_bucket(self, spr: float) -> int:
        """
        Get SPR bucket (0=shallow, 1=medium, 2=deep).

        SPR (Stack-to-Pot Ratio) thresholds:
        - Shallow: SPR < 4 (push/fold decisions)
        - Medium: 4 <= SPR < 13 (standard play)
        - Deep: SPR >= 13 (complex postflop play)

        Args:
            spr: Stack-to-pot ratio

        Returns:
            Bucket index (0, 1, or 2)
        """
        from src.abstraction.constants import SPR_DEEP_THRESHOLD, SPR_SHALLOW_THRESHOLD

        if spr < SPR_SHALLOW_THRESHOLD:
            return 0  # Shallow
        elif spr < SPR_DEEP_THRESHOLD:
            return 1  # Medium
        else:
            return 2  # Deep

    def _normalize_betting_sequence(self) -> str:
        """Normalize betting history to string for infoset key."""
        if not self.betting_history:
            return ""

        # Convert actions to normalized strings
        normalized = []
        for action in self.betting_history:
            normalized.append(action.normalize(self.pot))

        return "-".join(normalized)

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
