"""
Canonical (hand, board) representations for combo-level abstraction.

Uses suit isomorphism to collapse strategically identical situations:
- 169 classes treat AKs as a single entity, ignoring which suits
- Combo-level tracks actual suits relative to the board
- A♠K♠ on T♠9♠8♣ is different from A♥K♥ on T♠9♠8♣ (flush vs no flush)
- But A♠K♠ on T♠9♠8♣ is equivalent to A♥K♥ on T♥9♥8♣ (suit isomorphism)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.game.state import Card
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    RANKS,
    SUITS,
    CanonicalCard,
    canonicalize_board,
    canonicalize_hand,
    get_canonical_board_id,
    get_canonical_hand_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class CanonicalHand:
    """
    A canonical (hand, board) pair.

    Represents a postflop situation in canonical form, where:
    - Board suits are assigned labels 0,1,2,3 in order of appearance
    - Hand suits use the board's mapping, extending for new suits

    This is the fundamental unit for combo-level bucketing.
    """

    hand: tuple[CanonicalCard, CanonicalCard]
    board: tuple[CanonicalCard, ...]

    @property
    def hand_id(self) -> int:
        """Unique ID for the canonical hand."""
        return get_canonical_hand_id(self.hand)

    @property
    def board_id(self) -> int:
        """Unique ID for the canonical board."""
        return get_canonical_board_id(self.board)

    def __repr__(self) -> str:
        hand_str = f"({self.hand[0]}, {self.hand[1]})"
        board_str = " ".join(str(c) for c in self.board)
        return f"CanonicalHand(hand={hand_str}, board=[{board_str}])"


@dataclass(frozen=True)
class HandClass:
    """One canonical hand class on a specific board, with the concrete pair that
    represents it and how many combos map to it.
    """

    canonical: CanonicalHand
    representative: tuple[Card, Card]
    multiplicity: int


def generate_all_cards() -> list[Card]:
    """Generate all 52 cards."""
    return [Card.new(f"{rank}{suit}") for rank in RANKS for suit in SUITS]


def enumerate_hand_classes(board: tuple[Card, ...]) -> list[HandClass]:
    """
    Enumerate all canonical hand classes on a board, with multiplicities.

    Every concrete hole-card pair that doesn't collide with the board maps to
    exactly one class; class members are strategically identical (equal equity)
    by suit symmetry, so one representative suffices for equity computation
    while ``multiplicity`` preserves the class weight for bucketing.
    """
    canonical_board, suit_mapping = canonicalize_board(board)
    board_card_set = set(board)
    cards = [c for c in generate_all_cards() if c not in board_card_set]

    classes: dict[tuple[tuple[int, int], tuple[int, int]], HandClass] = {}

    for i, c1 in enumerate(cards):
        for c2 in cards[i + 1 :]:
            canonical_hand = canonicalize_hand((c1, c2), suit_mapping)
            key = (canonical_hand[0].to_tuple(), canonical_hand[1].to_tuple())

            existing = classes.get(key)
            if existing is None:
                classes[key] = HandClass(
                    canonical=CanonicalHand(hand=canonical_hand, board=canonical_board),
                    representative=(c1, c2),
                    multiplicity=1,
                )
            else:
                classes[key] = HandClass(
                    canonical=existing.canonical,
                    representative=existing.representative,
                    multiplicity=existing.multiplicity + 1,
                )

    return list(classes.values())


def get_all_canonical_hands(
    board: tuple[Card, ...], exclude_board_cards: bool = True
) -> Iterator[CanonicalHand]:
    """Every canonical combo on a board.

    ``exclude_board_cards`` drops hands sharing a card with the board.
    """
    canonical_board, suit_mapping = canonicalize_board(board)
    board_card_set = set(board)

    seen_canonical: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    cards = generate_all_cards()

    for i, c1 in enumerate(cards):
        if exclude_board_cards and c1 in board_card_set:
            continue

        for c2 in cards[i + 1 :]:
            if exclude_board_cards and c2 in board_card_set:
                continue

            canonical_hand = canonicalize_hand((c1, c2), suit_mapping)
            canonical_key = (canonical_hand[0].to_tuple(), canonical_hand[1].to_tuple())

            if canonical_key in seen_canonical:
                continue

            seen_canonical.add(canonical_key)

            yield CanonicalHand(hand=canonical_hand, board=canonical_board)
