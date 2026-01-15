"""
Canonical (hand, board) representations for combo-level abstraction.

Uses suit isomorphism to collapse strategically identical situations:
- 169 classes treat AKs as a single entity, ignoring which suits
- Combo-level tracks actual suits relative to the board
- A♠K♠ on T♠9♠8♣ is different from A♥K♥ on T♠9♠8♣ (flush vs no flush)
- But A♠K♠ on T♠9♠8♣ is equivalent to A♥K♥ on T♥9♥8♣ (suit isomorphism)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.core.game.state import Card
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    RANKS,
    SUITS,
    CanonicalCard,
    SuitMapping,
    canonicalize_board,
    canonicalize_hand,
    get_canonical_board_id,
    get_canonical_hand_id,
)


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

    def to_key(self) -> tuple[int, int]:
        """Get (board_id, hand_id) tuple for dictionary keys."""
        return (self.board_id, self.hand_id)

    def __repr__(self) -> str:
        hand_str = f"({self.hand[0]}, {self.hand[1]})"
        board_str = " ".join(str(c) for c in self.board)
        return f"CanonicalHand(hand={hand_str}, board=[{board_str}])"


@dataclass(frozen=True)
class HandClass:
    """
    One canonical hand class on a specific board.

    Attributes:
        canonical: The canonical (hand, board) pair
        representative: A concrete hole-card pair in this class
        multiplicity: Number of concrete combos that map to this class
    """

    canonical: CanonicalHand
    representative: tuple[Card, Card]
    multiplicity: int


def canonicalize_combo(hole_cards: tuple[Card, Card], board: tuple[Card, ...]) -> CanonicalHand:
    """Canonicalize a (hand, board) pair."""
    canonical_board, suit_mapping = canonicalize_board(board)
    canonical_hand = canonicalize_hand(hole_cards, suit_mapping)
    return CanonicalHand(hand=canonical_hand, board=canonical_board)


def generate_all_cards() -> list[Card]:
    """Generate all 52 cards."""
    cards = []
    for rank in RANKS:
        for suit in SUITS:
            cards.append(Card.new(f"{rank}{suit}"))
    return cards


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
    """
    Generate all canonical combos for a given board.

    Args:
        board: The board cards
        exclude_board_cards: If True, exclude hands that share cards with board

    Yields:
        CanonicalHand objects
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


def get_representative_hand(
    canonical_hand: tuple[CanonicalCard, CanonicalCard], suit_mapping: SuitMapping
) -> tuple[Card, Card]:
    """
    Convert a canonical hand back to a concrete hand.

    Uses the inverse of the suit mapping to get real suits.

    Args:
        canonical_hand: Canonical hand representation
        suit_mapping: Mapping used to canonicalize

    Returns:
        Tuple of two Card objects
    """
    # Invert the mapping
    inv_mapping = {v: k for k, v in suit_mapping.mapping.items()}

    # Assign remaining suits for any canonical labels not in mapping
    available_suits = [s for s in SUITS if s not in suit_mapping.mapping]
    for label in range(4):
        if label not in inv_mapping:
            if available_suits:
                inv_mapping[label] = available_suits.pop(0)
            else:
                inv_mapping[label] = "s"  # Fallback

    cards = []
    for cc in canonical_hand:
        rank_char = RANKS[cc.rank_idx]
        suit_char = inv_mapping[cc.suit_label]
        cards.append(Card.new(f"{rank_char}{suit_char}"))

    return (cards[0], cards[1])
