"""
Suit isomorphism canonicalization.

Provides functions to canonicalize boards and hands under suit isomorphism.
This is the foundation for combo-level abstraction in postflop poker.

Canonical Form:
- Suits are mapped to labels 0,1,2,3 in order of first appearance
- Board cards are processed left-to-right to establish the mapping
- Hand cards extend the mapping with any new suits

Example:
    Board [T♠ 9♥ 8♠] → mapping {♠:0, ♥:1}
    Canonical board: [T₀ 9₁ 8₀]

    Hand [A♠ K♠] → [A₀ K₀] (same flush potential as board)
    Hand [A♥ K♥] → [A₁ K₁] (matches board suit 1)
    Hand [A♦ K♦] → [A₂ K₂] (new suit, assigned 2)
    Hand [A♠ K♥] → [A₀ K₁] (mixed suits)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.game.state import Card

# Suit constants
SUITS = ["s", "h", "d", "c"]  # spades, hearts, diamonds, clubs
# eval7 uses sequential encoding: c=0, d=1, h=2, s=3
EVAL7_SUIT_TO_CHAR = {0: "c", 1: "d", 2: "h", 3: "s"}

# Rank ordering (A high)
RANKS = "AKQJT98765432"
# eval7 rank encoding: 0=2, 1=3, ..., 12=A
EVAL7_RANK_TO_OUR_IDX = {
    0: 12,
    1: 11,
    2: 10,
    3: 9,
    4: 8,
    5: 7,
    6: 6,
    7: 5,
    8: 4,
    9: 3,
    10: 2,
    11: 1,
    12: 0,
}


@dataclass(frozen=True)
class SuitMapping:
    """Mapping from real suit characters to canonical labels 0-3."""

    mapping: dict[str, int]
    next_label: int

    def __init__(self, mapping: dict[str, int] | None = None, next_label: int = 0):
        object.__setattr__(self, "mapping", dict(mapping) if mapping else {})
        object.__setattr__(self, "next_label", next_label)

    def get_or_assign(self, suit: str) -> tuple[SuitMapping, int]:
        """The label for a suit, assigning a new one if unseen, with the mapping that results."""
        if suit in self.mapping:
            return self, self.mapping[suit]

        new_mapping = dict(self.mapping)
        new_mapping[suit] = self.next_label
        return SuitMapping(new_mapping, self.next_label + 1), self.next_label

    def get(self, suit: str) -> int:
        """Get canonical label for a suit (must exist)."""
        return self.mapping[suit]


@dataclass(frozen=True)
class CanonicalCard:
    """A card in canonical form: rank index (0=A ... 12=2) and suit label 0-3."""

    rank_idx: int
    suit_label: int

    def __lt__(self, other: CanonicalCard) -> bool:
        """Ordering: by rank first, then suit."""
        if self.rank_idx != other.rank_idx:
            return self.rank_idx < other.rank_idx
        return self.suit_label < other.suit_label

    def to_tuple(self) -> tuple[int, int]:
        return (self.rank_idx, self.suit_label)

    def __repr__(self) -> str:
        rank_char = RANKS[self.rank_idx]
        return f"{rank_char}_{self.suit_label}"


def get_card_suit(card: Card) -> str:
    """Extract suit character from a Card."""
    # eval7 suit attribute returns 0-3 (c=0, d=1, h=2, s=3)
    return EVAL7_SUIT_TO_CHAR[card.suit_eval7()]


def get_card_rank_idx(card: Card) -> int:
    """Extract rank index from a Card (0=A, 1=K, ..., 12=2)."""
    # eval7 rank encoding: 0=2, 1=3, ..., 12=A
    return EVAL7_RANK_TO_OUR_IDX[card.rank_eval7()]


_SENTINEL = 99  # above every rank_idx, so a suit that stops appearing sorts last


def _suit_labels(cards_info: list[tuple[int, str]]) -> dict[str, int]:
    """The canonical suit labelling, derived rather than searched for.

    The canonical board is the lexicographically smallest relabelling, and this
    used to be found by trying all 4! of them — 24 sorts and 24 tuples per
    board, which measured 10.5us and was 68% of a river bucket lookup.

    It can be read off instead. Cards compare as ``rank_idx * 4 + label``, so
    rank dominates and the label only breaks ties *within* a rank. Minimising
    the sequence therefore means: the suit appearing at the highest card takes
    label 0, and where two suits first appear at the same rank, the one that
    keeps appearing at higher cards wins. That is a lexicographic order on each
    suit's ascending rank list — with SHORTER-IS-GREATER, because a suit that
    runs out of cards must lose to one that has more and plain tuple comparison
    gets that backwards (``[0] < [0, 3]``). Padding to the board width with a
    sentinel above every rank fixes it.

    The final tie-break is the suit CHARACTER, which is load-bearing rather
    than cosmetic: two suits with identical rank lists are interchangeable on
    the board but NOT for the hand read against the resulting mapping, so
    without a deterministic tie-break a hand's bucket would depend on the order
    the board tuple happened to be dealt in. Verified against the exhaustive
    relabelling search over every flop and turn and 200k random rivers, boards
    shuffled — ``test_suit_isomorphism`` — which is also how the tie-break was
    caught: enumerating boards in deck order never exercises it.
    """
    ranks_by_suit: dict[str, list[int]] = {}
    for rank_idx, suit in cards_info:
        ranks_by_suit.setdefault(suit, []).append(rank_idx)

    width = len(cards_info)
    order = sorted(
        ranks_by_suit,
        key=lambda suit: (
            (
                *sorted(ranks_by_suit[suit]),
                *((_SENTINEL,) * (width - len(ranks_by_suit[suit]))),
            ),
            suit,
        ),
    )
    return {suit: label for label, suit in enumerate(order)}


def canonicalize_board(
    board: tuple[Card, ...],
) -> tuple[tuple[CanonicalCard, ...], SuitMapping]:
    """Canonicalize a board under suit isomorphism, with the mapping it used.

    The canonical form is the lexicographically smallest suit relabelling, so boards
    differing only in which suits are used share one form -- [T♠ 9♥ 8♠] and
    [T♥ 9♠ 8♥] both give [T₀ 9₁ 8₀]. :func:`_suit_labels` derives that relabelling
    directly; the cards are then emitted in ``(rank, label)`` order.

    Callers that only want the id -- the runtime bucket lookup -- should use
    :func:`canonical_board_id`, which skips the ``CanonicalCard`` objects.
    """
    # Extract (rank_idx, suit_char) for each card
    cards_info = []
    for card in board:
        rank_idx = get_card_rank_idx(card)
        suit = get_card_suit(card)
        cards_info.append((rank_idx, suit))

    labels = _suit_labels(cards_info)
    codes = sorted(rank_idx * 4 + labels[suit] for rank_idx, suit in cards_info)
    canonical = tuple(CanonicalCard(code >> 2, code & 3) for code in codes)
    return canonical, SuitMapping(labels, len(labels))


def canonicalize_hand(
    hole_cards: tuple[Card, Card], suit_mapping: SuitMapping
) -> tuple[CanonicalCard, CanonicalCard]:
    """Canonicalize a hand against a mapping, ordered high to low.

    The mapping typically comes from canonicalizing the board first. A suit not in
    it is assigned the next available label, so given {♠:0, ♥:1} a [A♦ K♦] becomes
    (A₂, K₂).
    """
    mapping = suit_mapping
    canonical_cards = []

    # Assign new labels in rank order (high card first) so the canonical form
    # is independent of the input order of the hole cards. Without this,
    # (A♦, K♥) and (K♥, A♦) would canonicalize differently whenever both
    # suits are new to the mapping (A₂K₃ vs A₃K₂).
    ordered_cards = sorted(hole_cards, key=get_card_rank_idx)

    for card in ordered_cards:
        suit = get_card_suit(card)
        rank_idx = get_card_rank_idx(card)

        mapping, suit_label = mapping.get_or_assign(suit)
        canonical_cards.append(CanonicalCard(rank_idx, suit_label))

    # Order cards: higher rank first, then by suit label
    canonical_cards.sort()

    return (canonical_cards[0], canonical_cards[1])


def get_canonical_board_id(canonical_board: tuple[CanonicalCard, ...]) -> int:
    """A unique integer id for a canonical board, for hashing and lookup."""
    # Each card: rank (0-12) + suit (0-3) = 13*4 = 52 possible values
    # But canonical suits are assigned in order, so actual space is smaller
    # Use simple polynomial hash
    result = 0
    for card in canonical_board:
        result = result * 52 + (card.rank_idx * 4 + card.suit_label)
    return result


def get_canonical_hand_id(canonical_hand: tuple[CanonicalCard, CanonicalCard]) -> int:
    """A unique integer id for a canonical hand: 0 to ~2703 for two cards."""
    c1, c2 = canonical_hand
    # Each card: rank (0-12) * 4 + suit (0-3)
    idx1 = c1.rank_idx * 4 + c1.suit_label
    idx2 = c2.rank_idx * 4 + c2.suit_label

    # Combine (ordered pair within 52*52 space, but actually much smaller
    # since c1 <= c2 in canonical ordering)
    return idx1 * 52 + idx2


def canonical_board_id(board: tuple[Card, ...]) -> tuple[int, dict[str, int]]:
    """``(board id, suit labels)`` without building the cards in between.

    What :meth:`DenseBucketer._board_row` wants is an integer to binary-search
    and a mapping to read the hand against. Going through
    ``canonicalize_board`` allocates a ``CanonicalCard`` per board card and a
    ``SuitMapping``, then ``get_canonical_board_id`` walks the tuple once and
    throws it away. On the river that is five dataclasses per lookup, on a path
    where a fresh runout misses both LRUs every visit.

    Same labelling as :func:`canonicalize_board` — both call
    :func:`_suit_labels` — and the id is the same polynomial
    :func:`get_canonical_board_id` computes, so the two agree by construction
    and are pinned to agree by test.
    """
    cards_info = [(get_card_rank_idx(card), get_card_suit(card)) for card in board]
    labels = _suit_labels(cards_info)

    board_id = 0
    for code in sorted(rank_idx * 4 + labels[suit] for rank_idx, suit in cards_info):
        board_id = board_id * 52 + code
    return board_id, labels


def canonical_hand_id(hole_cards: tuple[Card, Card], labels: dict[str, int]) -> int:
    """The hand's canonical id against a board's suit labels, in one pass.

    Mirrors ``get_canonical_hand_id(canonicalize_hand(...))``, including the
    part that is easy to miss: a hole-card suit absent from the board takes the
    next free label, and those are handed out in RANK order (high card first)
    so that ``(Ad, Kh)`` and ``(Kh, Ad)`` cannot canonicalise differently.

    ``labels`` is not mutated — a new suit's label is local to this hand.
    """
    (rank_a, suit_a), (rank_b, suit_b) = sorted(
        ((get_card_rank_idx(card), get_card_suit(card)) for card in hole_cards),
        key=lambda pair: pair[0],
    )

    next_label = len(labels)
    label_a = labels.get(suit_a)
    if label_a is None:
        label_a = next_label
        next_label += 1
    if suit_b == suit_a:
        label_b = label_a
    else:
        label_b = labels.get(suit_b)
        if label_b is None:
            label_b = next_label

    first, second = sorted((rank_a * 4 + label_a, rank_b * 4 + label_b))
    return first * 52 + second


def hand_relative_to_board(
    hole_cards: tuple[Card, Card], board: tuple[Card, ...]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The canonical ``(rank_idx, suit_label)`` pair for a hand on a board.

    The core of postflop bucket lookup.
    """
    _, suit_mapping = canonicalize_board(board)
    canonical_hand = canonicalize_hand(hole_cards, suit_mapping)
    return (canonical_hand[0].to_tuple(), canonical_hand[1].to_tuple())
