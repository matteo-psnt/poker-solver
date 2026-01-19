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

from dataclasses import dataclass
from itertools import permutations

from treys import Card as TreysCard

from src.game.state import Card

# Suit constants
SUITS = ["s", "h", "d", "c"]  # spades, hearts, diamonds, clubs
# Treys uses power-of-2 encoding: s=1, h=2, d=4, c=8
TREYS_SUIT_INT_TO_CHAR = {1: "s", 2: "h", 4: "d", 8: "c"}
SUIT_TO_IDX = {"s": 0, "h": 1, "d": 2, "c": 3}
CANONICAL_SUITS = [0, 1, 2, 3]  # Canonical suit labels

# Rank ordering (A high)
RANKS = "AKQJT98765432"
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
# Treys rank encoding: 0=2, 1=3, ..., 12=A
TREYS_RANK_TO_OUR_IDX = {
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
    """
    Mapping from real suits to canonical suit labels.

    Attributes:
        mapping: Dict mapping suit characters to canonical labels (0-3)
        next_label: Next canonical label to assign for unseen suits
    """

    mapping: dict[str, int]
    next_label: int

    def __init__(self, mapping: dict[str, int] | None = None, next_label: int = 0):
        object.__setattr__(self, "mapping", dict(mapping) if mapping else {})
        object.__setattr__(self, "next_label", next_label)

    def get_or_assign(self, suit: str) -> tuple["SuitMapping", int]:
        """
        Get canonical label for a suit, assigning a new one if needed.

        Returns:
            Tuple of (new SuitMapping, canonical label)
        """
        if suit in self.mapping:
            return self, self.mapping[suit]

        new_mapping = dict(self.mapping)
        new_mapping[suit] = self.next_label
        return SuitMapping(new_mapping, self.next_label + 1), self.next_label

    def get(self, suit: str) -> int:
        """Get canonical label for a suit (must exist)."""
        return self.mapping[suit]

    def has(self, suit: str) -> bool:
        """Check if suit is in mapping."""
        return suit in self.mapping


@dataclass(frozen=True)
class CanonicalCard:
    """
    A card in canonical form.

    Attributes:
        rank_idx: Rank index (0=A, 1=K, ..., 12=2)
        suit_label: Canonical suit label (0-3)
    """

    rank_idx: int
    suit_label: int

    def __lt__(self, other: "CanonicalCard") -> bool:
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
    # Use treys' get_suit_int which returns power of 2 (1,2,4,8)
    suit_int = TreysCard.get_suit_int(card.card_int)
    return TREYS_SUIT_INT_TO_CHAR[suit_int]


def get_card_rank_idx(card: Card) -> int:
    """Extract rank index from a Card (0=A, 1=K, ..., 12=2)."""
    # Treys: get_rank_int returns 0=2, 1=3, ..., 12=A
    treys_rank = TreysCard.get_rank_int(card.card_int)
    return TREYS_RANK_TO_OUR_IDX[treys_rank]


def canonicalize_board(
    board: tuple[Card, ...], initial_mapping: SuitMapping | None = None
) -> tuple[tuple[CanonicalCard, ...], SuitMapping]:
    """
    Canonicalize a board under suit isomorphism.

    The canonical form is determined by:
    1. Sorting cards by rank (high to low)
    2. Trying all possible suit permutations
    3. Choosing the lexicographically smallest result

    This ensures that isomorphic boards (boards that differ only in which
    specific suits are used) map to the same canonical form.

    Args:
        board: Tuple of Card objects
        initial_mapping: Optional pre-existing suit mapping (typically None)

    Returns:
        Tuple of (canonical board, final suit mapping)

    Example:
        [T♠ 9♥ 8♠] and [T♥ 9♠ 8♥] both → ([T₀ 9₁ 8₀], ...)
    """
    if initial_mapping is not None and len(initial_mapping.mapping) > 0:
        # If we have an existing mapping, use the simple left-to-right approach
        return _canonicalize_board_with_mapping(board, initial_mapping)

    # Extract (rank_idx, suit_char) for each card
    cards_info = []
    for card in board:
        rank_idx = get_card_rank_idx(card)
        suit = get_card_suit(card)
        cards_info.append((rank_idx, suit))

    # Sort by rank (lower rank_idx = higher rank, e.g., 0=Ace)
    sorted_cards = sorted(cards_info, key=lambda x: x[0])

    # Find all suits present in the board
    present_suits = list(dict.fromkeys(suit for _, suit in sorted_cards))

    # Try all possible suit relabelings and find the lexicographically smallest
    best_canonical: tuple[CanonicalCard, ...] | None = None
    best_mapping: dict[str, int] | None = None

    # Only need to try permutations of length equal to number of distinct suits
    for perm in permutations(range(len(present_suits))):
        # Create mapping: present_suits[i] -> perm[i]
        suit_to_label = {present_suits[i]: perm[i] for i in range(len(present_suits))}

        # Apply mapping to get canonical cards
        canonical = tuple(
            sorted(
                (CanonicalCard(rank_idx, suit_to_label[suit]) for rank_idx, suit in sorted_cards)
            )
        )

        # Compare lexicographically
        if best_canonical is None or canonical < best_canonical:
            best_canonical = canonical
            best_mapping = suit_to_label

    # Build SuitMapping from best mapping (guaranteed to be set after loop)
    assert best_mapping is not None and best_canonical is not None
    final_mapping = SuitMapping(best_mapping, len(best_mapping))

    return best_canonical, final_mapping


def _canonicalize_board_with_mapping(
    board: tuple[Card, ...], mapping: SuitMapping
) -> tuple[tuple[CanonicalCard, ...], SuitMapping]:
    """
    Canonicalize board with a pre-existing suit mapping.

    Used when extending a board (e.g., turn card added to flop).
    Cards are processed in order, preserving the original board structure.
    """
    canonical_cards = []
    current_mapping = mapping

    for card in board:
        suit = get_card_suit(card)
        rank_idx = get_card_rank_idx(card)

        current_mapping, suit_label = current_mapping.get_or_assign(suit)
        canonical_cards.append(CanonicalCard(rank_idx, suit_label))

    return tuple(sorted(canonical_cards)), current_mapping


def canonicalize_hand(
    hole_cards: tuple[Card, Card], suit_mapping: SuitMapping
) -> tuple[CanonicalCard, CanonicalCard]:
    """
    Canonicalize a hand relative to an existing suit mapping.

    The suit mapping typically comes from canonicalizing the board first.
    New suits in the hand that aren't in the board mapping are assigned
    the next available canonical label.

    Args:
        hole_cards: Tuple of two Card objects
        suit_mapping: Suit mapping from board canonicalization

    Returns:
        Tuple of two CanonicalCard objects (ordered high to low)

    Example:
        Given mapping {♠:0, ♥:1} from board:
        [A♠ K♠] → (A₀, K₀)
        [A♥ K♥] → (A₁, K₁)
        [A♦ K♦] → (A₂, K₂)  # ♦ gets label 2
        [A♠ K♥] → (A₀, K₁)
    """
    mapping = suit_mapping
    canonical_cards = []

    for card in hole_cards:
        suit = get_card_suit(card)
        rank_idx = get_card_rank_idx(card)

        mapping, suit_label = mapping.get_or_assign(suit)
        canonical_cards.append(CanonicalCard(rank_idx, suit_label))

    # Order cards: higher rank first, then by suit label
    canonical_cards.sort()

    return (canonical_cards[0], canonical_cards[1])


def get_canonical_board_id(canonical_board: tuple[CanonicalCard, ...]) -> int:
    """
    Compute a unique integer ID for a canonical board.

    This provides a compact representation for hashing/lookup.

    Args:
        canonical_board: Tuple of CanonicalCard objects

    Returns:
        Integer ID
    """
    # Each card: rank (0-12) + suit (0-3) = 13*4 = 52 possible values
    # But canonical suits are assigned in order, so actual space is smaller
    # Use simple polynomial hash
    result = 0
    for card in canonical_board:
        result = result * 52 + (card.rank_idx * 4 + card.suit_label)
    return result


def get_canonical_hand_id(canonical_hand: tuple[CanonicalCard, CanonicalCard]) -> int:
    """
    Compute a unique integer ID for a canonical hand.

    Args:
        canonical_hand: Tuple of two CanonicalCard objects

    Returns:
        Integer ID (0 to ~2703 for 2-card hands with 4 canonical suits)
    """
    c1, c2 = canonical_hand
    # Each card: rank (0-12) * 4 + suit (0-3)
    idx1 = c1.rank_idx * 4 + c1.suit_label
    idx2 = c2.rank_idx * 4 + c2.suit_label

    # Combine (ordered pair within 52*52 space, but actually much smaller
    # since c1 <= c2 in canonical ordering)
    return idx1 * 52 + idx2


def board_to_canonical_tuple(board: tuple[Card, ...]) -> tuple[tuple[int, int], ...]:
    """
    Convert board to a hashable canonical tuple representation.

    Useful for dictionary keys and caching.
    """
    canonical_board, _ = canonicalize_board(board)
    return tuple(c.to_tuple() for c in canonical_board)


def hand_relative_to_board(
    hole_cards: tuple[Card, Card], board: tuple[Card, ...]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Get canonical hand representation relative to a board.

    This is the core function for postflop bucket lookup.

    Args:
        hole_cards: Player's hole cards
        board: Current board

    Returns:
        Tuple of (canonical_card_1, canonical_card_2) each as (rank_idx, suit_label)
    """
    _, suit_mapping = canonicalize_board(board)
    canonical_hand = canonicalize_hand(hole_cards, suit_mapping)
    return (canonical_hand[0].to_tuple(), canonical_hand[1].to_tuple())


def count_canonical_boards(street_cards: int) -> int:
    """
    Count the number of canonical boards for a given street.

    Under suit isomorphism, many raw boards are equivalent.

    Args:
        street_cards: Number of board cards (3=flop, 4=turn, 5=river)

    Returns:
        Number of canonical boards

    Note:
        Flop: ~1755 canonical boards (vs 22100 raw)
        Turn: ~16432 canonical boards (vs 270725 raw)
        River: ~134459 canonical boards (vs 2598960 raw)
    """
    # These are precomputed values for Hold'em
    if street_cards == 3:
        return 1755
    elif street_cards == 4:
        return 16432
    elif street_cards == 5:
        return 134459
    else:
        raise ValueError(f"Invalid street_cards: {street_cards}")


def count_canonical_hands_given_board(board_suit_count: int) -> int:
    """
    Count canonical hands given a board with a certain number of distinct suits.

    Args:
        board_suit_count: Number of distinct suits on the board (1-4)

    Returns:
        Number of canonical hand combos
    """
    # With n suits on board, hand can use those n suits + up to (4-n) new suits
    # But new suits are interchangeable among themselves
    # This is complex to compute exactly; for now return upper bound
    # Actual: ~300-500 canonical combos per board typically
    return 1326  # Conservative upper bound (can optimize later)
