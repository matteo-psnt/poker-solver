"""
Canonical board enumeration under suit isomorphism.

Generates all unique canonical board representations for each street.
This is the foundation for the precomputation pipeline.

Key insight: Under suit isomorphism, many raw boards are equivalent.
For example, [A♠ K♠ Q♠] is equivalent to [A♥ K♥ Q♥] - both are
monotone AKQ boards.

This module provides:
1. Efficient enumeration of canonical boards (not all 22100 flops, just ~1755)
2. Mapping from raw boards to their canonical representatives
3. Storage-efficient board IDs for lookup tables
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

from src.bucketing.postflop.suit_isomorphism import (
    RANKS,
    CanonicalCard,
    canonicalize_board,
    get_canonical_board_id,
)
from src.game.state import Card, Street


@dataclass
class CanonicalBoardInfo:
    """
    Information about a canonical board.

    Attributes:
        canonical_board: The canonical form (tuple of CanonicalCard)
        board_id: Unique integer ID
        raw_count: Number of raw boards that map to this canonical form
        representative: A concrete example board (using real suits)
    """

    canonical_board: tuple[CanonicalCard, ...]
    board_id: int
    raw_count: int
    representative: tuple[Card, ...]


class CanonicalBoardEnumerator:
    """
    Enumerates and caches all canonical boards for a street.

    Usage:
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        for board_info in enumerator.iterate():
            process(board_info)
    """

    def __init__(self, street: Street):
        """
        Initialize enumerator for a street.

        Args:
            street: Which street to enumerate (FLOP, TURN, or RIVER)
        """
        self.street = street
        self.num_cards = self._get_num_cards(street)

        # Cache: board_id -> CanonicalBoardInfo
        self._cache: dict[int, CanonicalBoardInfo] = {}

        # Reverse lookup: canonical tuple -> board_id
        self._canonical_to_id: dict[tuple[tuple[int, int], ...], int] = {}

        self._enumerated = False

    @staticmethod
    def _get_num_cards(street: Street) -> int:
        """Get number of board cards for a street."""
        if street == Street.FLOP:
            return 3
        elif street == Street.TURN:
            return 4
        elif street == Street.RIVER:
            return 5
        else:
            raise ValueError(f"Invalid street: {street}")

    def enumerate(self) -> None:
        """
        Enumerate all canonical boards for this street.

        This is expensive but only needs to be done once.
        Results are cached internally.
        """
        if self._enumerated:
            return

        all_cards = Card.get_full_deck()
        seen_canonical: set[tuple[tuple[int, int], ...]] = set()

        for card_combo in combinations(all_cards, self.num_cards):
            board = tuple(card_combo)

            # Canonicalize
            canonical_board, _ = canonicalize_board(board)
            canonical_key = tuple(c.to_tuple() for c in canonical_board)

            if canonical_key in seen_canonical:
                # Already have this canonical form, increment count
                board_id = self._canonical_to_id[canonical_key]
                self._cache[board_id] = CanonicalBoardInfo(
                    canonical_board=self._cache[board_id].canonical_board,
                    board_id=board_id,
                    raw_count=self._cache[board_id].raw_count + 1,
                    representative=self._cache[board_id].representative,
                )
            else:
                # New canonical form
                board_id = get_canonical_board_id(canonical_board)
                seen_canonical.add(canonical_key)
                self._canonical_to_id[canonical_key] = board_id
                self._cache[board_id] = CanonicalBoardInfo(
                    canonical_board=canonical_board,
                    board_id=board_id,
                    raw_count=1,
                    representative=board,
                )

        self._enumerated = True

    def iterate(self) -> Iterator[CanonicalBoardInfo]:
        """
        Iterate over all canonical boards.

        Yields:
            CanonicalBoardInfo for each unique canonical board
        """
        if not self._enumerated:
            self.enumerate()

        yield from self._cache.values()

    def get_by_id(self, board_id: int) -> CanonicalBoardInfo | None:
        """Get canonical board info by ID."""
        if not self._enumerated:
            self.enumerate()
        return self._cache.get(board_id)

    def get_canonical_id(self, board: tuple[Card, ...]) -> int:
        """Get canonical board ID for a raw board."""
        canonical_board, _ = canonicalize_board(board)
        return get_canonical_board_id(canonical_board)

    def count(self) -> int:
        """Get total number of canonical boards."""
        if not self._enumerated:
            self.enumerate()
        return len(self._cache)

    def __len__(self) -> int:
        return self.count()


# Precomputed counts for validation
EXPECTED_CANONICAL_COUNTS = {
    Street.FLOP: 1755,
    Street.TURN: 16432,
    Street.RIVER: 134459,
}


def validate_canonical_count(street: Street) -> bool:
    """
    Validate that enumeration produces expected number of canonical boards.

    Returns:
        True if count matches expected
    """
    enumerator = CanonicalBoardEnumerator(street)
    enumerator.enumerate()
    actual = enumerator.count()
    expected = EXPECTED_CANONICAL_COUNTS.get(street, -1)

    if actual != expected:
        print(f"Warning: {street.name} has {actual} canonical boards, expected {expected}")
        return False
    return True


def get_canonical_board_representative(
    canonical_board: tuple[CanonicalCard, ...],
) -> tuple[Card, ...]:
    """
    Convert a canonical board back to a concrete board.

    Uses spades first, then hearts, diamonds, clubs for canonical suit labels.

    Args:
        canonical_board: Tuple of CanonicalCard objects

    Returns:
        Tuple of Card objects
    """
    suit_map = ["s", "h", "d", "c"]  # canonical 0,1,2,3 -> real suits

    cards = []
    for cc in canonical_board:
        rank_char = RANKS[cc.rank_idx]
        suit_char = suit_map[cc.suit_label]
        cards.append(Card.new(f"{rank_char}{suit_char}"))

    return tuple(cards)


def count_hands_without_conflicts(board: tuple[Card, ...]) -> int:
    """
    Count how many 2-card hands don't share cards with the board.

    For any board: 50 choose 2 = 1225 valid hands
    """
    total_cards = 52
    board_cards = len(board)
    remaining = total_cards - board_cards
    return remaining * (remaining - 1) // 2
