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

Enumeration results are cached on disk (``data/cache/canonical_boards/`` by
default) — the river's 2.6M raw boards take ~1 min to canonicalize but load
from cache in under a second.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from src.core.game.state import Card, Street
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    CanonicalCard,
    canonicalize_board,
    get_canonical_board_id,
)

DEFAULT_CACHE_DIR = Path("data/cache/canonical_boards")
_CACHE_FORMAT_VERSION = 1

# eval7 encodings used to serialize concrete cards as rank*4 + suit codes.
_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "cdhs"


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

    def __init__(self, street: Street, cache_dir: Path | None = DEFAULT_CACHE_DIR):
        """
        Initialize enumerator for a street.

        Args:
            street: Which street to enumerate (FLOP, TURN, or RIVER)
            cache_dir: Directory for the on-disk enumeration cache
                (None disables caching)
        """
        self.street = street
        self.num_cards = self._get_num_cards(street)
        self.cache_dir = cache_dir

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

        This is expensive but only needs to be done once per street: results
        are cached in memory and (if cache_dir is set) on disk.
        """
        if self._enumerated:
            return

        if self._load_from_disk():
            self._enumerated = True
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
        self._save_to_disk()

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return Path(self.cache_dir) / f"{self.street.name.lower()}_v{_CACHE_FORMAT_VERSION}.npz"

    def _save_to_disk(self) -> None:
        path = self._cache_path()
        if path is None:
            return

        infos = list(self._cache.values())
        n = len(infos)
        board_ids = np.array([info.board_id for info in infos], dtype=np.int64)
        raw_counts = np.array([info.raw_count for info in infos], dtype=np.int64)
        rep_codes = np.empty((n, self.num_cards), dtype=np.uint8)
        canon_codes = np.empty((n, self.num_cards), dtype=np.uint8)
        for i, info in enumerate(infos):
            for j, card in enumerate(info.representative):
                rep_codes[i, j] = card.rank_eval7() * 4 + card.suit_eval7()
            for j, canonical_card in enumerate(info.canonical_board):
                canon_codes[i, j] = canonical_card.rank_idx * 4 + canonical_card.suit_label

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            board_ids=board_ids,
            raw_counts=raw_counts,
            rep_codes=rep_codes,
            canon_codes=canon_codes,
        )

    def _load_from_disk(self) -> bool:
        path = self._cache_path()
        if path is None or not path.exists():
            return False

        data = np.load(path)
        board_ids = data["board_ids"]
        raw_counts = data["raw_counts"]
        rep_codes = data["rep_codes"]
        canon_codes = data["canon_codes"]

        expected = EXPECTED_CANONICAL_COUNTS.get(self.street)
        if expected is not None and board_ids.size != expected:
            return False  # stale/corrupt cache: fall back to full enumeration

        for i in range(board_ids.size):
            representative = tuple(
                Card.new(_RANK_CHARS[code // 4] + _SUIT_CHARS[code % 4]) for code in rep_codes[i]
            )
            canonical_board = tuple(
                CanonicalCard(rank_idx=int(code) // 4, suit_label=int(code) % 4)
                for code in canon_codes[i]
            )
            board_id = int(board_ids[i])
            self._cache[board_id] = CanonicalBoardInfo(
                canonical_board=canonical_board,
                board_id=board_id,
                raw_count=int(raw_counts[i]),
                representative=representative,
            )
            canonical_key = tuple(c.to_tuple() for c in canonical_board)
            self._canonical_to_id[canonical_key] = board_id

        return True

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
