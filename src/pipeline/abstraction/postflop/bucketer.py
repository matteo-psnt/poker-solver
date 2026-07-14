"""
Dense-array combo abstraction with full per-board coverage.

The artifact maps every canonical (board, hand) pair to a bucket via three
mmap-friendly numpy arrays per street:

- ``{street}_board_ids.npy``: sorted canonical board IDs (int64), one row per
  canonical board — all of them, so every legal postflop state resolves.
- ``{street}_buckets.npy``: bucket matrix ``[n_boards, N_HAND_COLUMNS]``
  (uint8/uint16). Cells for hand classes that cannot exist on a board hold the
  dtype's max value as a sentinel.
- ``hand_id_to_col.npy``: static mapping from canonical hand ID (0..2703) to
  matrix column (int32, -1 for impossible IDs).

Lookup is canonicalize → binary-search board row → index column → read cell.
There is no fallback path: a sentinel hit means the queried state is not a
legal (hand, board) combination.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.core.game.state import Card, Street
from src.pipeline.abstraction.base import BucketingStrategy
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    canonicalize_board,
    canonicalize_hand,
    get_canonical_board_id,
    get_canonical_hand_id,
)
from src.pipeline.abstraction.preflop.hand_classes import PreflopHandClasses

STORAGE_VERSION = 2
METADATA_FILENAME = "metadata.json"
HAND_COLUMN_FILENAME = "hand_id_to_col.npy"

POSTFLOP_STREETS = (Street.FLOP, Street.TURN, Street.RIVER)

# Canonical hand IDs live in [0, 52*52); exactly the C(52,2) ordered pairs of
# distinct canonical card codes are realizable.
N_HAND_IDS = 52 * 52
N_HAND_COLUMNS = 52 * 51 // 2

_PREFLOP_CLASSES = PreflopHandClasses()

_LOOKUP_CACHE_SIZE = 1 << 18


def build_hand_column_index() -> np.ndarray:
    """Static canonical-hand-ID → column mapping (-1 for impossible IDs)."""
    index = np.full(N_HAND_IDS, -1, dtype=np.int32)
    col = 0
    for idx1 in range(52):
        for idx2 in range(idx1 + 1, 52):
            index[idx1 * 52 + idx2] = col
            col += 1
    assert col == N_HAND_COLUMNS
    return index


def bucket_dtype(num_buckets: int) -> np.dtype:
    """Smallest uint dtype that fits ``num_buckets`` plus the invalid sentinel."""
    if num_buckets < np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if num_buckets < np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    raise ValueError(f"num_buckets={num_buckets} exceeds uint16 storage")


def _street_filenames(street: Street) -> tuple[str, str]:
    name = street.name.lower()
    return f"{name}_board_ids.npy", f"{name}_buckets.npy"


class DenseBucketer(BucketingStrategy):
    """
    Runtime bucket lookup over full-coverage dense bucket matrices.

    Preflop uses the 169 lossless hand classes; postflop streets read the
    precomputed matrices. Lookups are memoized (CFR revisits states heavily).
    """

    def __init__(
        self,
        num_buckets_by_street: dict[Street, int],
        board_ids_by_street: dict[Street, np.ndarray],
        buckets_by_street: dict[Street, np.ndarray],
        hand_id_to_col: np.ndarray,
        source_path: Path | None = None,
    ):
        self._num_buckets = dict(num_buckets_by_street)
        self._board_ids = dict(board_ids_by_street)
        self._buckets = dict(buckets_by_street)
        self._hand_id_to_col = hand_id_to_col
        self._source_path = source_path
        self._sentinels = {
            street: np.iinfo(matrix.dtype).max for street, matrix in self._buckets.items()
        }
        self._cached_postflop_bucket = lru_cache(maxsize=_LOOKUP_CACHE_SIZE)(self._postflop_bucket)

    def __getstate__(self) -> dict:
        """
        Pickle support (training workers receive the abstraction by pickle).

        A bucketer loaded from disk serializes as just its path — each worker
        re-mmaps the same artifact files instead of receiving a copy of the
        matrices. In-memory bucketers (precompute results, tests) serialize
        their arrays directly.
        """
        if self._source_path is not None:
            return {"_source_path": self._source_path}
        state = dict(self.__dict__)
        state.pop("_cached_postflop_bucket")  # lru_cache wrapper is not picklable
        state["_buckets"] = {street: np.asarray(matrix) for street, matrix in self._buckets.items()}
        return state

    def __setstate__(self, state: dict) -> None:
        if set(state) == {"_source_path"}:
            loaded = DenseBucketer.load(state["_source_path"])
            self.__dict__.update(loaded.__dict__)
            return
        self.__dict__.update(state)
        self._cached_postflop_bucket = lru_cache(maxsize=_LOOKUP_CACHE_SIZE)(self._postflop_bucket)

    @classmethod
    def load(cls, path: Path) -> DenseBucketer:
        """Load a precomputed abstraction directory (bucket matrices are mmapped)."""
        path = Path(path)
        metadata_path = path / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No abstraction metadata at {metadata_path}. "
                "Run 'Precompute Combo Abstraction' from the CLI to generate it."
            )
        with open(metadata_path) as f:
            metadata = json.load(f)

        version = metadata.get("storage_version")
        if version != STORAGE_VERSION:
            raise RuntimeError(
                f"Abstraction at {path} has storage_version={version}, expected "
                f"{STORAGE_VERSION}. Regenerate it via 'Precompute Combo Abstraction'."
            )

        street_stats = metadata.get("streets", {})
        num_buckets: dict[Street, int] = {}
        board_ids: dict[Street, np.ndarray] = {}
        buckets: dict[Street, np.ndarray] = {}
        for street in POSTFLOP_STREETS:
            stats = street_stats.get(street.name)
            if stats is None:
                continue
            ids_file, buckets_file = _street_filenames(street)
            num_buckets[street] = int(stats["num_buckets"])
            board_ids[street] = np.load(path / ids_file)
            buckets[street] = np.load(path / buckets_file, mmap_mode="r")

        hand_id_to_col = np.load(path / HAND_COLUMN_FILENAME)
        return cls(num_buckets, board_ids, buckets, hand_id_to_col, source_path=path)

    def save_arrays(self, path: Path) -> None:
        """Write the per-street arrays (metadata is written by the precomputer)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / HAND_COLUMN_FILENAME, self._hand_id_to_col)
        for street in self._buckets:
            ids_file, buckets_file = _street_filenames(street)
            np.save(path / ids_file, self._board_ids[street])
            np.save(path / buckets_file, self._buckets[street])

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        """
        Get bucket ID for a (hand, board) pair.

        Raises:
            KeyError: If the (hand, board) pair is not a legal combination.
        """
        if street == Street.PREFLOP:
            return _PREFLOP_CLASSES.get_hand_index(hole_cards)
        return self._cached_postflop_bucket(hole_cards, tuple(board), street)

    def _postflop_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        matrix = self._buckets.get(street)
        if matrix is None:
            raise KeyError(f"No buckets loaded for street {street.name}")

        canonical_board, suit_mapping = canonicalize_board(board)
        board_id = get_canonical_board_id(canonical_board)

        board_ids = self._board_ids[street]
        row = int(np.searchsorted(board_ids, board_id))
        if row >= board_ids.size or board_ids[row] != board_id:
            raise KeyError(
                f"Board {board} (canonical id {board_id}) not found for {street.name}; "
                "the abstraction is incomplete or the board is malformed."
            )

        canonical_hand = canonicalize_hand(hole_cards, suit_mapping)
        col = int(self._hand_id_to_col[get_canonical_hand_id(canonical_hand)])
        bucket = int(matrix[row, col]) if col >= 0 else self._sentinels[street]
        if bucket == self._sentinels[street]:
            raise KeyError(f"Hand {hole_cards} is not a legal combo on board {board}")
        return bucket

    def num_buckets(self, street: Street) -> int:
        """Get number of buckets for a street."""
        if street == Street.PREFLOP:
            return 169
        return self._num_buckets.get(street, 0)

    def get_bucket_distribution(self, street: Street) -> dict[int, int]:
        """Bucket → canonical-class count over all boards for a street."""
        matrix = self._buckets.get(street)
        if matrix is None:
            return {}
        counts = np.bincount(np.asarray(matrix).ravel(), minlength=self._sentinels[street] + 1)
        return {
            bucket: int(count)
            for bucket, count in enumerate(counts[: self._sentinels[street]])
            if count > 0
        }

    def __str__(self) -> str:
        parts = [f"PREFLOP={self.num_buckets(Street.PREFLOP)}"]
        parts.extend(
            f"{street.name}={self._num_buckets[street]}"
            for street in POSTFLOP_STREETS
            if street in self._num_buckets
        )
        return f"DenseBucketer({', '.join(parts)})"
