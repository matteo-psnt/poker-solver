"""Columnar on-disk table of infoset keys and their legal-action signatures.

Replaces the two pickled dicts a checkpoint used to carry -- ``{InfoSetKey: id}``
and ``{id: [(action_type, amount)]}``. Pickle forces an all-or-nothing read, so
every worker had to materialize the *whole* checkpoint just to keep the ~1/N shard
it owns. At 18.9M keys that is 5.5 GB of keys plus 7.5 GB of signatures per worker,
and the load path read each of them twice: ~28 GB per worker, ~444 GB across 16.
That, not fork copy-on-write, is what made resuming a grown checkpoint OOM.

The fix is to make ownership decidable without reconstructing anything. Each row
carries the ``stable_hash`` its owner is derived from, so a worker memory-maps one
uint64 column, computes ``hash % num_workers == worker_id`` vectorised, and
materializes Python objects for its own rows only.

Layout (one directory, row ``i`` is infoset id ``i``, so no id column is needed):

    key_hash.npy        uint64   ownership; the whole point of the format
    player_position.npy int8
    street.npy          int8     Street value
    spr_bucket.npy      int16
    postflop_bucket.npy int32    -1 encodes None
    preflop_code.npy    int32    index into vocab["preflop_hands"], -1 encodes None
    seq_code.npy        int32    index into vocab["sequences"]
    sig_offsets.npy     int64    length n+1; row i's actions are [off[i], off[i+1])
    sig_type_code.npy   int8     index into vocab["action_types"]
    sig_amount.npy      int32
    vocab.json                   the three string vocabularies

Betting sequences and preflop hands are dictionary-encoded because they repeat
heavily across rows (many buckets share one sequence), which is what keeps the
string data small enough to be irrelevant.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.game.actions import Action, ActionType
from src.core.game.state import Street
from src.engine.solver.infoset import InfoSetKey

VOCAB_FILE = "vocab.json"

_COLUMNS: dict[str, np.dtype] = {
    "key_hash": np.dtype(np.uint64),
    "player_position": np.dtype(np.int8),
    "street": np.dtype(np.int8),
    "spr_bucket": np.dtype(np.int16),
    "postflop_bucket": np.dtype(np.int32),
    "preflop_code": np.dtype(np.int32),
    "seq_code": np.dtype(np.int32),
}
_SIG_COLUMNS: dict[str, np.dtype] = {
    "sig_offsets": np.dtype(np.int64),
    "sig_type_code": np.dtype(np.int8),
    "sig_amount": np.dtype(np.int32),
}

# Rows whose action list is absent (as opposed to empty) are written as a
# zero-length run; the loader raises on those, matching the old dict-lookup
# KeyError. An allocated infoset always has at least one legal action.
_MISSING_ACTIONS = 0


@dataclass(frozen=True)
class OwnedRows:
    """One worker's shard of a key table."""

    keys: list[InfoSetKey]
    row_ids: np.ndarray  # dense ids in the checkpoint, i.e. row indices
    action_lists: list[list[Action]]


def table_files(table_dir: Path) -> list[Path]:
    """Every file a complete table consists of (for existence checks)."""
    names = [*_COLUMNS, *_SIG_COLUMNS]
    return [table_dir / f"{name}.npy" for name in names] + [table_dir / VOCAB_FILE]


def write_key_table(
    table_dir: Path,
    keys: Sequence[InfoSetKey],
    key_hashes: Sequence[int],
    action_lists: Sequence[Sequence[Action] | None],
) -> None:
    """Write ``keys`` (in dense-id order) and their action lists as a columnar table.

    ``key_hashes`` must be ``stable_hash(key)`` for each key; it is passed in rather
    than recomputed so the writer cannot disagree with the ownership function.
    """
    if not (len(keys) == len(key_hashes) == len(action_lists)):
        raise ValueError(
            f"key table inputs disagree: {len(keys)} keys, {len(key_hashes)} hashes, "
            f"{len(action_lists)} action lists"
        )
    table_dir.mkdir(parents=True, exist_ok=True)
    n = len(keys)

    seq_vocab: dict[str, int] = {}
    hand_vocab: dict[str, int] = {}
    type_vocab: dict[str, int] = {}

    cols = {name: np.empty(n, dtype=dtype) for name, dtype in _COLUMNS.items()}
    offsets = np.zeros(n + 1, dtype=np.int64)
    type_codes: list[int] = []
    amounts: list[int] = []

    for i, key in enumerate(keys):
        cols["key_hash"][i] = np.uint64(key_hashes[i])
        cols["player_position"][i] = key.player_position
        cols["street"][i] = int(key.street.value)
        cols["spr_bucket"][i] = key.spr_bucket
        cols["postflop_bucket"][i] = -1 if key.postflop_bucket is None else int(key.postflop_bucket)
        if key.preflop_hand is None:
            cols["preflop_code"][i] = -1
        else:
            cols["preflop_code"][i] = hand_vocab.setdefault(key.preflop_hand, len(hand_vocab))
        cols["seq_code"][i] = seq_vocab.setdefault(key.betting_sequence, len(seq_vocab))

        actions = action_lists[i]
        if actions:
            for action in actions:
                type_codes.append(type_vocab.setdefault(action.type.name, len(type_vocab)))
                amounts.append(int(action.amount))
        offsets[i + 1] = len(type_codes)

    for name, array in cols.items():
        np.save(table_dir / f"{name}.npy", array)
    np.save(table_dir / "sig_offsets.npy", offsets)
    np.save(table_dir / "sig_type_code.npy", np.asarray(type_codes, dtype=np.int8))
    np.save(table_dir / "sig_amount.npy", np.asarray(amounts, dtype=np.int32))

    (table_dir / VOCAB_FILE).write_text(
        json.dumps(
            {
                "sequences": _vocab_list(seq_vocab),
                "preflop_hands": _vocab_list(hand_vocab),
                "action_types": _vocab_list(type_vocab),
                "num_rows": n,
            }
        )
    )


def _vocab_list(vocab: dict[str, int]) -> list[str]:
    out = [""] * len(vocab)
    for value, code in vocab.items():
        out[code] = value
    return out


def num_rows(table_dir: Path) -> int:
    """Row count, read from the vocab sidecar without touching the columns."""
    return int(json.loads((table_dir / VOCAB_FILE).read_text())["num_rows"])


def read_owned_rows(table_dir: Path, num_workers: int, worker_id: int) -> OwnedRows:
    """Materialize only the rows this worker owns.

    Ownership is decided on the memory-mapped hash column alone, so the Python
    objects built here are the shard's, never the whole table's.
    """
    hashes = np.load(table_dir / "key_hash.npy", mmap_mode="r")
    if num_workers == 1:
        rows = np.arange(len(hashes), dtype=np.int64)
    else:
        # Modulo on the mapped column; only this and the row index survive.
        rows = np.flatnonzero(np.mod(hashes, np.uint64(num_workers)) == np.uint64(worker_id))
        rows = rows.astype(np.int64, copy=False)
    return _materialize(table_dir, rows)


def read_all_rows(table_dir: Path) -> OwnedRows:
    """Materialize every row (single-process readers: evaluation, charts, migration)."""
    return _materialize(table_dir, np.arange(num_rows(table_dir), dtype=np.int64))


def _materialize(table_dir: Path, rows: np.ndarray) -> OwnedRows:
    vocab = json.loads((table_dir / VOCAB_FILE).read_text())
    sequences: list[str] = vocab["sequences"]
    hands: list[str] = vocab["preflop_hands"]
    action_types: list[str] = vocab["action_types"]

    def column(name: str) -> np.ndarray:
        # Fancy-indexing a memmap materializes only the selected rows.
        return np.load(table_dir / f"{name}.npy", mmap_mode="r")[rows]

    positions = column("player_position")
    streets = column("street")
    sprs = column("spr_bucket")
    buckets = column("postflop_bucket")
    hand_codes = column("preflop_code")
    seq_codes = column("seq_code")

    offsets = np.load(table_dir / "sig_offsets.npy", mmap_mode="r")
    type_code_col = np.load(table_dir / "sig_type_code.npy", mmap_mode="r")
    amount_col = np.load(table_dir / "sig_amount.npy", mmap_mode="r")

    keys: list[InfoSetKey] = []
    action_lists: list[list[Action]] = []
    for i, row in enumerate(rows.tolist()):
        bucket = int(buckets[i])
        hand_code = int(hand_codes[i])
        keys.append(
            InfoSetKey(
                player_position=int(positions[i]),
                street=Street(int(streets[i])),
                betting_sequence=sequences[int(seq_codes[i])],
                preflop_hand=None if hand_code < 0 else hands[hand_code],
                postflop_bucket=None if bucket < 0 else bucket,
                spr_bucket=int(sprs[i]),
            )
        )
        start, end = int(offsets[row]), int(offsets[row + 1])
        if end - start == _MISSING_ACTIONS:
            raise ValueError(f"key table {table_dir}: missing action signatures for row {row}")
        action_lists.append(
            [
                _action(action_types[int(type_code_col[j])], int(amount_col[j]))
                for j in range(start, end)
            ]
        )

    return OwnedRows(keys=keys, row_ids=rows, action_lists=action_lists)


def _action(type_name: str, amount: int) -> Action:
    try:
        action_type = ActionType[type_name]
    except KeyError as exc:
        raise ValueError(f"key table: unknown action type {type_name!r}") from exc
    return Action(action_type, amount)
