"""Ownership and stable-hashing helpers for shared-array storage."""

from __future__ import annotations

import xxhash

from src.engine.solver.infoset import InfoSetKey


def stable_hash(key: InfoSetKey) -> int:
    """
    Compute stable hash of InfoSetKey using xxhash.

    Python's built-in hash() is randomized per process, which breaks
    ownership consistency across workers.
    """
    parts = [
        str(key.player_position).encode(),
        key.street.name.encode(),
        key.betting_sequence.encode(),
        (key.preflop_hand or "").encode(),
        str(key.postflop_bucket if key.postflop_bucket is not None else -1).encode(),
        str(key.spr_bucket).encode(),
    ]
    key_bytes = b"|".join(parts)
    return xxhash.xxh64(key_bytes).intdigest()


def owner_for_key(key: InfoSetKey, num_workers: int) -> int:
    """Determine owner worker for an infoset key."""
    return stable_hash(key) % num_workers
