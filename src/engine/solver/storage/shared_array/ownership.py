"""Ownership and stable-hashing helpers for shared-array storage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import xxhash

from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage.shared_array.types import AllocationLike, RegionLike


def _unpack_region(region: tuple[int, int, int, int] | RegionLike) -> tuple[int, int, int, int]:
    """Extract typed region fields from tuple/dataclass representations."""
    if isinstance(region, tuple):
        return cast(tuple[int, int, int, int], region)

    region_like = cast(RegionLike, region)
    return (
        region_like.start,
        region_like.total,
        region_like.base,
        region_like.remainder,
    )


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


def is_owned_by_id(
    infoset_id: int,
    unknown_id: int,
    id_range_start: int,
    id_range_end: int,
    extra_allocations: Sequence[AllocationLike],
) -> bool:
    """Determine if a worker owns a specific infoset id."""
    if infoset_id == unknown_id:
        return False
    if id_range_start <= infoset_id < id_range_end:
        return True
    return any(alloc.start <= infoset_id < alloc.end for alloc in extra_allocations)


def owner_for_id(
    infoset_id: int,
    *,
    unknown_id: int,
    base_slots_per_worker: int,
    num_workers: int,
    extra_regions: Sequence[tuple[int, int, int, int] | RegionLike],
) -> int | None:
    """
    Determine owner worker for a given infoset id.

    Uses initial base ranges plus any appended resize regions.
    """
    if infoset_id == unknown_id:
        return None

    base_end = 1 + base_slots_per_worker * num_workers
    if 1 <= infoset_id < base_end:
        return (infoset_id - 1) // base_slots_per_worker

    for region in extra_regions:
        extra_start, extra_total, base, remainder = _unpack_region(region)
        extra_end = extra_start + extra_total
        if extra_start <= infoset_id < extra_end:
            offset = infoset_id - extra_start
            if base == 0:
                return offset if offset < remainder else None
            threshold = (base + 1) * remainder
            if offset < threshold:
                return offset // (base + 1)
            return remainder + (offset - threshold) // base

    return None
