"""Capacity and resize helpers for SharedArrayStorage."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from src.engine.solver.storage.shared_array.types import ExtraAllocation, ExtraRegion
from src.pipeline.abstraction.utils.infoset import InfoSetKey

if TYPE_CHECKING:
    from src.engine.solver.storage.shared_array.storage import SharedArrayStorage


def get_capacity_usage(storage: SharedArrayStorage) -> float:
    """Get fraction of this worker's ID range that is used."""
    base_size = storage.id_range_end - storage.id_range_start
    extra_size = sum(alloc.size for alloc in storage.state.extra_allocations)
    total_size = base_size + extra_size
    if total_size == 0:
        return 1.0

    base_used = max(
        0,
        min(storage.state.next_local_id, storage.id_range_end) - storage.id_range_start,
    )
    extra_used = sum(alloc.used for alloc in storage.state.extra_allocations)
    return (base_used + extra_used) / total_size


def needs_resize(storage: SharedArrayStorage) -> bool:
    """Check if storage usage crossed the threshold."""
    return get_capacity_usage(storage) >= storage.CAPACITY_THRESHOLD


def get_resize_stats(storage: SharedArrayStorage) -> dict[str, int | float]:
    """Get statistics used for resize decisions."""
    range_size = storage.id_range_end - storage.id_range_start
    extra_size = sum(alloc.size for alloc in storage.state.extra_allocations)
    used = max(
        0,
        min(storage.state.next_local_id, storage.id_range_end) - storage.id_range_start,
    )
    extra_used = sum(alloc.used for alloc in storage.state.extra_allocations)

    return {
        "worker_id": storage.worker_id,
        "id_range_start": storage.id_range_start,
        "id_range_end": storage.id_range_end,
        "next_local_id": storage.state.next_local_id,
        "range_size": range_size,
        "extra_size": extra_size,
        "used": used + extra_used,
        "capacity_usage": get_capacity_usage(storage),
        "initial_capacity": storage.capacity,
    }


def resize(storage: SharedArrayStorage, new_capacity: int) -> None:
    """Resize storage to new capacity (coordinator only)."""
    if not storage.is_coordinator:
        raise RuntimeError("Only coordinator can resize storage")

    if new_capacity <= storage.capacity:
        raise RuntimeError(
            f"New size {new_capacity} must be larger than current {storage.capacity}"
        )

    old_capacity = storage.capacity
    old_regrets = storage.shared_regrets
    old_strategy = storage.shared_strategy_sum
    old_action_counts = storage.shared_action_counts
    old_reach_counts = storage.shared_reach_counts
    old_cumulative_utility = storage.shared_cumulative_utility

    print(
        f"Resizing storage: {old_capacity:,} -> {new_capacity:,} infosets "
        f"(growth factor: {new_capacity / old_capacity:.1f}x)"
    )

    old_shm_regrets = storage.state.shm_regrets
    old_shm_strategy = storage.state.shm_strategy
    old_shm_actions = storage.state.shm_actions
    old_shm_reach = storage.state.shm_reach
    old_shm_utility = storage.state.shm_utility

    storage.capacity = new_capacity
    storage.session_id = uuid.uuid4().hex[:8]

    storage._create_shared_memory()

    storage.shared_regrets[:old_capacity, :] = old_regrets[:, :]
    storage.shared_strategy_sum[:old_capacity, :] = old_strategy[:, :]
    storage.shared_action_counts[:old_capacity] = old_action_counts[:]
    storage.shared_reach_counts[:old_capacity] = old_reach_counts[:]
    storage.shared_cumulative_utility[:old_capacity] = old_cumulative_utility[:]

    add_extra_region(storage, old_capacity, new_capacity)

    try:
        for shm in (
            old_shm_regrets,
            old_shm_strategy,
            old_shm_actions,
            old_shm_reach,
            old_shm_utility,
        ):
            if shm is None:
                continue
            shm.close()
            shm.unlink()
    except Exception as exc:
        print(f"Warning: Error cleaning up old shared memory: {exc}")

    print(f"Resize complete: new capacity {new_capacity:,}, new session_id={storage.session_id}")


def reattach_after_resize(
    storage: SharedArrayStorage,
    new_session_id: str,
    new_capacity: int,
    preserved_keys: dict[InfoSetKey, int],
    preserved_next_id: int,
) -> None:
    """Reattach worker to resized shared memory."""
    if storage.state.shm_regrets:
        storage.state.shm_regrets.close()
    if storage.state.shm_strategy:
        storage.state.shm_strategy.close()
    if storage.state.shm_actions:
        storage.state.shm_actions.close()
    if storage.state.shm_reach:
        storage.state.shm_reach.close()
    if storage.state.shm_utility:
        storage.state.shm_utility.close()

    old_capacity = storage.capacity

    storage.session_id = new_session_id
    storage.capacity = new_capacity
    storage.state.next_local_id = preserved_next_id
    storage.state.owned_keys = preserved_keys

    storage._attach_shared_memory()
    add_extra_region(storage, old_capacity, new_capacity)

    print(
        f"Worker {storage.worker_id} reattached after resize: "
        f"session={new_session_id}, initial_capacity={new_capacity:,}, "
        f"id_range=[{storage.id_range_start}, {storage.id_range_end}), "
        f"next_id={storage.state.next_local_id}"
    )


def add_extra_region(storage: SharedArrayStorage, extra_start: int, extra_end: int) -> None:
    """Register a new resize region and allocate this worker's slice."""
    total = extra_end - extra_start
    if total <= 0:
        return

    base = total // storage.num_workers
    remainder = total % storage.num_workers
    storage.state.extra_regions.append(
        ExtraRegion(
            start=extra_start,
            total=total,
            base=base,
            remainder=remainder,
        )
    )

    if base == 0 and storage.worker_id >= remainder:
        return

    start = extra_start + storage.worker_id * base + min(storage.worker_id, remainder)
    end = start + base + (1 if storage.worker_id < remainder else 0)
    if start >= end:
        return

    storage.state.extra_allocations.append(
        ExtraAllocation(
            start=start,
            end=end,
            next=start,
        )
    )
