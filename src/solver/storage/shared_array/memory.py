"""Shared-memory lifecycle helpers for SharedArrayStorage."""

from __future__ import annotations

import time
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np

from src.solver.storage.shared_array_layout import get_shm_name

if TYPE_CHECKING:
    from src.solver.storage.shared_array.storage import SharedArrayStorage


def get_shm_name_for_storage(storage: SharedArrayStorage, base: str) -> str:
    """Get session-namespaced shared memory name."""
    return get_shm_name(base, storage.session_id)


def create_shared_memory(storage: SharedArrayStorage) -> None:
    """Create all shared memory segments (coordinator only)."""
    cleanup_stale_shm(storage)

    regrets_size = storage.capacity * storage.max_actions * np.dtype(np.float64).itemsize
    strategy_size = storage.capacity * storage.max_actions * np.dtype(np.float64).itemsize
    actions_size = storage.capacity * np.dtype(np.int32).itemsize
    reach_size = storage.capacity * np.dtype(np.int64).itemsize
    utility_size = storage.capacity * np.dtype(np.float64).itemsize

    storage.state.shm_regrets = shared_memory.SharedMemory(
        create=True,
        size=regrets_size,
        name=get_shm_name_for_storage(storage, storage.SHM_REGRETS),
    )
    storage.state.shm_strategy = shared_memory.SharedMemory(
        create=True,
        size=strategy_size,
        name=get_shm_name_for_storage(storage, storage.SHM_STRATEGY),
    )
    storage.state.shm_actions = shared_memory.SharedMemory(
        create=True,
        size=actions_size,
        name=get_shm_name_for_storage(storage, storage.SHM_ACTIONS),
    )
    storage.state.shm_reach = shared_memory.SharedMemory(
        create=True,
        size=reach_size,
        name=get_shm_name_for_storage(storage, storage.SHM_REACH),
    )
    storage.state.shm_utility = shared_memory.SharedMemory(
        create=True,
        size=utility_size,
        name=get_shm_name_for_storage(storage, storage.SHM_UTILITY),
    )

    create_numpy_views(storage)

    storage.shared_regrets.fill(0)
    storage.shared_strategy_sum.fill(0)
    storage.shared_action_counts.fill(0)
    storage.shared_reach_counts.fill(0)
    storage.shared_cumulative_utility.fill(0)

    print(
        "Master created shared memory: "
        f"regrets={regrets_size // 1024 // 1024}MB, "
        f"strategy={strategy_size // 1024 // 1024}MB"
    )

    if storage.ready_event is not None:
        storage.ready_event.set()


def attach_shared_memory(storage: SharedArrayStorage) -> None:
    """Attach to existing shared memory segments (worker)."""
    if storage.ready_event is not None:
        wait_timeout = 30.0
        if not storage.ready_event.wait(timeout=wait_timeout):
            raise RuntimeError(
                f"Worker {storage.worker_id} timed out waiting for coordinator "
                f"to create shared memory (waited {wait_timeout}s)"
            )

    max_retries = 5
    retry_delay = 0.1

    for attempt in range(max_retries):
        try:
            storage.state.shm_regrets = shared_memory.SharedMemory(
                name=get_shm_name_for_storage(storage, storage.SHM_REGRETS)
            )
            storage.state.shm_strategy = shared_memory.SharedMemory(
                name=get_shm_name_for_storage(storage, storage.SHM_STRATEGY)
            )
            storage.state.shm_actions = shared_memory.SharedMemory(
                name=get_shm_name_for_storage(storage, storage.SHM_ACTIONS)
            )
            storage.state.shm_reach = shared_memory.SharedMemory(
                name=get_shm_name_for_storage(storage, storage.SHM_REACH)
            )
            storage.state.shm_utility = shared_memory.SharedMemory(
                name=get_shm_name_for_storage(storage, storage.SHM_UTILITY)
            )

            create_numpy_views(storage)
            return

        except FileNotFoundError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Worker {storage.worker_id} failed to attach to shared memory "
                    f"after {max_retries} attempts. "
                    "Ensure coordinator creates memory before workers start."
                )


def create_numpy_views(storage: SharedArrayStorage) -> None:
    """Create NumPy array views into shared memory."""
    if (
        storage.state.shm_regrets is None
        or storage.state.shm_strategy is None
        or storage.state.shm_actions is None
        or storage.state.shm_reach is None
        or storage.state.shm_utility is None
    ):
        raise RuntimeError("Shared memory buffers are not initialized")

    storage.shared_regrets = np.ndarray(
        (storage.capacity, storage.max_actions),
        dtype=np.float64,
        buffer=storage.state.shm_regrets.buf,
    )
    storage.shared_strategy_sum = np.ndarray(
        (storage.capacity, storage.max_actions),
        dtype=np.float64,
        buffer=storage.state.shm_strategy.buf,
    )
    storage.shared_action_counts = np.ndarray(
        (storage.capacity,),
        dtype=np.int32,
        buffer=storage.state.shm_actions.buf,
    )
    storage.shared_reach_counts = np.ndarray(
        (storage.capacity,),
        dtype=np.int64,
        buffer=storage.state.shm_reach.buf,
    )
    storage.shared_cumulative_utility = np.ndarray(
        (storage.capacity,),
        dtype=np.float64,
        buffer=storage.state.shm_utility.buf,
    )


def cleanup_stale_shm(storage: SharedArrayStorage) -> None:
    """Clean up stale shared memory from previous runs."""
    shm_names = [
        get_shm_name_for_storage(storage, storage.SHM_REGRETS),
        get_shm_name_for_storage(storage, storage.SHM_STRATEGY),
        get_shm_name_for_storage(storage, storage.SHM_ACTIONS),
        get_shm_name_for_storage(storage, storage.SHM_REACH),
        get_shm_name_for_storage(storage, storage.SHM_UTILITY),
    ]

    for name in shm_names:
        try:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass


def cleanup(storage: SharedArrayStorage) -> None:
    """Clean up shared memory (coordinator unlinks, workers just close)."""
    handles = [
        storage.state.shm_regrets,
        storage.state.shm_strategy,
        storage.state.shm_actions,
        storage.state.shm_reach,
        storage.state.shm_utility,
    ]

    for shm in handles:
        if shm is not None:
            try:
                shm.close()
                if storage.is_coordinator:
                    shm.unlink()
            except FileNotFoundError:
                pass

    storage.state.shm_regrets = None
    storage.state.shm_strategy = None
    storage.state.shm_actions = None
    storage.state.shm_reach = None
    storage.state.shm_utility = None
