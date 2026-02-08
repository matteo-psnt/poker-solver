"""Shared-memory lifecycle helpers for SharedArrayStorage."""

from __future__ import annotations

import time
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np

from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.shared_array_layout import get_shm_name

if TYPE_CHECKING:
    from src.engine.solver.storage.shared_array.storage import SharedArrayStorage


def get_shm_name_for_storage(storage: SharedArrayStorage, base: str) -> str:
    """Get session-namespaced shared memory name."""
    return get_shm_name(base, storage.session_id)


def create_shared_memory(storage: SharedArrayStorage) -> None:
    """Create all shared memory segments (coordinator only)."""
    cleanup_stale_shm(storage)

    sizes: dict[str, int] = {}
    for spec in ARRAY_SPECS:
        size = spec.nbytes(storage.capacity, storage.max_actions)
        sizes[spec.attr] = size
        setattr(
            storage.state,
            spec.shm_attr,
            shared_memory.SharedMemory(
                create=True,
                size=size,
                name=get_shm_name_for_storage(storage, spec.shm_base),
            ),
        )

    create_numpy_views(storage)

    for spec in ARRAY_SPECS:
        getattr(storage, spec.attr).fill(0)

    print(
        "Master created shared memory: "
        f"regrets={sizes['shared_regrets'] // 1024 // 1024}MB, "
        f"strategy={sizes['shared_strategy_sum'] // 1024 // 1024}MB"
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
            for spec in ARRAY_SPECS:
                setattr(
                    storage.state,
                    spec.shm_attr,
                    shared_memory.SharedMemory(
                        name=get_shm_name_for_storage(storage, spec.shm_base)
                    ),
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
    if any(getattr(storage.state, spec.shm_attr) is None for spec in ARRAY_SPECS):
        raise RuntimeError("Shared memory buffers are not initialized")

    for spec in ARRAY_SPECS:
        shm = getattr(storage.state, spec.shm_attr)
        setattr(
            storage,
            spec.attr,
            np.ndarray(
                spec.shape(storage.capacity, storage.max_actions),
                dtype=spec.dtype,
                buffer=shm.buf,
            ),
        )


def cleanup_stale_shm(storage: SharedArrayStorage) -> None:
    """Clean up stale shared memory from previous runs."""
    for spec in ARRAY_SPECS:
        try:
            stale = shared_memory.SharedMemory(
                name=get_shm_name_for_storage(storage, spec.shm_base)
            )
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass


def cleanup(storage: SharedArrayStorage) -> None:
    """Clean up shared memory (coordinator unlinks, workers just close)."""
    for spec in ARRAY_SPECS:
        shm = getattr(storage.state, spec.shm_attr)
        if shm is not None:
            try:
                shm.close()
                if storage.is_coordinator:
                    shm.unlink()
            except FileNotFoundError:
                pass
        setattr(storage.state, spec.shm_attr, None)
