"""Shared-array resize operations for worker manager."""

from __future__ import annotations

import queue
import time
from typing import TYPE_CHECKING, cast

from src.pipeline.training.parallel_protocol import JobType

if TYPE_CHECKING:
    from .manager import SharedArrayWorkerManager


def check_and_resize_if_needed(
    manager: SharedArrayWorkerManager,
    max_worker_capacity: float,
    timeout: float = 120.0,
    verbose: bool = True,
) -> bool:
    """Check whether shared storage is near capacity and trigger resize."""
    if max_worker_capacity < manager.storage.CAPACITY_THRESHOLD:
        return False

    new_max = int(manager.capacity * manager.storage.GROWTH_FACTOR)
    if verbose:
        print(
            "[Master] Storage resize triggered: "
            f"worker at {max_worker_capacity:.1%} capacity, "
            f"growing {manager.capacity:,} -> {new_max:,}",
            flush=True,
        )

    return resize_storage(manager, new_capacity=new_max, timeout=timeout, verbose=verbose)


def resize_storage(
    manager: SharedArrayWorkerManager,
    new_capacity: int,
    timeout: float = 120.0,
    verbose: bool = True,
) -> bool:
    """
    Resize storage to new capacity (stop-the-world operation).

    1. Coordinator creates new larger shared memory
    2. Existing data is copied by storage backend
    3. Workers are notified to reattach
    4. Coordinator waits for all resize acknowledgements
    """
    resize_start = time.time()

    if verbose:
        print(
            f"[Master] Resizing storage: {manager.storage.capacity:,} -> {new_capacity:,}",
            flush=True,
        )

    manager.storage.resize(new_capacity)
    new_session_id = manager.storage.session_id
    manager.capacity = new_capacity

    if verbose:
        print(
            f"[Master] New shared memory created (session={new_session_id}), notifying workers...",
            flush=True,
        )

    for _ in range(manager.num_workers):
        manager.job_queue.put(
            {
                "type": JobType.RESIZE_STORAGE.value,
                "new_session_id": new_session_id,
                "new_capacity": new_capacity,
            }
        )

    acks = []
    for _ in range(manager.num_workers):
        try:
            raw_result = manager.result_queue.get(timeout=timeout)
            if not isinstance(raw_result, dict):
                continue
            result = cast(dict[str, object], raw_result)
            if result.get("type") == "resize_ack":
                acks.append(result)
                if verbose:
                    worker_id = cast(int, result["worker_id"])
                    new_range = cast(tuple[int, int], result["new_id_range"])
                    print(
                        f"[Master] Worker {worker_id} reattached (range={new_range})",
                        flush=True,
                    )
        except queue.Empty:
            raise RuntimeError(
                f"Timeout waiting for resize acks ({len(acks)}/{manager.num_workers} received)"
            )

    resize_time = time.time() - resize_start
    if verbose:
        print(
            f"[Master] Resize complete in {resize_time:.1f}s, "
            f"new capacity: {new_capacity:,} infosets",
            flush=True,
        )

    return True
