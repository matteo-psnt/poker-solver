"""Worker startup and shutdown lifecycle for parallel training."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, cast

from src.engine.solver.storage.shared_array import SharedArrayStorage
from src.pipeline.training.parallel_protocol import JobType
from src.pipeline.training.parallel_worker import _worker_loop

if TYPE_CHECKING:
    from .manager import MessageQueue, SharedArrayWorkerManager


def initialize_runtime(manager: SharedArrayWorkerManager) -> None:
    """Create coordinator storage and communication primitives."""
    manager.ready_event = mp.Event()

    print(
        f"[Master] Creating shared memory (session={manager.session_id})...",
        flush=True,
    )
    manager.storage = SharedArrayStorage(
        num_workers=manager.num_workers,
        worker_id=0,
        session_id=manager.session_id,
        initial_capacity=manager.capacity,
        max_actions=manager.max_actions,
        is_coordinator=True,
        checkpoint_dir=Path(manager.checkpoint_dir) if manager.checkpoint_dir else None,
        ready_event=manager.ready_event,
        load_checkpoint_on_init=False,
        zarr_compression_level=manager.config.storage.zarr_compression_level,
        zarr_chunk_size=manager.config.storage.zarr_chunk_size,
    )

    total_mb = manager.capacity * manager.max_actions * 4 * 2 // 1024 // 1024
    print(f"[Master] Shared memory created: {total_mb}MB total", flush=True)

    manager.job_queue = cast("MessageQueue", mp.Queue())
    manager.result_queue = cast("MessageQueue", mp.Queue())
    manager.update_queues = [cast("MessageQueue", mp.Queue()) for _ in range(manager.num_workers)]
    manager.id_request_queues = [
        cast("MessageQueue", mp.Queue()) for _ in range(manager.num_workers)
    ]
    manager.id_response_queues = [
        cast("MessageQueue", mp.Queue()) for _ in range(manager.num_workers)
    ]


def start_workers(manager: SharedArrayWorkerManager) -> None:
    """Start all worker processes."""
    print(f"[Master] Starting {manager.num_workers} workers...", flush=True)

    for worker_id in range(manager.num_workers):
        process = mp.Process(
            target=_worker_loop,
            args=(
                worker_id,
                manager.num_workers,
                manager.session_id,
                manager.config,
                manager.serialized_action_model,
                manager.serialized_card_abstraction,
                manager.base_seed,
                manager.job_queue,
                manager.result_queue,
                manager.update_queues,
                manager.id_request_queues,
                manager.id_response_queues,
                manager.capacity,
                manager.max_actions,
                manager.checkpoint_dir,
                manager.ready_event,
            ),
        )
        process.start()
        manager.processes.append(process)

    print(f"[Master] All {manager.num_workers} workers started", flush=True)


def shutdown(manager: SharedArrayWorkerManager) -> None:
    """Shutdown all workers and cleanup shared memory."""
    print("[Master] Shutting down workers...", flush=True)

    for _ in range(manager.num_workers):
        manager.job_queue.put({"type": JobType.SHUTDOWN.value})

    for process in manager.processes:
        process.join(timeout=10)
        if process.is_alive():
            print(f"[Master] Force terminating worker {process.pid}", flush=True)
            process.terminate()

    manager.processes.clear()
    manager.storage.cleanup()
    print("[Master] All workers shut down", flush=True)
