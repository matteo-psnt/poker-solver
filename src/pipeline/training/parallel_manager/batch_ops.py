"""Batch execution and result aggregation operations."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypedDict, cast

from src.pipeline.training.parallel_protocol import JobType

from .gather import gather_worker_results

if TYPE_CHECKING:
    from .manager import SharedArrayWorkerManager


class BatchResult(TypedDict):
    """Aggregated output from one manager batch execution."""

    utilities: list[float]
    batch_time: float
    total_iterations: int
    worker_results: list[dict[str, object]]
    num_infosets: int
    dropped_unknown_id_updates: int
    max_worker_capacity: float
    resized: bool
    capacity: int
    interrupted: bool


def run_batch(
    manager: SharedArrayWorkerManager,
    iterations_per_worker: list[int],
    batch_id: int = 0,
    start_iteration: int = 0,
    timeout: float = 600.0,
    verbose: bool = True,
    auto_resize: bool = True,
) -> BatchResult:
    """Run a batch of iterations across all workers and aggregate metrics."""
    batch_start = time.time()

    iteration_offset = start_iteration
    active_workers = 0
    for worker_id, num_iterations in enumerate(iterations_per_worker):
        if num_iterations > 0:
            manager.job_queue.put(
                {
                    "type": JobType.RUN_ITERATIONS.value,
                    "target_worker": worker_id,
                    "num_iterations": num_iterations,
                    "batch_id": batch_id,
                    "iteration_offset": iteration_offset,
                }
            )
            iteration_offset += num_iterations
            active_workers += 1

    received, interrupted = gather_worker_results(
        manager,
        accept=lambda r: "error" in r or r.get("type") == "iterations_done",
        expected=active_workers,
        timeout=timeout,
        description="worker batch results",
        verbose=verbose,
    )

    all_utilities = []
    results = []
    errors = []
    total_owned = 0
    total_dropped = 0
    max_worker_capacity = 0.0
    for result in received:
        if "error" in result:
            errors.append(result)
            print(
                f"[Master] Worker {result['worker_id']} error: {result['error']}",
                flush=True,
            )
        else:
            results.append(result)
            all_utilities.extend(cast(list[float], result.get("utilities", [])))
            total_owned += cast(int, result.get("num_owned_infosets", 0))
            total_dropped += cast(int, result.get("dropped_unknown_id_updates", 0))
            worker_capacity = float(cast(float | int, result.get("capacity_usage", 0.0)))
            max_worker_capacity = max(max_worker_capacity, worker_capacity)

    batch_time = time.time() - batch_start
    total_iters = sum(iterations_per_worker)

    if verbose:
        print(
            f"[Master] Batch {batch_id} complete in {batch_time:.1f}s "
            f"({total_iters / batch_time:.1f} iter/s)",
            flush=True,
        )

    if errors:
        error_details = "\n".join(
            [
                f"  Worker {error['worker_id']}:\n"
                f"    Error: {error['error']}\n"
                f"    Traceback:\n{error.get('traceback', '    (no traceback available)')}"
                for error in errors
            ]
        )
        raise RuntimeError(
            f"{len(errors)} worker(s) failed during batch {batch_id}:\n{error_details}"
        )

    resized = False
    if auto_resize:
        resized = manager.check_and_resize_if_needed(
            max_worker_capacity=max_worker_capacity,
            timeout=timeout,
            verbose=verbose,
        )

    return {
        "utilities": all_utilities,
        "batch_time": batch_time,
        "total_iterations": total_iters,
        "worker_results": results,
        "num_infosets": total_owned,
        "dropped_unknown_id_updates": total_dropped,
        "max_worker_capacity": max_worker_capacity,
        "resized": resized,
        "capacity": manager.capacity,
        "interrupted": interrupted,
    }
