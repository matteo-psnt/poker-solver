"""Batch execution and result aggregation operations."""

from __future__ import annotations

import queue
import time
from typing import TYPE_CHECKING, TypedDict, cast

from src.pipeline.training.parallel_protocol import JobType

if TYPE_CHECKING:
    from .manager import SharedArrayWorkerManager


class BatchResult(TypedDict):
    """Aggregated output from one manager batch execution."""

    utilities: list[float]
    batch_time: float
    total_iterations: int
    worker_results: list[dict[str, object]]
    num_infosets: int
    max_worker_capacity: float
    resized: bool
    capacity: int
    interrupted: bool
    fallback_stats: dict[str, int | float]


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
                    "num_iterations": num_iterations,
                    "batch_id": batch_id,
                    "iteration_offset": iteration_offset,
                }
            )
            iteration_offset += num_iterations
            active_workers += 1

    all_utilities = []
    results = []
    errors = []
    total_owned = 0
    max_worker_capacity = 0.0
    interrupted = False

    received = 0
    while received < active_workers:
        try:
            raw_result = manager.result_queue.get(timeout=timeout)
            if not isinstance(raw_result, dict):
                if verbose:
                    print(f"[Master] Ignoring unexpected non-dict result: {raw_result}", flush=True)
                continue
            result = cast(dict[str, object], raw_result)

            if "error" in result:
                errors.append(result)
                print(
                    f"[Master] Worker {result['worker_id']} error: {result['error']}",
                    flush=True,
                )
                received += 1
            elif result.get("type") == "iterations_done":
                results.append(result)
                all_utilities.extend(cast(list[float], result.get("utilities", [])))
                total_owned += cast(int, result.get("num_owned_infosets", 0))
                worker_capacity = float(cast(float | int, result.get("capacity_usage", 0.0)))
                max_worker_capacity = max(max_worker_capacity, worker_capacity)
                fallback_stats = result.get("fallback_stats")
                if isinstance(fallback_stats, dict):
                    worker_id = cast(int, result["worker_id"])
                    manager.fallback_stats_by_worker[worker_id] = cast(
                        dict[str, int | float], fallback_stats
                    )
                received += 1
            else:
                if verbose:
                    print(f"[Master] Ignoring unexpected result: {result}", flush=True)
        except queue.Empty:
            raise RuntimeError(
                f"Timeout waiting for workers ({len(results)}/{active_workers} received)"
            )
        except KeyboardInterrupt:
            interrupted = True
            if verbose:
                print(
                    "⚠️  Interrupt received; waiting for current batch to finish...",
                    flush=True,
                )

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

    total_lookups = sum(
        stats.get("total_lookups", 0) for stats in manager.fallback_stats_by_worker.values()
    )
    fallback_count = sum(
        stats.get("fallback_count", 0) for stats in manager.fallback_stats_by_worker.values()
    )
    fallback_stats = {
        "total_lookups": total_lookups,
        "fallback_count": fallback_count,
        "fallback_rate": (fallback_count / total_lookups) if total_lookups > 0 else 0.0,
    }

    return {
        "utilities": all_utilities,
        "batch_time": batch_time,
        "total_iterations": total_iters,
        "worker_results": results,
        "num_infosets": total_owned,
        "max_worker_capacity": max_worker_capacity,
        "resized": resized,
        "capacity": manager.capacity,
        "interrupted": interrupted,
        "fallback_stats": fallback_stats,
    }
