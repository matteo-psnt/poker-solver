"""ID exchange and update-application synchronization operations."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, cast

from src.pipeline.training.parallel_protocol import JobType

if TYPE_CHECKING:
    from .manager import SharedArrayWorkerManager


def exchange_ids(
    manager: SharedArrayWorkerManager,
    timeout: float = 60.0,
    verbose: bool = True,
) -> dict[str, object]:
    """
    Trigger batched ID exchange between workers.

    Workers send pending ID requests to owners and process responses.
    """
    for worker_id in range(manager.num_workers):
        manager.job_queue.put({"type": JobType.EXCHANGE_IDS.value, "target_worker": worker_id})

    acks = []
    total_owned = 0
    for _ in range(manager.num_workers):
        try:
            raw_result = manager.result_queue.get(timeout=timeout)
            if not isinstance(raw_result, dict):
                continue
            result = cast(dict[str, object], raw_result)
            if result.get("type") == "exchange_ids_ack":
                acks.append(result)
                total_owned += cast(int, result.get("num_owned", 0))
        except queue.Empty:
            raise RuntimeError("Timeout waiting for ID exchange acks")

    if verbose:
        print("[Master] ID exchange complete", flush=True)

    return {"total_owned": total_owned, "acks": acks}


def apply_pending_updates(
    manager: SharedArrayWorkerManager,
    timeout: float = 60.0,
    verbose: bool = True,
) -> dict[str, object]:
    """Trigger workers to apply pending cross-partition updates."""
    for _ in range(manager.num_workers):
        manager.job_queue.put({"type": JobType.APPLY_UPDATES.value})

    acks = []
    for _ in range(manager.num_workers):
        try:
            raw_result = manager.result_queue.get(timeout=timeout)
            if not isinstance(raw_result, dict):
                continue
            result = cast(dict[str, object], raw_result)
            if result.get("type") == "apply_updates_ack":
                acks.append(result)
        except queue.Empty:
            raise RuntimeError("Timeout waiting for apply updates acks")

    if verbose:
        print("[Master] Pending updates applied", flush=True)

    return {"acks": acks}
