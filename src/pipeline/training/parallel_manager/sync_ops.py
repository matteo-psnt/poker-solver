"""ID exchange and update-application synchronization operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from src.pipeline.training.parallel_protocol import JobType

from .gather import gather_worker_results

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

    acks, _ = gather_worker_results(
        manager,
        accept=lambda r: r.get("type") == "exchange_ids_ack",
        expected=manager.num_workers,
        timeout=timeout,
        description="ID exchange acks",
    )
    total_owned = sum(cast(int, result.get("num_owned", 0)) for result in acks)

    if verbose:
        print("[Master] ID exchange complete", flush=True)

    return {"total_owned": total_owned, "acks": acks}
