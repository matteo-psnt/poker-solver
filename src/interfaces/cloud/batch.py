"""The Batch data plane: jobs, tasks and the pool.

Every function here returns plain data rather than SDK objects, so the headless
commands can render it and ``--json`` can serialise it without knowing that
Azure exists.

Authentication is the ``az login`` session, and **explicitly
``AzureCliCredential`` rather than ``DefaultAzureCredential``** -- see
:func:`client` for why that distinction is load-bearing here. Shared-key auth
is disabled on this account (the pool runs in UserSubscription mode, where
Entra ID is the only way in), so there is no fallback to configure.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

from azure.batch import BatchClient
from azure.batch.models import (
    BatchJobConstraints,
    BatchJobCreateOptions,
    BatchPoolEvaluateAutoScaleOptions,
    BatchPoolInfo,
    BatchTaskConstraints,
    BatchTaskCreateOptions,
    EnvironmentSetting,
)
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import AzureCliCredential

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.spec import LegSpec, daily_job_id, leg_command, suffixed_job_id

# Batch calls are latency-bound round trips, not work, so independent ones are
# worth issuing together. 16 matches the share downloader.
_PARALLEL_CALLS = 16

JOB_MAX_WALL_CLOCK = timedelta(days=2)
TASK_MAX_WALL_CLOCK = timedelta(days=1)
TASK_RETRIES = 2


def client(config: CloudConfig) -> BatchClient:
    """Build an authenticated data-plane client.

    ``AzureCliCredential``, NOT ``DefaultAzureCredential``. The default chain
    probes for a managed identity before it reaches the CLI credential, and on
    a laptop that probe is a request to the link-local IMDS address
    ``169.254.169.254`` -- which on this machine does not refuse the connection,
    it simply never answers. Measured: every data-plane call hung past 120s with
    the default chain and returned in 1.3s with this one, while
    ``az account get-access-token`` was healthy throughout. A credential chain
    that silently depends on how the network treats an unroutable address is not
    a credential chain worth having when the intended source is known.
    """
    return BatchClient(endpoint=config.batch_endpoint, credential=AzureCliCredential())


def _task_record(task: Any) -> dict[str, Any]:
    """One task, in this project's vocabulary rather than Batch's.

    Everything `leg_log.reconcile` needs to explain a leg the node never got to
    account for: `result` separates a task that finished from one that merely
    stopped, and the timestamps bound a death the EXIT trap could not report.
    """
    execution = task.execution_info
    return {
        "task": task.id,
        "state": str(task.state) if task.state else None,
        "result": str(execution.result) if execution is not None and execution.result else None,
        "exit_code": execution.exit_code if execution is not None else None,
        "failure": _failure(execution),
        "start_time": _isoformat(execution.start_time) if execution is not None else None,
        "end_time": _isoformat(execution.end_time) if execution is not None else None,
        "node": task.node_info.node_id if task.node_info is not None else None,
    }


def _isoformat(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _failure(execution: Any) -> dict[str, Any] | None:
    """Batch's failure detail, kept as data rather than a stringified object."""
    info = execution.failure_info if execution is not None else None
    if info is None:
        return None
    return {"code": info.code, "message": info.message, "category": str(info.category or "")}


def list_jobs(batch: BatchClient) -> list[dict[str, Any]]:
    """Every job, WITHOUT its tasks. One call, ~1.1s against 44 jobs.

    Split from the tasks deliberately. Listing tasks costs ~0.39s per job, so
    the caller that wants only the live ones must be able to decide WHICH jobs
    are worth that before paying it -- see :func:`attach_tasks`.
    """
    return [
        {"job": job.id, "state": str(job.state) if job.state else None} for job in batch.list_jobs()
    ]


def attach_tasks(
    batch: BatchClient,
    jobs: list[dict[str, Any]],
    *,
    want: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Fill in ``tasks`` for the jobs ``want`` selects; ``[]`` for the rest.

    Two costs are removed here, and the first is the larger. Tasks used to be
    listed for EVERY job and then discarded by the caller: 44 jobs at ~0.39s to
    render the 2 that were active. Whatever survives the filter is then fetched
    CONCURRENTLY, because the calls are independent and the latency is the
    round trip, not the work.

    ``exit_code`` is surfaced beside ``state`` because a task that completed is
    not necessarily a task that succeeded, and the two together are what tells
    a drained pool from a failed leg.
    """
    wanted = [job for job in jobs if want is None or want(job)]
    if not wanted:
        return [{**job, "tasks": []} for job in jobs]

    def _tasks(job: dict[str, Any]) -> list[dict[str, Any]]:
        return [_task_record(task) for task in batch.list_tasks(job["job"])]

    with ThreadPoolExecutor(max_workers=min(_PARALLEL_CALLS, len(wanted))) as pool:
        fetched = dict(zip([job["job"] for job in wanted], pool.map(_tasks, wanted), strict=True))
    return [{**job, "tasks": fetched.get(job["job"], [])} for job in jobs]


def task_record(batch: BatchClient, job_id: str, task_id: str) -> dict[str, Any] | None:
    """One task, asked for by name. ``None`` if Batch has forgotten it.

    The targeted alternative to enumerating every job: a reconcile knows the
    exact (job, task) pairs it has open questions about, so asking about those
    costs one call each rather than one call per job in the account's history.
    """
    try:
        return {**_task_record(batch.get_task(job_id, task_id)), "job": job_id}
    except ResourceNotFoundError:
        return None


def pool_status(batch: BatchClient, pool_id: str) -> dict[str, Any]:
    """Node counts, and -- the point of the command -- why a resize failed.

    Batch reports every allocation problem as a generic ``AllocationFailed``.
    The actual cause (a Gen1-vs-Gen2 image, a policy denial, quota) is escaped
    JSON inside a resize error's values, and reading it is the difference
    between a one-line fix and an afternoon: it is how the Gen2-only
    requirement on ``als_v6`` was found. Every value is kept rather than only
    the ``*Json`` ones, since a cause that arrives under an unfamiliar name is
    exactly the one worth seeing.
    """
    pool = batch.get_pool(pool_id)
    errors = [
        {
            "code": error.code,
            "message": error.message,
            "values": {value.name: value.value for value in (error.error_values or [])},
        }
        for error in (pool.resize_errors or [])
    ]
    return {
        "pool_id": pool.id,
        "allocation_state": str(pool.allocation_state) if pool.allocation_state else None,
        "current_dedicated_nodes": pool.current_dedicated_nodes,
        "target_dedicated_nodes": pool.target_dedicated_nodes,
        "vm_size": pool.vm_size,
        "resize_errors": errors,
    }


def evaluate_autoscale(batch: BatchClient, pool_id: str, formula: str) -> dict[str, Any]:
    """Evaluate a formula server-side, returning BOTH its variables and its error.

    The error half is the reason this exists. An invalid or throwing formula
    still returns partial ``results``, so reporting results alone makes a broken
    formula look healthy -- which is how a ``#`` comment (Batch wants ``//``)
    and a one-argument ``GetSample`` both went unnoticed while the pool quietly
    stopped scaling.
    """
    run = batch.evaluate_pool_auto_scale(
        pool_id, BatchPoolEvaluateAutoScaleOptions(auto_scale_formula=formula)
    )
    variables = [part.strip() for part in (run.results or "").split(";") if part.strip()]
    error = None
    if run.error is not None:
        error = {
            "code": run.error.code,
            "message": run.error.message,
            "values": {value.name: value.value for value in (run.error.error_values or [])},
        }
    return {"variables": variables, "error": error}


def ensure_job(batch: BatchClient, pool_id: str, now: datetime) -> str:
    """Return an id for a job that can accept tasks, creating it if needed.

    One job per UTC day, created on demand. The fallback matters more than it
    looks: a job that has been stopped answers ``JobCompleted`` to every task
    creation, so without a second id one ``panic`` would block all submissions
    until midnight UTC.

    The two constraints are set explicitly rather than left to service
    defaults, because they are billing controls and a billing control that
    depends on a default is one upgrade away from not existing.
    ``max_task_retry_count=0`` at the JOB level is deliberate -- per-task
    retries are set per task, and a job-wide retry would re-burn a node on a
    deterministic failure.
    """
    daily = daily_job_id(now)
    try:
        existing = batch.get_job(daily)
    except ResourceNotFoundError:
        _create_job(batch, daily, pool_id)
        return daily

    if str(existing.state).rsplit(".", 1)[-1].lower() == "active":
        return daily

    fallback = suffixed_job_id(now)
    _create_job(batch, fallback, pool_id)
    return fallback


def _create_job(batch: BatchClient, job_id: str, pool_id: str) -> None:
    """Create a job, tolerating a concurrent submission that got there first."""
    try:
        batch.create_job(
            BatchJobCreateOptions(
                id=job_id,
                pool_info=BatchPoolInfo(pool_id=pool_id),
                constraints=BatchJobConstraints(
                    max_wall_clock_time=JOB_MAX_WALL_CLOCK, max_task_retry_count=0
                ),
            )
        )
    except ResourceExistsError:
        return


def submit_leg(
    batch: BatchClient,
    job_id: str,
    task_id: str,
    spec: LegSpec,
    *,
    max_wall_clock: timedelta = TASK_MAX_WALL_CLOCK,
    retries: int = TASK_RETRIES,
) -> None:
    """Queue one leg as a Batch task.

    ``max_wall_clock`` is the only thing standing between a hung task and
    indefinite billing: the pool scales down on pending-task count, so a task
    that never exits keeps its node alive forever. It is deliberately far above
    any real leg and far below a month of compute, and it is the outer of two
    nested ceilings -- ``RUN_TIMEOUT`` kills the training process first and
    still runs the wrapper's publish trap, losing at most one rung interval.

    Retries are always safe: the iteration target is absolute and the wrapper
    derives a stable run id from the task id, so a retry continues the same run
    rather than starting a second one from zero.
    """
    batch.create_task(
        job_id,
        BatchTaskCreateOptions(
            id=task_id,
            command_line=f"/bin/bash -c '{leg_command(spec.code_snapshot)}'",
            environment_settings=[
                EnvironmentSetting(name=name, value=value)
                for name, value in spec.environment().items()
            ],
            constraints=BatchTaskConstraints(
                max_wall_clock_time=max_wall_clock, max_task_retry_count=retries
            ),
        ),
    )


def cancel_task(batch: BatchClient, job_id: str, task_id: str) -> None:
    """Terminate one task. Its wrapper publishes what it has before exiting."""
    batch.terminate_task(job_id, task_id)


def task_file(batch: BatchClient, job_id: str, task_id: str, file_path: str) -> str:
    """Read one of a task's node-side files (``stdout.txt`` / ``stderr.txt``).

    Node-side, so this only works while the node still exists. The pool scales
    to zero within minutes of a task ending, which is why the published copy on
    the share is the durable path and this one is the live-debugging path.
    """
    stream = batch.download_task_file(job_id, task_id, file_path)
    return b"".join(stream).decode("utf-8", errors="replace")
