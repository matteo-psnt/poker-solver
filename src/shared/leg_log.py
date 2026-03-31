"""Durable per-leg outcome records for cloud training on Azure Batch.

``.run.json`` records what a *living* process did. It cannot record how an
attempt died: a container killed by OOM, ``maxWallClockTime`` or node loss is
gone before it can write. Batch sees those deaths but retains them for far less
time than the run lives. This is the join.

Under ``<share>/legs/``, per task and per attempt::

    <task>.<attempt>.start.json   run_leg.sh, at entry
    <task>.<attempt>.exit.json    run_leg.sh, from its EXIT trap
    <task>.observed.json          just legs, from Batch's executionInfo

One writer per file, because the share is SMB: no atomic rename, no atomic
append, so a read-modify-write from two sides would interleave. Start and exit
are separate because ``write_text`` truncates -- a kill mid-write would
otherwise make the leg vanish from the listing entirely, in exactly the SIGKILL
window this exists for. Numbered by attempt because a Batch retry reuses the
task id and Batch describes only the latest, so one file would let the retry
erase the OOM that caused it.

The node alone can tell a hang from a crash from a cancellation; Batch alone can
explain a death the node did not survive.

SNAPSHOTs in :mod:`src.shared.records` terms, share-scoped -- which is why they
are written directly rather than through a temporary file: SMB has no atomic
rename, so the per-file layout carries the safety instead.

Stdlib-only AND 3.10-compatible, both enforced by tests: ``run_leg.sh`` imports
this with the NODE's system ``python3``, which on the pinned Ubuntu 22.04 image
is 3.10 -- not the 3.12+ this project is developed against. ``datetime.UTC`` is
3.11+, and importing it here raised inside a call whose errors are swallowed, so
the whole feature was silently dead on the only machine that runs it.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared import records

LEGS_DIRNAME = "legs"

EVENT_STARTED = "started"
EVENT_FINISHED = "finished"

START_SUFFIX = ".start.json"
EXIT_SUFFIX = ".exit.json"
OBSERVED_SUFFIX = ".observed.json"

# Batch ``executionInfo.result`` / task state -> coarse exit cause. The dead
# process cannot report these; Batch can. Note that FAILURE conflates an
# in-container error with an OOM-kill, exactly as Modal's does -- the node is
# gone either way, and only the published leg log can tell them apart.
_OBSERVED_CAUSE_BY_RESULT: dict[str, str] = {
    "success": "completed",
    "failure": "failed",
}

# Finer than Batch's success/failure, because a terminal cause suppresses
# reconciliation: a WRONG one loses the observer half of the join permanently.
#   timeout    the RUN_TIMEOUT guard expired (124) -- a hang, not a crash
#   killed     SIGKILL from outside (137) -- the OOM killer. `timeout` returns
#              124 even when its own --kill-after fires, so 137 is never it
#   cancelled  the wrapper took SIGTERM -- `just cancel`, or maxWallClockTime
#   partial    an evaluate leg scored some rungs and failed others; it exits 0
#              for Batch's retry economics, which is not a claim of success
CAUSE_COMPLETED = "completed"
CAUSE_FAILED = "failed"
CAUSE_TIMEOUT = "timeout"
CAUSE_KILLED = "killed"
CAUSE_CANCELLED = "cancelled"
CAUSE_PARTIAL = "partial"

TERMINAL_CAUSES = frozenset(
    {
        CAUSE_COMPLETED,
        CAUSE_FAILED,
        CAUSE_TIMEOUT,
        CAUSE_KILLED,
        CAUSE_CANCELLED,
        CAUSE_PARTIAL,
    }
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def legs_dir(share: str | os.PathLike[str]) -> Path:
    return Path(share) / LEGS_DIRNAME


def write_node_record(
    share: str | os.PathLike[str],
    *,
    task_id: str,
    event: str,
    run_id: str = "",
    job_id: str = "",
    op: str = "",
    config: str = "",
    target_iteration: str = "",
    node_id: str = "",
    exit_code: int | None = None,
    cause: str | None = None,
) -> Path:
    """Record what the node knows about this leg. One file per event, per attempt.

    Never overwrites -- see the module docstring for why the attempt number and
    the start/exit split are both load-bearing.
    """
    directory = legs_dir(share)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = (
        _next_attempt(directory, task_id)
        if event == EVENT_STARTED
        else _latest_attempt(directory, task_id)
    )
    record = {
        "source": "node",
        "task_id": task_id,
        "attempt": attempt,
        "job_id": job_id,
        "run_id": run_id,
        "op": op or "train",
        "config": config,
        "target_iteration": target_iteration,
        "node_id": node_id,
        "event": event,
        "ts": _utcnow(),
        "exit_code": exit_code,
        "cause": cause,
    }
    suffix = START_SUFFIX if event == EVENT_STARTED else EXIT_SUFFIX
    path = directory / f"{task_id}.{attempt}{suffix}"
    records.write_snapshot(path, record, records.REGISTRY[f"legs/*{suffix}"])
    return path


def _next_attempt(directory: Path, task_id: str) -> int:
    """1 for a first run, 2 for Batch's first retry, and so on."""
    return len(list(directory.glob(f"{_escape(task_id)}.*{START_SUFFIX}"))) + 1


def _latest_attempt(directory: Path, task_id: str) -> int:
    """The attempt the terminal record belongs to.

    Derived from disk rather than passed through the shell: the exit trap may
    have lost anything the entry point computed.
    """
    return max(_next_attempt(directory, task_id) - 1, 1)


def _escape(task_id: str) -> str:
    """Glob-safe task id. Batch ids are ``[A-Za-z0-9_-]`` but never assume it."""
    return glob.escape(task_id)


def write_observed_record(
    share: str | os.PathLike[str],
    *,
    task_id: str,
    job_id: str,
    state: str,
    result: str | None = None,
    exit_code: int | None = None,
    failure: dict[str, Any] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    node_id: str = "",
) -> Path:
    """Record what Batch says happened, from the client.

    Its own filename, so the two sides never contend. Joins to the task's LATEST
    attempt -- Batch describes no other.
    """
    directory = legs_dir(share)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "source": "batch",
        "task_id": task_id,
        "job_id": job_id,
        "state": state,
        "result": result,
        "exit_code": exit_code,
        "failure": failure,
        "start_time": start_time,
        "end_time": end_time,
        "node_id": node_id,
        "observed_at": _utcnow(),
    }
    path = directory / f"{task_id}{OBSERVED_SUFFIX}"
    records.write_snapshot(path, record, records.REGISTRY[f"legs/*{OBSERVED_SUFFIX}"])
    return path


def observed_cause(record: dict[str, Any]) -> str:
    """Coarse exit cause from a Batch-observed record.

    Anything not ``completed`` reports its own state rather than guessing:
    ``preparing`` is more useful than a wrong ``failed``.
    """
    state = (record.get("state") or "").lower()
    if state != "completed":
        return state or "unknown"
    result = (record.get("result") or "").lower()
    if result in _OBSERVED_CAUSE_BY_RESULT:
        return _OBSERVED_CAUSE_BY_RESULT[result]
    return "unknown"


def read_legs(share: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Join the node and observer records into one row per attempt, newest last.

    ``cause`` prefers the node's own account when it reached a terminal event --
    it distinguishes a hang from a crash where Batch reports both as
    ``failure``. Otherwise the observer's view is all there is.
    """
    directory = legs_dir(share)
    if not directory.is_dir():
        return []

    # Keyed by (task_id, attempt): a Batch retry reuses the task id, and the
    # failed attempt is the one worth keeping.
    attempts: dict[tuple[str, int], dict[str, Any]] = {}
    for suffix, slot in ((START_SUFFIX, "start"), (EXIT_SUFFIX, "exit")):
        for path in sorted(directory.glob(f"*{suffix}")):
            record = _load(path)
            if record and record.get("task_id"):
                key = (record["task_id"], int(record.get("attempt", 1)))
                attempts.setdefault(key, {})[slot] = record

    # Batch describes only the LATEST attempt of a task, so its record joins
    # there and nowhere else -- attaching it to an earlier attempt would explain
    # the wrong death.
    observed: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"*{OBSERVED_SUFFIX}")):
        record = _load(path)
        if record and record.get("task_id"):
            observed[record["task_id"]] = record
    latest = {task: max(a for t, a in attempts if t == task) for task, _ in attempts}

    # Known to Batch but never recorded by the node -- killed before its first
    # write. Still gets a row: a leg that vanishes is indistinguishable from one
    # that never ran.
    for task_id in observed:
        if task_id not in latest:
            attempts[(task_id, 1)] = {}
            latest[task_id] = 1

    joined = []
    for (task_id, attempt), sources in attempts.items():
        start, exit_record = sources.get("start", {}), sources.get("exit", {})
        batch = observed.get(task_id, {}) if latest.get(task_id) == attempt else {}
        node_cause = exit_record.get("cause")
        if node_cause in TERMINAL_CAUSES:
            cause, cause_source = node_cause, "node"
        elif batch:
            cause, cause_source = observed_cause(batch), "batch"
        elif start or exit_record:
            # Started and never finished, with nothing observed yet. Not the same
            # as "running": the record cannot tell them apart, and saying so is
            # better than picking one.
            cause, cause_source = "unresolved", "node"
        else:
            cause, cause_source = "unknown", "none"
        node = exit_record or start
        joined.append(
            {
                "task_id": task_id,
                "attempt": attempt,
                "job_id": node.get("job_id") or batch.get("job_id", ""),
                "run_id": node.get("run_id", ""),
                "op": node.get("op", ""),
                "config": node.get("config", ""),
                "target_iteration": node.get("target_iteration", ""),
                "cause": cause,
                "cause_source": cause_source,
                # Not dict.get's default: the node record always carries the
                # key and leaves it null, so a default would never be reached
                # and the observer's code -- the only one a killed leg has --
                # would be dropped.
                "exit_code": _first_not_none(exit_record.get("exit_code"), batch.get("exit_code")),
                # From the node's own records: an observer record exists only
                # for an unresolved leg, so reading times off it would blank
                # every cleanly-finished one.
                "started_at": _first_not_none(start.get("ts"), batch.get("start_time")),
                "ended_at": _first_not_none(exit_record.get("ts"), batch.get("end_time")),
                "failure": batch.get("failure"),
                "node_id": node.get("node_id") or batch.get("node_id", ""),
            }
        )
    # ended_at then started_at, so a still-running leg sorts beside the one it
    # followed rather than at the front.
    joined.sort(key=lambda r: (r.get("ended_at") or r.get("started_at") or "", r["task_id"]))
    return joined


def unresolved_task_ids(share: str | os.PathLike[str]) -> list[str]:
    """Tasks whose node record never reached a terminal event -- exactly the
    ones worth asking Batch about."""
    return sorted(
        {row["task_id"] for row in read_legs(share) if row["cause"] not in TERMINAL_CAUSES}
    )


def _first_not_none(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _load(path: Path) -> dict[str, Any] | None:
    """Skipped, never fatal: a half-written file is the expected residue of a
    leg killed mid-write, and must not take down the listing that explains it."""
    return records.read_snapshot(path)


def format_table(rows: Iterable[dict[str, Any]]) -> str:
    """Compact fixed-width listing, one row per leg."""
    columns = ("task_id", "attempt", "op", "run_id", "cause", "exit_code", "ended_at")
    materialised = [{c: _cell(r.get(c)) for c in columns} for r in rows]
    if not materialised:
        return "  no leg records on the share"
    widths = {c: max(len(c), *(len(r[c]) for r in materialised)) for c in columns}
    lines = ["  " + "  ".join(c.ljust(widths[c]) for c in columns)]
    lines.append("  " + "  ".join("-" * widths[c] for c in columns))
    lines += ["  " + "  ".join(r[c].ljust(widths[c]) for c in columns) for r in materialised]
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def reconcile(share: str | os.PathLike[str], tasks: Iterable[dict[str, Any]]) -> list[str]:
    """Write observer records for legs the node never explained.

    ``tasks`` is `batch.list_jobs_with_tasks` output, flattened so each task
    carries its `job`. Only unresolved legs: otherwise the cost scales with
    history rather than with open questions. Returns the ids newly explained.
    """
    open_questions = set(unresolved_task_ids(share))
    explained = []
    for task in tasks:
        task_id = task.get("task")
        if not task_id or task_id not in open_questions:
            continue
        write_observed_record(
            share,
            task_id=task_id,
            job_id=task.get("job", ""),
            state=task.get("state") or "",
            result=task.get("result"),
            exit_code=task.get("exit_code"),
            failure=task.get("failure"),
            start_time=task.get("start_time"),
            end_time=task.get("end_time"),
            node_id=task.get("node") or "",
        )
        explained.append(task_id)
    return explained
