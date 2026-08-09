"""Durable per-task outcome records for cloud training on Azure Batch.

``.run.json`` records what a *living* process did. It cannot record how an
attempt died: a container killed by OOM, ``maxWallClockTime`` or node loss is
gone before it can write. Batch sees those deaths but retains them for far less
time than the run lives. This is the join.

Under ``<share>/legs/``, per task and per attempt::

    <task>.<attempt>.start.json   run_task.py, at entry
    <task>.<attempt>.exit.json    run_task.py, from its exit accounting
    <task>.observed.json          poker-solver tasks, from Batch's executionInfo

One writer per file, because the share is SMB: no atomic rename, no atomic
append, so a read-modify-write from two sides would interleave. Start and exit
are separate because ``write_text`` truncates -- a kill mid-write would
otherwise make the task vanish from the listing entirely, in exactly the SIGKILL
window this exists for. Numbered by attempt because a Batch retry reuses the
task id and Batch describes only the latest, so one file would let the retry
erase the OOM that caused it.

The node alone can tell a hang from a crash from a cancellation; Batch alone can
explain a death the node did not survive.

SNAPSHOTs in :mod:`src.shared.records` terms, share-scoped -- which is why they
are written directly rather than through a temporary file: SMB has no atomic
rename, so the per-file layout carries the safety instead.

Stdlib-only -- the wrapper imports this before `uv sync`. Enforced by
``tests/shared/node/test_node_interpreter.py``: the node wrapper imports this
with the NODE's system ``python3``, which on the pinned Ubuntu 22.04 image
is 3.10 -- not the 3.12+ this project is developed against. ``datetime.UTC`` is
3.11+, and importing it here raised inside a call whose errors are swallowed, so
the whole feature was silently dead on the only machine that runs it.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.shared import records
from src.shared.cloudtask import kinds

RECORDS_DIRNAME = "legs"

EVENT_STARTED = "started"
EVENT_FINISHED = "finished"

START_SUFFIX = ".start.json"
EXIT_SUFFIX = ".exit.json"
OBSERVED_SUFFIX = ".observed.json"
PROGRESS_SUFFIX = ".progress.json"

# Batch ``executionInfo.result`` / task state -> coarse exit cause. The dead
# process cannot report these; Batch can. Note that FAILURE conflates an
# in-container error with an OOM-kill, exactly as Modal's does -- the node is
# gone either way, and only the published task log can tell them apart.
_OBSERVED_CAUSE_BY_RESULT: dict[str, str] = {
    "success": "completed",
    "failure": "failed",
}

# Finer than Batch's success/failure, because a terminal cause suppresses
# reconciliation: a WRONG one loses the observer half of the join permanently.
#   timeout    the RUN_TIMEOUT guard expired (124) -- a hang, not a crash
#   killed     SIGKILL from outside (137) -- the OOM killer. `timeout` returns
#              124 even when its own --kill-after fires, so 137 is never it
#   cancelled  the wrapper took SIGTERM -- `cancel`, or maxWallClockTime
#   partial    an evaluate task scored some rungs and failed others; it exits 0
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

"""Causes that mean a node is committed RIGHT NOW
-------------------------------------------------
Batch's own state strings, lowercased by ``observed_cause``. Not-terminal is
NOT the same as live: ``unresolved`` is what a task gets when the node wrote a
start, never wrote an end, and Batch has nothing to say about it either -- which
covers a task that is running and a task that died without stamping an end, and
the record genuinely cannot tell them apart.

That distinction has to be drawn somewhere, because anything reading an open
interval has to decide whether to run it to ``now``. Doing that on
not-terminal credited four attempts abandoned on 2026-08-04 with 455 of the
718 node-hours the cost screen reported, growing by four hours per elapsed
hour. Only a live cause is positive evidence the clock is still running.

``active`` is deliberately absent: a task waiting for a node has no
``started_at`` yet, so it contributes nothing either way, and including it would
invite crediting queue time as node time.
"""
CAUSE_RUNNING = "running"
CAUSE_PREPARING = "preparing"

LIVE_CAUSES = frozenset({CAUSE_RUNNING, CAUSE_PREPARING})


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def tasks_dir(share: str | os.PathLike[str]) -> Path:
    return Path(share) / RECORDS_DIRNAME


TASK_ID_ENV = "AZ_BATCH_TASK_ID"


def current_task_id(default: str = "") -> str:
    """Which Batch task this process IS, if it is one.

    Defined here because this module owns "which task is this" as its primary
    key, and two callers need the same answer: the wrapper writing the task
    record, and the evaluator stamping its document so the number it produced
    can be traced back to the task that produced it.

    Empty off a node, deliberately: an evaluation run from anywhere else
    genuinely has no task, and "" says that where a placeholder would read as a
    task that could be looked up.
    """
    return os.environ.get(TASK_ID_ENV, default)


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
    eval_at: str = "",
    eval_flags: tuple[str, ...] = (),
    workers: int = 0,
    units: float = 0.0,
    units_unit: str = "",
    node_id: str = "",
    code_snapshot: str = "",
    git_commit: str = "",
    git_dirty: str = "",
    git_branch: str = "",
    exit_code: int | None = None,
    cause: str | None = None,
) -> Path:
    """Record what the node knows about this task. One file per event, per attempt.

    Never overwrites -- see the module docstring for why the attempt number and
    the start/exit split are both load-bearing.

    ``eval_at`` and ``eval_flags`` exist because ``target_iteration`` is
    ``RUN_TO``, which an evaluate task does not use -- so every one of the 38
    evaluate tasks on the share recorded a target of ``0`` and the rung and board
    seed were written down NOWHERE. Three evaluations of one checkpoint at three
    seeds were indistinguishable in the record, and the eval documents they
    produced carry no task reference to join back on. Nothing can recover those;
    this is what stops the next set going the same way.

    ``code_snapshot`` is the same shape of omission, one level up: it names the
    exact bytes a task ran -- uncommitted changes and all -- and it travelled all
    the way to the node as a fetch instruction while being recorded nowhere. The
    commit alone does not identify a program here, because experiments live in
    parallel worktrees that share a hash and differ only in what is uncommitted;
    the branch narrows that and the snapshot closes it.
    """
    directory = tasks_dir(share)
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
        "eval_at": eval_at,
        "eval_flags": list(eval_flags),
        # What this task ACHIEVED and how wide it ran -- the two things an
        # estimate for the next one needs. `units` is work done BY THIS TASK,
        # not the counter's value: a task resuming at 140M and reaching 150M did
        # 10M, and recording 150M would claim fifteen times its real throughput.
        "workers": workers,
        "units": units,
        # A count is not a measurement without its unit. `evaluate` moved from
        # rungs to flop branches, and a rung-rate mixed into a branch-rate
        # would not fail -- it would quietly predict ~30x wrong. Recorded so a
        # reader can tell the lineages apart instead of averaging them.
        "units_unit": units_unit,
        "node_id": node_id,
        "code_snapshot": code_snapshot,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_branch": git_branch,
        "event": event,
        "ts": _utcnow(),
        "exit_code": exit_code,
        "cause": cause,
    }
    suffix = START_SUFFIX if event == EVENT_STARTED else EXIT_SUFFIX
    path = directory / f"{task_id}.{attempt}{suffix}"
    records.write_snapshot(path, record, records.REGISTRY[f"legs/*{suffix}"])
    return path


def write_progress_record(
    share: str | os.PathLike[str],
    *,
    task_id: str,
    progress: kinds.Progress,
) -> Path:
    """How far along a RUNNING task is. Overwritten, unlike start and exit.

    Current state, not history: only the latest matters, and keeping every
    sample would put one file per tick per task on a share where file COUNT is
    what makes every read slow.

    Torn writes are expected and tolerated rather than prevented -- SMB has no
    atomic rename, and this is refreshed every couple of minutes anyway, so a
    reader that cannot parse one sample simply shows the task with no bar.
    Never worth failing a task over: see the caller, which swallows everything.
    """
    directory = tasks_dir(share)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task_id,
        "attempt": _latest_attempt(directory, task_id),
        "progress": progress.as_record(),
        "ts": _utcnow(),
    }
    path = directory / f"{task_id}{PROGRESS_SUFFIX}"
    records.write_snapshot(path, record, records.REGISTRY[f"legs/*{PROGRESS_SUFFIX}"])
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


def observed_record(
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
) -> dict[str, Any]:
    """What Batch says happened, as a document, without writing it.

    Split from the write so :func:`reconcile` can ask whether the observation
    it just made says anything the share does not already hold.
    """
    return {
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
    only_if_new: bool = False,
) -> Path | None:
    """Record what Batch says happened, from the client.

    Its own filename, so the two sides never contend. Joins to the task's LATEST
    attempt -- Batch describes no other.

    ``only_if_new`` answers None when the stored record already says this, so a
    caller that publishes what changed has nothing to publish. See
    :data:`_VOLATILE_OBSERVED_FIELDS` for why "already says this" cannot be a
    byte comparison.
    """
    directory = tasks_dir(share)
    path = directory / f"{task_id}{OBSERVED_SUFFIX}"
    record = observed_record(
        task_id=task_id,
        job_id=job_id,
        state=state,
        result=result,
        exit_code=exit_code,
        failure=failure,
        start_time=start_time,
        end_time=end_time,
        node_id=node_id,
    )
    if only_if_new and _says_the_same(_load(path), record):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    records.write_snapshot(path, record, records.REGISTRY[f"legs/*{OBSERVED_SUFFIX}"])
    return path


"""When an observation is worth writing down
------------------------------------------
``observed_at`` is stamped on every read, so byte-comparing two observations of
the same finished task always differs -- which made every reconcile re-write and
re-upload records that said nothing new. Measured: six unchanged observations
re-uploaded on EVERY console poll, 14.1s of serial share writes for no
information. A task that is still running is legitimately re-observed; one that
finished last week is not.
"""
_VOLATILE_OBSERVED_FIELDS = frozenset({"observed_at", "schema_version"})


def _says_the_same(existing: dict[str, Any] | None, fresh: dict[str, Any]) -> bool:
    """Whether a stored observation already carries this one's information."""
    if existing is None:
        return False

    def substance(record: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in record.items() if k not in _VOLATILE_OBSERVED_FIELDS}

    return substance(existing) == substance(fresh)


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


def read_tasks(share: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Join the node and observer records into one row per attempt, newest last.

    ``cause`` prefers the node's own account when it reached a terminal event --
    it distinguishes a hang from a crash where Batch reports both as
    ``failure``. Otherwise the observer's view is all there is.
    """
    directory = tasks_dir(share)
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
    # Progress is per TASK, not per attempt: a retry starts over and overwrites
    # it, and a stale sample from a dead attempt would show a bar that cannot move.
    running: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"*{PROGRESS_SUFFIX}")):
        record = _load(path)
        if record and record.get("task_id"):
            running[record["task_id"]] = record

    latest = {task: max(a for t, a in attempts if t == task) for task, _ in attempts}

    # Known to Batch but never recorded by the node -- killed before its first
    # write. Still gets a row: a task that vanishes is indistinguishable from one
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
                "eval_at": node.get("eval_at", ""),
                "eval_flags": node.get("eval_flags", []),
                "workers": node.get("workers", 0),
                "units": (exit_record or {}).get("units", 0.0),
                "units_unit": (exit_record or {}).get("units_unit", ""),
                # WHAT CODE RAN. Both node records carry it, so `node` --
                # already exit-or-start -- answers for a task that died before
                # its exit record too, which is when the question is asked most.
                "code_snapshot": node.get("code_snapshot", ""),
                "git_commit": node.get("git_commit", ""),
                "git_dirty": node.get("git_dirty", ""),
                "git_branch": node.get("git_branch", ""),
                # One phrase saying what this task DID, derived here so the
                # terminal and the console cannot word it differently.
                "what": kinds.describe(node),
                "cause": cause,
                "cause_source": cause_source,
                # Only for the attempt Batch is describing, and only while the
                # task has not ended: a finished task showing "62%" is a sample
                # that stopped arriving, not a task stuck at 62%.
                "progress": (
                    (running.get(task_id) or {}).get("progress")
                    if attempt == latest.get(task_id) and cause not in TERMINAL_CAUSES
                    else None
                ),
                # Not dict.get's default: the node record always carries the
                # key and leaves it null, so a default would never be reached
                # and the observer's code -- the only one a killed task has --
                # would be dropped.
                "exit_code": _first_not_none(exit_record.get("exit_code"), batch.get("exit_code")),
                # From the node's own records: an observer record exists only
                # for an unresolved task, so reading times off it would blank
                # every cleanly-finished one.
                "started_at": _first_not_none(start.get("ts"), batch.get("start_time")),
                "ended_at": _first_not_none(exit_record.get("ts"), batch.get("end_time")),
                "failure": batch.get("failure"),
                "node_id": node.get("node_id") or batch.get("node_id", ""),
            }
        )
    # ended_at then started_at, so a still-running task sorts beside the one it
    # followed rather than at the front.
    joined.sort(key=lambda r: (r.get("ended_at") or r.get("started_at") or "", r["task_id"]))
    now = _utcnow()
    for row in joined:
        row["eta_seconds"] = kinds.remaining(row, joined, now)
    return joined


def unresolved_tasks(share: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The rows whose node record never reached a terminal event.

    Returned whole, not as ids, so a caller can ask Batch about exactly these
    ``(job_id, task_id)`` pairs. Enumerating every job in the account to find
    the one or two open questions cost ~0.39s per job -- the answer scaled with
    history rather than with what was actually unexplained.
    """
    return [row for row in read_tasks(share) if row["cause"] not in TERMINAL_CAUSES]


def unresolved_task_ids(share: str | os.PathLike[str]) -> list[str]:
    """Tasks whose node record never reached a terminal event -- exactly the
    ones worth asking Batch about."""
    return sorted({row["task_id"] for row in unresolved_tasks(share)})


def _first_not_none(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _load(path: Path) -> dict[str, Any] | None:
    """Skipped, never fatal: a half-written file is the expected residue of a
    task killed mid-write, and must not take down the listing that explains it."""
    return records.read_snapshot(path)


def format_table(rows: Iterable[dict[str, Any]]) -> str:
    """Compact fixed-width listing, one row per task."""
    # `what` rather than `op`: it IS the op for a task that recorded nothing more,
    # and the op plus what it was aimed at for one that did.
    # `code` is the branch, not the snapshot id: the id is exact but twenty
    # characters of timestamp, and this table is for scanning. The exact answer
    # is one `--limit 0` payload or one console click away, and both carry all
    # three. A column here at all because comparing two arms means first knowing
    # which arm each row IS, and every row used to look identical.
    columns = (
        "task_id",
        "attempt",
        "what",
        "run_id",
        "code",
        "cause",
        "done",
        "left",
        "ended_at",
    )
    # `done` is the running task's bar in a terminal: a phrase, since a
    # fixed-width table has nowhere to draw one, and it says what is being
    # counted rather than only how much of it.
    materialised = [{c: _cell(_derived(r, c)) for c in columns} for r in rows]
    if not materialised:
        return "  no task records on the share"
    widths = {c: max(len(c), *(len(r[c]) for r in materialised)) for c in columns}
    lines = ["  " + "  ".join(c.ljust(widths[c]) for c in columns)]
    lines.append("  " + "  ".join("-" * widths[c] for c in columns))
    lines += ["  " + "  ".join(r[c].ljust(widths[c]) for c in columns) for r in materialised]
    return "\n".join(lines)


def _derived(row: dict[str, Any], column: str) -> Any:
    if column == "code":
        return code_label(row)
    if column == "done":
        progress = kinds.Progress.from_record(row.get("progress"))
        return f"{progress.fraction:.0%} {progress.phrase}" if progress is not None else ""
    if column == "left":
        return _duration(row.get("eta_seconds"))
    return row.get(column)


def code_label(row: dict[str, Any]) -> str:
    """One short phrase for which code a task ran, for a column or a chip.

    The branch when there is one, because that is the name the work has while it
    is being done -- `worktree-hybrid-kernels` says what the arm IS, where
    `c13dcb7` says only which history it forked from and is shared by every
    worktree that has not committed yet. A short commit when the checkout was
    detached, and `+` when the tree was dirty on top of it.

    Empty for the tasks that pre-date this being recorded, which is most of the
    ones on the share: a blank column reads as "not known" where a plausible
    filler would read as an answer.
    """
    branch = row.get("git_branch") or ""
    commit = row.get("git_commit") or ""
    base = branch or commit[:7]
    if not base:
        return ""
    return f"{base}+" if row.get("git_dirty") == "1" else base


def _duration(seconds: Any) -> str:
    """`2h 14m`, `3m`, `40s`.

    Seconds below a minute rather than `~0m`, which reads as "no estimate"
    when it means "nearly done" -- the first probe finished in under a minute
    and reported exactly that.
    """
    if not isinstance(seconds, int | float):
        return ""
    if seconds < 60:
        return f"~{int(seconds)}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"~{minutes}m"
    return f"~{minutes // 60}h {minutes % 60}m"


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def reconcile(share: str | os.PathLike[str], tasks: Iterable[dict[str, Any]]) -> list[str]:
    """Write observer records for tasks the node never explained.

    ``tasks`` is `batch.list_jobs_with_tasks` output, flattened so each task
    carries its `job`. Only unresolved tasks: otherwise the cost scales with
    history rather than with open questions.

    Returns the ids whose observation is NEW -- which the caller then publishes,
    so an observation that repeats what the share already holds costs no write.
    A task Batch cannot explain any better than last time (one still running,
    most of all) stays unresolved forever, and re-uploading its unchanged record
    on every read is the difference between a poll costing nothing and costing
    14 seconds.
    """
    open_questions = set(unresolved_task_ids(share))
    explained = []
    for task in tasks:
        task_id = task.get("task")
        if not task_id or task_id not in open_questions:
            continue
        written = write_observed_record(
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
            only_if_new=True,
        )
        if written is not None:
            explained.append(task_id)
    return explained
