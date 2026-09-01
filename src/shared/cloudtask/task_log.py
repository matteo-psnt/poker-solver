"""Writing a task's own account of itself, from the node.

``.run.json`` records what a *living* process did. It cannot record how an
attempt died: a container killed by OOM, ``maxWallClockTime`` or node loss is
gone before it can write. Batch sees those deaths but retains them for far less
time than the run lives. This is the node's half of the join;
:mod:`src.shared.task_history` is the reading half, and holds the observer's.

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

SNAPSHOTs in :mod:`src.shared.records` terms, share-scoped, written directly
rather than through a temporary file: the per-file layout carries the safety
that an atomic rename would.

The WRITER only, and stdlib-only with it -- see `.claude/rules/cloud.md`. The
reading half lives outside this package so the fail-closed import guard does not
hold 260 lines of laptop-only code to the node's stdlib-only rule.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared import records

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from src.shared.cloudtask import kinds

RECORDS_DIRNAME = "legs"

EVENT_STARTED = "started"
EVENT_FINISHED = "finished"

START_SUFFIX = ".start.json"
EXIT_SUFFIX = ".exit.json"
PROGRESS_SUFFIX = ".progress.json"

# Finer than Batch's success/failure, because a terminal cause suppresses
# reconciliation and a WRONG one loses the observer half of the join for good:
#   timeout    the RUN_TIMEOUT guard expired (124) -- a hang, not a crash
#   killed     SIGKILL from outside (137). `timeout` returns 124 even when its
#              own --kill-after fires, so 137 is never it
#   cancelled  the wrapper took SIGTERM -- `cancel`, or maxWallClockTime
#   partial    an evaluate task scored some rungs and failed others
# Which of these are FINAL is a reader's question -- see `task_history`.
CAUSE_COMPLETED = "completed"
CAUSE_FAILED = "failed"
CAUSE_TIMEOUT = "timeout"
CAUSE_KILLED = "killed"
CAUSE_CANCELLED = "cancelled"
CAUSE_PARTIAL = "partial"

# A leg document is normally its own file -- one writer per file, which is what
# makes writing to a share with no atomic rename safe. A reader pays a round trip
# per file for that, so sealed documents may also live in a bundle and
# :func:`read_documents` returns both alike. A loose file WINS over a bundled
# copy of the same name. Defined here because :func:`_next_attempt` needs it.
BUNDLE_SUFFIX = ".bundle.json"

TASK_ID_ENV = "AZ_BATCH_TASK_ID"


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def tasks_dir(share: str | os.PathLike[str]) -> Path:
    return Path(share) / RECORDS_DIRNAME


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


def read_documents(directory: Path) -> dict[str, dict[str, Any]]:
    """Every leg document in ``directory``, by filename, bundles included.

    The one reading primitive the WRITER also needs, for :func:`_next_attempt`.
    Everything built on top of it -- the join, compaction, reconciliation --
    is in :mod:`src.shared.task_history`.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"*{BUNDLE_SUFFIX}")):
        bundle = _load(path)
        found.update(
            {
                name: document
                for name, document in (bundle or {}).get("records", {}).items()
                if isinstance(document, dict)
            }
        )
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(BUNDLE_SUFFIX):
            continue
        document = _load(path)
        if document:
            found[path.name] = document
    return found


# `progress` and `observed` are written per TASK, not per attempt, so their
# filenames carry no attempt number. They are not attempt-scoped facts: the join
# attaches them to the LATEST attempt. Carried under this sentinel so the
# record's primary key still holds and the difference stays visible, rather than
# being flattened onto attempt 0 where it would collide with a real first one.
TASK_SCOPED = -1

# When a leg happened, from whichever field its writer used. Not one field,
# because the writers are different programs: the node stamps `ts`, while the
# READER'S observation stamps `observed_at` -- when IT looked, not when
# something happened. Batch's own times come last so a record with neither is
# not dropped for want of a clock.
_INSTANT_FIELDS = ("ts", "observed_at", "end_time", "start_time")


def leg_row(task_id: str, attempt: int, leg: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """One leg document as the flat row BOTH stores hold.

    Here, in the stdlib-only half, because there are three writers and one of
    them is the node -- which runs on an interpreter that has no pydantic and no
    ORM. A row built two ways is two answers about the same record, which is
    what `backfill-record --verify` would then report as a divergence forever.
    """
    return {
        "task_id": task_id,
        "attempt": attempt,
        "leg": leg,
        "run_id": document.get("run_id") or None,
        "at": next((document[f] for f in _INSTANT_FIELDS if document.get(f)), None),
        "body": dict(document),
    }


def rows_from_documents(
    documents: dict[str, dict[str, Any]],
) -> list[tuple[str, int, str, dict[str, Any]]]:
    """Filename-keyed documents as `(task_id, attempt, leg, body)`.

    TWO NAME SHAPES, and requiring the first silently dropped 4,591 of 13,440
    documents -- a third of the record, including every one of the 1,823
    `observed` legs, which are the only account of a death the node did not
    survive:

        <task>.<attempt>.start.json      per ATTEMPT -- a retry reuses the id
        <task>.<attempt>.exit.json
        <task>.progress.json             per TASK
        <task>.observed.json             per TASK, written by the READER
    """
    rows: list[tuple[str, int, str, dict[str, Any]]] = []
    seen: set[tuple[str, int, str]] = set()
    for name, document in documents.items():
        stem = name.removesuffix(".json")
        parts = stem.rsplit(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            task_id, attempt, leg = parts[0], int(parts[1]), parts[2]
        elif len(parts) >= 2:
            task_id, attempt, leg = stem.rsplit(".", 1)[0], TASK_SCOPED, parts[-1]
        else:
            continue
        key = (task_id, attempt, leg)
        if key in seen:
            continue
        seen.add(key)
        rows.append((task_id, attempt, leg, document))
    return rows


def documents_from_rows(
    rows: Iterable[tuple[str, int, str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """`(task_id, attempt, leg, body)` rows as the filename-keyed dict a reader
    joins over -- the inverse of :func:`rows_from_documents`, and beside it so
    the two cannot drift.

    The names are REBUILT rather than stored, from the same suffixes the node
    writes with, so both stores present the join with the same keys and the
    filename ordering that decides ties still decides them the same way.
    """
    named: dict[str, dict[str, Any]] = {}
    for task_id, attempt, leg, body in rows:
        stem = task_id if attempt == TASK_SCOPED else f"{task_id}.{attempt}"
        named[f"{stem}.{leg}.json"] = body
    return named


def read_task_documents(directory: Path, task_id: str) -> dict[str, dict[str, Any]]:
    """One task's leg documents, by filename.

    Scoped by NAME rather than filtering what :func:`read_documents` returns,
    and the difference is the whole point: that reads and parses every file in
    the directory, which is 13,600 of them on the share. A node mirroring its
    own two records paid ~100s for that, on every call.

    Bundles are not consulted, which is correct rather than a shortcut: a bundle
    holds records `compact-legs` has SEALED, and a task asking about itself is
    by definition not sealed yet.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"{task_id}.*.json")):
        if path.name.endswith(BUNDLE_SUFFIX):
            continue
        document = _load(path)
        if document:
            found[path.name] = document
    return found


def _load(path: Path) -> dict[str, Any] | None:
    """Skipped, never fatal: a half-written file is the expected residue of a
    task killed mid-write, and must not take down the listing that explains it."""
    return records.read_snapshot(path)


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
        "ts": utcnow(),
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
        "attempt": _this_attempt(directory, task_id),
        "progress": progress.as_record(),
        "ts": utcnow(),
    }
    path = directory / f"{task_id}{PROGRESS_SUFFIX}"
    records.write_snapshot(path, record, records.REGISTRY[f"legs/*{PROGRESS_SUFFIX}"])
    return path


# The attempt number cannot change inside one process -- a Batch retry is a NEW
# process -- and deriving it walks every document in `legs/`. Uncached, the
# 15-second progress write re-globbed and re-parsed the whole share directory
# (thousands of files, each a round trip) for a number that was already known.
_ATTEMPT_MEMO: dict[tuple[str, str], int] = {}


def _this_attempt(directory: Path, task_id: str) -> int:
    """:func:`_latest_attempt`, resolved once per process."""
    key = (str(directory), task_id)
    if key not in _ATTEMPT_MEMO:
        _ATTEMPT_MEMO[key] = _latest_attempt(directory, task_id)
    return _ATTEMPT_MEMO[key]


def _next_attempt(directory: Path, task_id: str) -> int:
    """1 for a first run, 2 for Batch's first retry, and so on.

    Counted across bundles as well as loose files, and that is load-bearing
    rather than tidy: this number NAMES the file the next attempt writes. If
    compaction swept an earlier attempt's ``.start.json`` into a bundle and this
    counted only what is loose, a retry would compute an attempt number that has
    already been used and overwrite the record of the failure that caused it --
    silently, and on the durable copy.
    """
    prefix = f"{task_id}."
    starts = [
        name
        for name in read_documents(directory)
        if name.startswith(prefix) and name.endswith(START_SUFFIX)
    ]
    return len(starts) + 1


def _latest_attempt(directory: Path, task_id: str) -> int:
    """The attempt the terminal record belongs to.

    Derived from disk rather than passed through the shell: the exit trap may
    have lost anything the entry point computed.
    """
    return max(_next_attempt(directory, task_id) - 1, 1)
