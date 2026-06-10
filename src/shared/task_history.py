"""Reading the per-task record: the join, and what may be compacted away.

:mod:`src.shared.cloudtask.task_log` WRITES a task's account, on the node,
before ``uv sync``, on the pinned image's ``python3``. This READS it, and runs
nowhere near a node -- a laptop, the console, `report`. The two halves shared a
file for a while and the cost was not length: it put a stdlib-only constraint on
260 lines that never execute under it, and it made "what does the node load"
unanswerable from the directory.

The split is exact. The node calls three names from the writer and none from
here; every function in this module is reached from `interfaces` or from
`pipeline.services`. It sits at ``shared/`` rather than under ``cloudtask/`` for
that reason -- inside, it would still be walked by the node's fail-closed import
guard, which is the constraint being removed.

It cannot live under ``interfaces`` either, which is where its callers mostly
are: ``pipeline.services.experiments`` joins task rows into a run digest, and
``pipeline -> interfaces`` is forbidden. Layer-neutral is what it actually is.

THE JOIN. The node alone can tell a hang from a crash from a cancellation --
Batch reports both as ``failure``. Batch alone can explain a death the node did
not survive, because the container is gone before it can write anything. Neither
view is sufficient and this is where they meet: one row per task-attempt, the
node's cause preferred when it reached a terminal event, the observer's used
when it did not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from src.shared import records, task_states
from src.shared.cloudtask import kinds
from src.shared.cloudtask.task_log import (
    BUNDLE_SUFFIX,
    CAUSE_CANCELLED,
    CAUSE_COMPLETED,
    CAUSE_FAILED,
    CAUSE_KILLED,
    CAUSE_PARTIAL,
    CAUSE_TIMEOUT,
    EXIT_SUFFIX,
    PROGRESS_SUFFIX,
    START_SUFFIX,
    read_documents,
    tasks_dir,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable
    from pathlib import Path

# Written by the READER, not the node. Its own filename, so the two sides never
# contend on a share with no atomic append; joins to the task's LATEST attempt.
OBSERVED_SUFFIX = ".observed.json"

# Batch ``executionInfo.result`` / task state -> coarse exit cause. The dead
# process cannot report these; Batch can. FAILURE conflates an in-container
# error with an OOM-kill -- the node is gone either way, and only the published
# task log can tell them apart.
_OBSERVED_CAUSE_BY_RESULT: dict[str, str] = {
    "success": "completed",
    "failure": "failed",
}

# A terminal cause SUPPRESSES reconciliation -- the reader stops asking Batch
# about a task it believes explained -- so a wrong one loses the observer half of
# the join forever. That is why 124 (a hang) and 137 (the OOM killer) are kept
# apart. The vocabulary itself is the writer's, in `task_log`.
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

# Not-terminal is NOT the same as live: ``unresolved`` covers both a running task
# and one that died without stamping an end. Running an open interval to ``now``
# on not-terminal credited four abandoned attempts with 455 of 718 node-hours.
# DERIVED from :data:`~src.shared.task_states.OCCUPIES_A_NODE`, which is where
# the decision is made -- this is the same decision in the SHARE's vocabulary.
CAUSE_RUNNING = task_states.cause_of(task_states.Phase.RUNNING)
CAUSE_PREPARING = task_states.cause_of(task_states.Phase.STARTING)

LIVE_CAUSES = task_states.LIVE_CAUSES


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _first_not_none(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


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

    ``state`` is stored SHORT -- ``completed``, not ``BatchTaskState.COMPLETED``.
    Every observed record on the share already holds the short word, and the
    caller no longer shortens it: `tasks._translate` used to, and was deleted
    when `batch.BatchTask` began carrying a classified `phase` beside the raw
    state. Normalising here rather than at the caller is what keeps the record
    format the reader's business, and stops Azure's spelling from being the
    thing that lands in a durable record.
    """
    return {
        "source": "batch",
        "task_id": task_id,
        "job_id": job_id,
        "state": (state or "").rsplit(".", 1)[-1].lower(),
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
    if only_if_new and _says_the_same(records.read_snapshot(path), record):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    records.write_snapshot(path, record, records.REGISTRY[f"legs/*{OBSERVED_SUFFIX}"])
    return path


# ``observed_at`` is stamped on every read, so two observations of one finished
# task always differ and every reconcile re-uploaded records saying nothing new:
# 14.1s of serial share writes per console poll.
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

    A task that has not FINISHED reports its own phase rather than guessing:
    ``preparing`` is more useful than a wrong ``failed``. The spelling comes from
    :func:`~src.shared.task_states.cause_of`, because it is what lands in a
    record on the share and `TERMINAL_CAUSES` is read against those words.
    """
    state = (record.get("state") or "").rsplit(".", 1)[-1].lower()
    phase = task_states.phase_of(state)
    if phase is task_states.Phase.UNKNOWN:
        # The raw word, not "unknown": a state this does not recognise is the one
        # worth reading verbatim, and flattening it would hide that Batch has
        # started saying something new.
        return state or "unknown"
    if phase is not task_states.Phase.FINISHED:
        return task_states.cause_of(phase)
    result = (record.get("result") or "").lower()
    if result in _OBSERVED_CAUSE_BY_RESULT:
        return _OBSERVED_CAUSE_BY_RESULT[result]
    return "unknown"


def compactable(directory: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Which loose documents may be bundled, and which files they came from.

    ONLY attempts whose node record reached a terminal cause. Two things make
    that the boundary rather than age:

    * an attempt with no terminal record is one `tasks` still asks Batch about,
      and `reconcile` writes its answer to ``<task>.observed.json`` -- a
      filename, not a bundle entry. Bundling the unsealed half would strand that.
    * a task that is still RUNNING has a `.progress.json` being overwritten
      under it.

    A task's documents move together or not at all, so a bundled attempt is
    never split across the two representations. Returns the documents to bundle
    and the loose filenames they replace -- the caller decides whether to delete
    them, because that is the irreversible half.
    """
    documents = read_documents(directory)
    loose = {
        path.name for path in directory.glob("*.json") if not path.name.endswith(BUNDLE_SUFFIX)
    }

    sealed: set[str] = set()
    unsealed: set[str] = set()
    for name, document in documents.items():
        task_id = document.get("task_id")
        if not task_id:
            continue
        if name.endswith(EXIT_SUFFIX) and document.get("cause") in TERMINAL_CAUSES:
            sealed.add(task_id)
        elif name.endswith((START_SUFFIX, PROGRESS_SUFFIX)):
            unsealed.add(task_id)
    # A task is sealed only if EVERY attempt of it is: a retry in flight shares
    # the id with the failed attempt before it.
    for name, document in documents.items():
        if not name.endswith(START_SUFFIX):
            continue
        task_id, attempt = document.get("task_id"), int(document.get("attempt", 1))
        exit_name = f"{task_id}.{attempt}{EXIT_SUFFIX}"
        exit_record = documents.get(exit_name, {})
        if exit_record.get("cause") not in TERMINAL_CAUSES:
            sealed.discard(str(task_id))
    unsealed -= sealed

    movable = {
        name: document
        for name, document in documents.items()
        if name in loose and document.get("task_id") in sealed
    }
    return movable, sorted(movable)


def bundle_document(
    records_by_name: dict[str, dict[str, Any]],
    *,
    previous: dict[str, Any] | None = None,
    compaction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The bundle payload: the documents VERBATIM, keyed by their filename.

    No reshaping at all. The join in :func:`read_tasks` is subtle -- attempt
    numbering, which record explains a death, whose view wins -- and a bundle
    storing a digested form would be a second implementation of it that could drift.

    ``previous`` is the bundle already written at this name, and it is not optional
    in practice: :func:`compactable` offers only documents that are still LOOSE, so
    a bundle built from its answer alone would drop everything an earlier
    compaction absorbed.

    ``compaction`` entries ACCUMULATE, so a bundle carries its own history. It sits
    beside ``records`` rather than inside it because
    :func:`~src.shared.cloudtask.task_log.read_documents` reads only that key: the
    provenance must never reach the join.
    """
    carried = (previous or {}).get("records", {})
    history = list((previous or {}).get("compactions", []))
    if compaction is not None:
        history.append(compaction)
    return {
        "records": dict(sorted({**carried, **records_by_name}.items())),
        "compactions": history,
    }


def _with_suffix(documents: dict[str, dict[str, Any]], suffix: str) -> list[dict[str, Any]]:
    """The documents whose filename ends in ``suffix``, in filename order.

    Order is preserved because it decided ties: two records claiming the same
    slot resolved to whichever sorted last, and a bundle must not quietly
    reshuffle that.
    """
    return [document for name, document in sorted(documents.items()) if name.endswith(suffix)]


def _by_task_id(documents: dict[str, dict[str, Any]], suffix: str) -> dict[str, dict[str, Any]]:
    """Documents of one suffix keyed by task id, latest spelling winning.

    Used for the two that are per-TASK rather than per-attempt. Batch describes
    only a task's latest attempt, so its record joins there and nowhere else;
    progress is overwritten by a retry, so a sample from a dead attempt would
    show a bar that cannot move.
    """
    return {
        document["task_id"]: document
        for document in _with_suffix(documents, suffix)
        if document.get("task_id")
    }


def _cause(
    start: dict[str, Any], exit_record: dict[str, Any], batch: dict[str, Any]
) -> tuple[str, str]:
    """The cause of this attempt, and whose account it came from.

    Precedence, and it is the whole point of the join: the node's own terminal
    account wins, because it can tell a hang from a crash from a cancellation
    where Batch reports every one of them as ``failure``. Batch answers when the
    node never got to.
    """
    node_cause = exit_record.get("cause")
    if node_cause in TERMINAL_CAUSES:
        return node_cause, "node"
    if batch:
        return observed_cause(batch), "batch"
    if start or exit_record:
        # Started and never finished, with nothing observed yet. Not the same
        # as "running": the record cannot tell them apart, and saying so is
        # better than picking one.
        return "unresolved", "node"
    return "unknown", "none"


class TaskProgress(BaseModel):
    """How far a RUNNING task has got. Absent once it ends -- a finished task
    showing "62%" is a sample that stopped arriving, not a task stuck at 62%.

    TOLERANT, and matched deliberately to `kinds.Progress.from_record`, which
    reads the same bytes and accepts a missing `unit` and float counts. These
    are untyped JSON off an SMB share, written by a wrapper that may be an older
    version or may have been killed halfway through the write -- and this module
    reads `.get` with a default everywhere else for exactly that reason. A
    required `unit` here would turn one malformed document into a failure of
    `tasks`, `cost`, `runinfo` and `reconcile` at once.
    """

    done: float
    total: float
    unit: str = ""
    """The window the task measured its own rate over, mirrored from
    `kinds.Progress` -- which is what reads them back, to say how long is left.
    Zero on a wrapper that predates them, which costs an estimate, not a bar."""
    base: float = 0.0
    window_seconds: float = 0.0

    @field_validator("unit", mode="before")
    @classmethod
    def _blank_when_absent(cls, value: object) -> object:
        """A null `unit` is a blank one, exactly as `from_record` reads it.

        The wrapper writes the key with no value on a kind that cannot name its
        unit, so `None` is a shape that occurs rather than a malformed record.
        """
        return "" if value is None else value


class TaskRow(BaseModel):
    """One task-attempt, as the flat record every surface reads.

    THE declaration of this shape, and the only one. It used to exist four
    times -- as `_row`'s dict here, as `contract.TaskRow`, as a hand-written
    fixture, and again in TypeScript -- with nothing checking any pair of them
    against another.

    The node's WRITER stays dicts, deliberately: it runs under
    :mod:`src.shared.cloudtask` before ``uv sync``, where pydantic does not
    exist yet. This is the reading half, and constructing it from `_row`'s
    output is what binds the two.
    """

    task_id: str
    attempt: int
    job_id: str = ""
    run_id: str = ""
    op: str = ""
    config: str = ""
    target_iteration: int | str = ""
    eval_at: str = ""
    eval_flags: list[str] = []
    workers: int = 0
    """What a FINISHED task achieved, and how wide it ran -- the two things an
    estimate for the next one is built from."""
    units: float = 0.0
    units_unit: str = ""
    """WHICH CODE RAN. Three answers that do not replace one another: the commit
    names a history, the branch names the line of work, the snapshot names the
    exact bytes. Empty on every task recorded before this existed."""
    code_snapshot: str = ""
    git_commit: str = ""
    git_dirty: str = ""
    git_branch: str = ""
    """One phrase saying what this task DID, derived once so the terminal and
    the console cannot word it differently."""
    what: str = ""
    cause: str
    """Whose account the cause came from: the node's, or Batch's."""
    cause_source: str
    progress: TaskProgress | None = None
    exit_code: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    failure: dict[str, Any] | None = None
    node_id: str = ""
    """Seconds left, derived here so both surfaces agree. Set after the join,
    because an estimate reads the OTHER attempts to know how fast this kind of
    task goes."""
    eta_seconds: float | None = None


def _row(
    task_id: str,
    attempt: int,
    sources: dict[str, dict[str, Any]],
    batch: dict[str, Any],
    progress: dict[str, Any] | None,
) -> dict[str, Any]:
    """One task-attempt as a plain dict, for the join to finish.

    Still a dict here rather than a :class:`TaskRow`, and that is not an
    oversight: `kinds.remaining` reads these rows and lives under
    ``cloudtask/``, which is held to the node's stdlib-only rule and cannot be
    handed a model. :func:`read_tasks` constructs the model once the
    estimate is in, and that construction is what checks these keys.
    """
    start, exit_record = sources.get("start", {}), sources.get("exit", {})
    cause, cause_source = _cause(start, exit_record, batch)
    # Exit-or-start, so a task that died before writing an exit still answers
    # for everything fixed at launch -- which is when the question is asked most.
    node = exit_record or start
    return {
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
        "units": exit_record.get("units", 0.0),
        "units_unit": exit_record.get("units_unit", ""),
        # WHAT CODE RAN. Both node records carry it, so `node` answers for a
        # task that died before its exit record too.
        "code_snapshot": node.get("code_snapshot", ""),
        "git_commit": node.get("git_commit", ""),
        "git_dirty": node.get("git_dirty", ""),
        "git_branch": node.get("git_branch", ""),
        # One phrase saying what this task DID, derived here so the terminal and
        # the console cannot word it differently.
        "what": kinds.describe(node),
        "cause": cause,
        "cause_source": cause_source,
        # Only while the task has not ended: a finished task showing "62%" is a
        # sample that stopped arriving, not a task stuck at 62%.
        "progress": (progress or {}).get("progress") if cause not in TERMINAL_CAUSES else None,
        # Not dict.get's default: the node record always carries the key and
        # leaves it null, so a default would never be reached and the observer's
        # code -- the only one a killed task has -- would be dropped.
        "exit_code": _first_not_none(exit_record.get("exit_code"), batch.get("exit_code")),
        # From the node's own records: an observer record exists only for an
        # unresolved task, so reading times off it would blank every
        # cleanly-finished one.
        "started_at": _first_not_none(start.get("ts"), batch.get("start_time")),
        "ended_at": _first_not_none(exit_record.get("ts"), batch.get("end_time")),
        "failure": batch.get("failure"),
        "node_id": node.get("node_id") or batch.get("node_id", ""),
    }


def read_tasks(share: str | os.PathLike[str]) -> list[TaskRow]:
    """Join the node and observer records into one row per attempt, newest last.

    The rows are assembled as dicts and become :class:`TaskRow` at the end. That
    ordering is forced: `kinds.remaining` needs every row to estimate any one of
    them, and it runs under the node's stdlib-only rule where a model cannot go. The
    construction is the check -- pydantic rejects a key `_row` invented and a
    field it stopped emitting, which is the binding this shape never had.
    """
    directory = tasks_dir(share)
    if not directory.is_dir():
        return []

    documents = read_documents(directory)

    # Keyed by (task_id, attempt): a Batch retry reuses the task id, and the
    # failed attempt is the one worth keeping.
    attempts: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for suffix, slot in ((START_SUFFIX, "start"), (EXIT_SUFFIX, "exit")):
        for record in _with_suffix(documents, suffix):
            if record.get("task_id"):
                key = (record["task_id"], int(record.get("attempt", 1)))
                attempts.setdefault(key, {})[slot] = record

    observed = _by_task_id(documents, OBSERVED_SUFFIX)
    running = _by_task_id(documents, PROGRESS_SUFFIX)

    latest = {task: max(a for t, a in attempts if t == task) for task, _ in attempts}

    # Known to Batch but never recorded by the node -- killed before its first
    # write. Still gets a row: a task that vanishes is indistinguishable from one
    # that never ran.
    for task_id in observed:
        if task_id not in latest:
            attempts[(task_id, 1)] = {}
            latest[task_id] = 1

    joined = [
        _row(
            task_id,
            attempt,
            sources,
            observed.get(task_id, {}) if latest.get(task_id) == attempt else {},
            running.get(task_id) if latest.get(task_id) == attempt else None,
        )
        for (task_id, attempt), sources in attempts.items()
    ]
    # ended_at then started_at, so a still-running task sorts beside the one it
    # followed rather than at the front.
    joined.sort(key=lambda r: (r.get("ended_at") or r.get("started_at") or "", r["task_id"]))
    now = _utcnow()
    for row in joined:
        row["eta_seconds"] = kinds.remaining(row, joined, now)
    return [TaskRow(**row) for row in joined]


def unresolved_tasks(share: str | os.PathLike[str]) -> list[TaskRow]:
    """The rows whose node record never reached a terminal event.

    Returned whole, not as ids, so a caller can ask Batch about exactly these
    ``(job_id, task_id)`` pairs. Enumerating every job in the account to find
    the one or two open questions cost ~0.39s per job -- the answer scaled with
    history rather than with what was actually unexplained.
    """
    return [row for row in read_tasks(share) if row.cause not in TERMINAL_CAUSES]


def unresolved_task_ids(share: str | os.PathLike[str]) -> list[str]:
    """Tasks whose node record never reached a terminal event -- exactly the
    ones worth asking Batch about."""
    return sorted({row.task_id for row in unresolved_tasks(share)})


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
