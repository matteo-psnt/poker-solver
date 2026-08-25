"""The `tasks` subcommand: what happened to every task, including the silent deaths.

``jobs`` reads Batch directly, so it shows only what Batch still retains -- and a
task's record ages out while the run it belonged to lives on. This reads the
durable copy in the record instead, then asks Batch about the tasks the record
cannot explain.

Neither side can answer alone. The node wrapper writes its own account on entry and
from its EXIT trap, covering every death the shell survives and distinguishing a
hang from an OOM from a cancellation -- Batch reports all three as ``failure``.
The deaths it cannot cover (OOM-kill, SIGKILL, node loss, task-level wall clock)
leave a task stuck at ``started``, because the trap never ran; only Batch can
explain those.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.adapters.postgres import connect, observations, queries
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command
from src.shared import task_history
from src.shared.cloudtask import kinds, task_log
from src.shared.task_history import TaskRow

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable

# Which version of each record a materialised tree holds: one `etag<TAB>name`
# line per file. BESIDE `legs/`, not inside it, where `read_documents` would
# read it as a leg -- and not JSON, because it is not an artifact: a tree's
# private note to its successor, gone with the tree.
_ETAGS_NAME = "legs.etags"


class TasksPayload(BaseModel):
    """What `tasks` answers: one row per attempt, newest last."""

    op: Literal["tasks"] = "tasks"
    rows: list[TaskRow] = []
    """How many attempts Batch was asked about. Null when nothing was."""
    reconciled: int | None = None
    """Trimmed by ``--limit``. The count is the fact, the list is the display."""
    hidden_rows: int = 0


class TasksSummary(BaseModel):
    """A `tasks` payload fetched only to be JOINED, with its rows removed.

    Its own type, because the alternative was one model describing two different
    things. `views._summarised` replaces `rows` with a count when a view wants
    the join and not the log, so `parts.tasks.payload` was sometimes four
    hundred rows and sometimes a stub -- and `Tasks` had `source_rows` optional
    to cover both, which meant a page reading `.rows` on a trimmed part got `[]`
    and was correct about nothing. It survived on the author remembering to read
    the join instead.

    There is deliberately no ``rows`` field. TypeScript cannot offer one, so the
    join is the only thing available, which is what it always should have been.

    ``source_rows`` rather than silence: a run page showing two tasks out of a
    log of four hundred has to say which number is which, because a join that
    quietly returns few rows out of many is indistinguishable from a broken one.
    """

    op: Literal["tasks"] = "tasks"
    source_rows: int
    reconciled: int | None = None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver tasks`."""
    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Read the record without asking Batch about unresolved tasks.",
    )
    parser.add_argument(
        "--tasks-dir",
        default=None,
        help="Read a local legs/ directory instead of the record. Implies --skip-reconcile.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show only the last N attempts (0 = all, the default). Unlike `jobs`, "
        "this does NOT truncate by default: the row worth finding here is a death, "
        "and hiding old rows by default would hide exactly the ones being looked for.",
    )


def _result(rows: list[TaskRow], reconciled: int | None, limit: int) -> TasksPayload:
    """One payload shape for both sources, newest last."""
    shown = rows[-limit:] if limit > 0 else rows
    return TasksPayload(rows=shown, reconciled=reconciled, hidden_rows=len(rows) - len(shown))


def run(args: argparse.Namespace) -> TasksPayload:
    """Join the node's account with Batch's, and report one row per attempt."""
    if args.tasks_dir:
        return _result(task_history.read_tasks(Path(args.tasks_dir)), None, args.limit)

    return _from_database(connect.engine_from_environment(), args)


def _from_database(engine: Any, args: argparse.Namespace) -> TasksPayload:
    """The same join, over rows instead of files.

    Materialising `legs/` to answer this costs 88.5s cold -- every CLI call and
    the console's first screen -- for 13,900 documents whose history is
    immutable. The join is `task_history.join_documents` either way; what
    changes is where the documents came from.

    Reconciliation still asks BATCH, which is the half no store can make
    cheaper. What the database makes cheap is knowing WHICH tasks to ask about
    and which answers are new, neither of which needs the tree any more.
    """
    rows = task_history.join_documents(task_log.documents_from_rows(queries.leg_rows(engine)))
    open_tasks = _still_open(rows)
    if args.skip_reconcile or not open_tasks:
        return _result(rows, None, args.limit)

    config = CloudConfig.load()
    fresh = _new_observations(
        _ask_batch(config, open_tasks), open_tasks, queries.observed_legs(engine)
    )
    if fresh:
        observations.record_observations(engine, fresh)
    return _result(rows, len(fresh), args.limit)


def _still_open(rows: list[TaskRow]) -> list[TaskRow]:
    """The rows Batch could still explain: non-terminal AND the latest attempt: Batch describes only a task's current
    attempt, so an earlier one is unresolved by construction and asking about it
    returns the answer for a different attempt. 1,326 non-terminal rows, of
    which 1,290 were superseded.
    """
    latest: dict[str, int] = {}
    for row in rows:
        latest[row.task_id] = max(latest.get(row.task_id, 0), row.attempt)
    return [
        row
        for row in rows
        if row.cause not in task_history.TERMINAL_CAUSES and row.attempt == latest[row.task_id]
    ]


def _new_observations(
    seen: list[dict[str, Any]],
    open_tasks: list[TaskRow],
    stored: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Batch's answers that say something the record does not already say.

    `says_the_same` rather than a byte comparison, and the reason is measured:
    `observed_at` is stamped on every read, so two observations of one finished
    task differ in a field that means nothing. Re-publishing those cost 14.1s
    per poll restating what the record already said.
    """
    open_ids = {row.task_id for row in open_tasks}
    fresh: dict[str, dict[str, Any]] = {}
    for task in seen:
        task_id = task.get("task")
        if not task_id or task_id not in open_ids:
            continue
        document = task_history.observed_record(
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
        if not task_history.says_the_same(stored.get(task_id), document):
            fresh[task_id] = document
    return fresh


def _ask_batch(config: CloudConfig, open_tasks: list[TaskRow]) -> list[dict[str, Any]]:
    """Ask Batch about exactly the tasks the record could not explain.

    One ``get_task`` per open question, concurrently, rather than listing every
    task of every job in the account -- which cost ~0.39s per job and scaled
    with history rather than with what was unexplained. A task record carries
    its own ``job_id``, so the pair is already known.

    A row with no ``job_id`` cannot be addressed this way and falls back to the
    old enumeration. That is not dead code: it covers records written before
    the field existed, and losing the explanation would be worse than the cost.

    Returns DICTS, and that is the one boundary in this file: `observed_record`
    reads these with ``.get()`` and lives in :mod:`src.shared.task_history`,
    which is layer-neutral and cannot import a shape from `interfaces`.
    `batch._task_record` classifies the state once, at the source.
    """
    client = batch.client(config)
    pairs = {(row.job_id, row.task_id) for row in open_tasks if row.job_id}
    if len(pairs) < len({row.task_id for row in open_tasks}):
        listed = batch.attach_tasks(client, batch.list_jobs(client))
        return [task.model_dump() for job in listed for task in job.tasks]

    with ThreadPoolExecutor(max_workers=min(16, len(pairs) or 1)) as pool:
        fetched = pool.map(lambda pair: batch.task_record(client, *pair), sorted(pairs))
    return [task.model_dump() for task in fetched if task]


def _etags(tree: Path | None) -> dict[str, str]:
    """The versions a tree was built from. Empty for a tree with no manifest."""
    if tree is None or not (tree / _ETAGS_NAME).is_file():
        return {}
    found: dict[str, str] = {}
    for line in (tree / _ETAGS_NAME).read_text().splitlines():
        etag, sep, name = line.partition("\t")
        if sep and etag:
            found[name] = etag
    return found


def _link(source: Path, destination: Path) -> None:
    """A hard link where the filesystem allows one, a copy where it does not."""
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, destination)


def format_table(rows: Iterable[TaskRow]) -> str:
    """Compact fixed-width listing, one row per task.

    The terminal's renderer, and it lives with the command that prints it: for
    any other surface the payload IS the interface. It spent a while in
    `task_log` instead, which put `ljust` column arithmetic inside the module
    the node imports before `uv sync`.
    """
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
        return "  no task records"
    widths = {c: max(len(c), *(len(r[c]) for r in materialised)) for c in columns}
    lines: list[str] = ["  " + "  ".join(c.ljust(widths[c]) for c in columns)]
    lines.append("  " + "  ".join("-" * widths[c] for c in columns))
    lines += ["  " + "  ".join(r[c].ljust(widths[c]) for c in columns) for r in materialised]
    return "\n".join(lines)


def _derived(row: TaskRow, column: str) -> Any:
    if column == "code":
        return code_label(row)
    if column == "done":
        # Dumped because `kinds` reads a Mapping: it lives under `cloudtask/`,
        # which is held to the node's stdlib-only rule.
        progress = kinds.Progress.from_record(row.progress.model_dump() if row.progress else None)
        return f"{progress.fraction:.0%} {progress.phrase}" if progress is not None else ""
    if column == "left":
        return _duration(row.eta_seconds)
    return getattr(row, column, None)


def code_label(row: TaskRow) -> str:
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
    base = row.git_branch or row.git_commit[:7]
    if not base:
        return ""
    return f"{base}+" if row.git_dirty == "1" else base


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


def render(payload: TasksPayload) -> None:
    if payload.reconciled:
        print(f"Asked Batch about {payload.reconciled} task(s) the share could not explain.")
    print(format_table(payload.rows))
    if payload.hidden_rows:
        print(f"  {payload.hidden_rows} earlier attempt(s) hidden — show with --limit 0")


COMMAND = Command(
    name="tasks",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Per-task outcomes from the share, reconciled against Batch.",
)
