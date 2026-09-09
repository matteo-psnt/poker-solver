"""One screen, one request -- composed from commands and nothing else.

    **This layer may COMPOSE command payloads. It may not COMPUTE one.**

A view fans out over :meth:`Command.invoke` and joins what comes back. A join
may filter, group and cross-reference; it may not derive a quantity no command
can answer. `tests/interfaces/web/test_no_second_read_path.py` is what says so.

Every view returns its raw ``parts`` alongside its joins. The joins are a
convenience, never a replacement: a part that failed still has to reach the UI
as a reason, so one panel greys out and the rest of the screen survives.
"""

from __future__ import annotations

from typing import Any

from src.interfaces.commands import (
    curve,
    jobs,
    ledger,
    pool_status,
    progress,
    runinfo,
    tasks,
)
from src.interfaces.commands import runs as runs_command
from src.interfaces.commands._compose import Invoke, Part, compose, payloads

# A glanceable screen cannot carry two hundred rows, and the cost of fetching
# them is the point: `tasks` is the slowest read in the console.
LIVE_LIMIT = 10

# How far back the live screen reads. `tasks` turns this into a bound on the
# QUERY, so it is the difference between 16,895 legs and ~350: 2.51s against
# 0.12s, on the one view the status bar polls from every page every 5 seconds.
#
# Generously above what the screen draws. It has to cover every task holding a
# node -- the pool ceiling is 36 -- plus `LIVE_LIMIT` deaths, and a task is
# written about every 60s while it runs, so a live one is always among the most
# recently touched. 200 is two orders of margin on the first and still cheap.
LIVE_WINDOW = 200

# Deliberately NOT `LIVE_LIMIT`. The run list uses jobs to check whether a run
# claiming to be running has a task executing, and a run outlives the daily job
# its tasks land in -- so a smaller limit reads a live run as abandoned.
RUN_LIST_JOB_LIMIT = 50


def now(*, invoke: Invoke | None = None) -> dict[str, Any]:
    """What is happening right now, and did anything die.

    Three questions. `pool-status` carries the nodes and the pool's own last
    autoscale decision, so the screen no longer pays a separate evaluation
    whose formula could go stale; `cost` is gone because nothing on the page
    drew it, and it re-derived the task log to not be drawn.

    No join. Everything here is a panel in its own right, and the
    cross-references the client draws -- a task onto its node, a progress bar
    onto a task -- are presentation, not data.
    """
    composed = compose(
        "view-now",
        [
            Part("pool", pool_status.COMMAND),
            Part("jobs", jobs.COMMAND, {"limit": LIVE_LIMIT}),
            Part("tasks", tasks.COMMAND, {"limit": LIVE_WINDOW}),
        ],
        invoke=invoke,
    )
    composed["parts"]["tasks"] = _live_and_recent(composed["parts"]["tasks"])
    return composed


def _live_and_recent(part: dict[str, Any]) -> dict[str, Any]:
    """A copy of the tasks part holding every row holding a node, plus the last
    `LIVE_LIMIT`.

    `--limit 10` was the wrong cut: with twenty-one tasks running, eleven of
    them had no row to draw a progress bar from. A running task is live by
    definition and there are never many; the recent ten are the deaths.

    Live is the ROW's own `phase`, which is `OCCUPIES_A_NODE` in the share's
    vocabulary. It was a missing `ended_at`, which is a different question and
    the wrong one: only a task that exits gracefully stamps an end, so every
    attempt killed by OOM, a wall clock or a lost node stays `unresolved`
    forever. 1,296 of 6,031 rows read as live that way, 1,293 of them superseded
    attempts, and this part shipped 1.1 MB of them every five seconds -- on
    every page, because the status bar polls it too -- to draw two progress bars.

    A COPY -- the payload is memoised and shared with `/api/tasks`.
    """
    payload = part.get("payload")
    if not isinstance(payload, tasks.TasksPayload):
        return part
    rows = payload.rows
    # "Recent" is a POSITION, so ask for the position. The membership test this
    # replaced compared each of 15,684 rows against the last ten -- and against
    # models rather than dicts that would be a field-by-field compare each time.
    cut = len(rows) - LIVE_LIMIT if LIVE_LIMIT > 0 else 0
    kept = [row for index, row in enumerate(rows) if row.holds_a_node or index >= cut]
    # ADDED to what the command already hid, not recomputed from `rows`: the
    # part arrives bounded by `LIVE_WINDOW`, so `len(rows)` is the size of the
    # window and not of the log.
    updated = {"rows": kept, "hidden_rows": payload.hidden_rows + len(rows) - len(kept)}
    return {**part, "payload": payload.model_copy(update=updated)}


def run(run_id: str, *, invoke: Invoke | None = None) -> dict[str, Any]:
    """Everything about one run: what it is, how it trained, what it scored.

    `ledger` has a `--run` flag, so asking it for one run's evals is the command's
    own answer to its own question. `tasks` has none, so the run's tasks are drawn
    out of the full log by :func:`_tasks_for` -- the rule in miniature.

    `progress` is fetched with ``last=0`` deliberately: `runinfo` carries a progress
    array too, truncated to its `--last` default of eight.

    The `tasks` part is answered and then discarded down to the join, which is the
    only place in this module a payload does not reach the client whole. It has to
    be -- shipping it under `parts` as well would leave every byte of the task log
    on the wire.
    """
    composed = compose(
        "view-run",
        [
            Part("run", runinfo.COMMAND, {"run": run_id}),
            Part("progress", progress.COMMAND, {"run": run_id, "last": 0}),
            Part("curve", curve.COMMAND, {"run": run_id}),
            Part(
                "evals",
                ledger.COMMAND,
                {"run": run_id, "limit": 0},
            ),
            Part("tasks", tasks.COMMAND),
        ],
        join=lambda parts: {"run_tasks": _tasks_for(run_id, parts)},
        invoke=invoke,
    )
    composed["parts"]["tasks"] = _summarised(composed["parts"]["tasks"])
    return composed


def runs(*, invoke: Invoke | None = None) -> dict[str, Any]:
    """Every published run, with what is needed to check its claimed status.

    A run's `status` is a CLAIM: it is written by the training process, so it
    records what a LIVING process did and cannot record how an attempt died. A task
    killed by OOM, `maxWallClockTime`, SIGKILL or node loss leaves the run claiming
    `running` forever. Checking that needs Batch (which TASKS are live) joined to
    the task log (which RUN each task was for); neither can answer alone.

    **What this ships is the projection, not the verdict.** Deciding which Batch
    states count as live stays in the client, for the same reason :func:`now` does
    not draw the progress bar here: that is this module deciding what "running"
    means, which is the line.

    The OTHER half of the check -- has this run ever had a task at all -- is
    `RunSummary.has_tasks`, which the `runs` command answers in SQL beside the
    query already listing the runs. It was a join over the whole task log:
    16,895 legs and 10.6 MB fetched to learn 367 run ids, where a distinct scan
    costs 14 ms. That is what lets the tasks part be BOUNDED here -- all it
    still owes is which of Batch's current tasks belong to which run, and a task
    Batch is holding is by definition among the most recently written about.
    """
    composed = compose(
        "view-runs",
        [
            Part("runs", runs_command.COMMAND, {"limit": 0, "loadable_only": False}),
            Part("jobs", jobs.COMMAND, {"limit": RUN_LIST_JOB_LIMIT}),
            Part("tasks", tasks.COMMAND, {"limit": LIVE_WINDOW}),
        ],
        join=lambda parts: {"task_runs": _task_runs(parts)},
        invoke=invoke,
    )
    composed["parts"]["tasks"] = _summarised(composed["parts"]["tasks"])
    return composed


def _summarised(part: dict[str, Any]) -> dict[str, Any]:
    """A copy of one part with its `rows` replaced by how many there were.

    For a part fetched only to be joined against. The part stays -- dropping it
    would drop its `error`, which the UI needs to grey one panel rather than claim
    there is nothing to show.

    A DIFFERENT TYPE, not the same one emptied: `TasksSummary` has no `rows` field,
    so the generated TypeScript cannot offer one and the join is all there is.

    A COPY, emphatically. The payload is memoised per (command, arguments) and
    shared by every reader for the TTL, so trimming in place would hand the next
    caller of `/api/tasks` an empty task log.

    `source_rows` rather than silence, so a join returning few rows out of many is
    distinguishable from a join that is broken.
    """
    payload = part.get("payload")
    if not isinstance(payload, tasks.TasksPayload):
        return part
    summary = tasks.TasksSummary(source_rows=len(payload.rows), reconciled=payload.reconciled)
    return {**part, "payload": summary}


def _tasks_for(run_id: str, parts: dict[str, dict[str, Any]]) -> list[tasks.TaskRow]:
    """The task-log rows belonging to one run.

    The join is `task.run_id`, and it deliberately crosses jobs: a run outlives the
    daily job its tasks land in, so grouping by job would split one lineage for a
    reason that is purely about scheduling.

    Read from the task log rather than `runinfo.tasks`, which is empty for runs
    whose records predate it -- the production run among them.

    Returns ``[]`` when the tasks part failed, which the caller must not read as
    "this run has no tasks": the part carries its own error. A join cannot signal
    failure and must not try.
    """
    available = payloads(parts).get("tasks")
    if not isinstance(available, tasks.TasksPayload):
        return []
    return [row for row in available.rows if row.run_id == run_id]


def _task_runs(parts: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Which run each of Batch's CURRENT tasks belonged to: `task_id -> run_id`.

    The page asks ONE question of this: which runs have a task Batch is
    currently running. ("Has this run ever had a task" is `RunSummary.has_tasks`,
    answered in SQL.) So it is restricted to the jobs part -- a cross-reference between two parts,
    not a computed quantity. Unrestricted it was every pair in the log: 6,031 of
    them, 334 KB of a 455 KB screen, to look up a few dozen. Which Batch states
    count as live is still the client's call, which is the line that matters.

    Tasks with no `run_id` are dropped rather than mapped to null: they belong to
    no run, so they cannot answer a question about one.
    """
    available = payloads(parts).get("tasks")
    if not isinstance(available, tasks.TasksPayload):
        return {}
    current = _tasks_batch_holds(parts)
    return {
        row.task_id: row.run_id for row in available.rows if row.task_id in current and row.run_id
    }


def _tasks_batch_holds(parts: dict[str, dict[str, Any]]) -> set[str]:
    """Every task id the `jobs` part lists, whatever state it is in.

    Membership only -- reading a task's PHASE here would be this module deciding
    what "running" means, which is the client's decision.
    """
    available = payloads(parts).get("jobs")
    if not isinstance(available, jobs.JobsPayload):
        return set()
    return {task.task for job in available.jobs for task in job.tasks}
