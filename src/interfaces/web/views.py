"""One screen, one request -- composed from commands and nothing else.

    **This layer may COMPOSE command payloads. It may not COMPUTE one.**

A view fans out over :meth:`Command.invoke` and joins what comes back. A join
may filter, group and cross-reference; it may not derive a quantity no command
can answer. `tests/interfaces/web/test_no_second_read_path.py` is what says so,
and the `web_reads_through_the_command_layer` import-linter contract is what
stops this module reaching past the command layer to do it anyway.

Every view returns its raw ``parts`` alongside its joins. The joins are a
convenience, never a replacement: a part that failed still has to reach the UI
as a reason, so one panel greys out and the rest of the screen survives.
"""

from __future__ import annotations

from typing import Any

from src.interfaces.commands import (
    autoscale_check,
    cost,
    curve,
    jobs,
    ledger,
    pool_status,
    progress,
    report,
    runinfo,
    tasks,
)
from src.interfaces.commands import runs as runs_command
from src.interfaces.commands._compose import Part, compose, payloads

# A glanceable screen cannot carry two hundred rows, and the cost of fetching
# them is the point: `tasks` is the slowest read in the console.
LIVE_LIMIT = 10

"""Why the run list asks for FIFTY jobs and the live screen asks for ten
----------------------------------------------------------------------
They are asking different questions. `now` shows what is happening, so ten is
generous. The run list uses jobs to decide whether a run that CLAIMS to be
running actually has a task executing -- and a run's live task can sit in a job
well down the list, because a run outlives the daily job its tasks land in.

Reusing `LIVE_LIMIT` here would make a run whose task is in the eleventh-most-
recent job read as abandoned: a false alarm on the exact screen the check exists
to make trustworthy. Pinned by a test, since nothing else would notice.
"""
RUN_LIST_JOB_LIMIT = 50


def now() -> dict[str, Any]:
    """What is happening right now, and did anything die.

    Five questions that were five requests. `cost` rides along because burn rate
    belongs beside node count -- it is the same question asked in dollars -- and
    it is nearly free here: it derives from the task log this view is already
    paying for, and the billing half is memoised for 15 minutes server-side
    because Cost Management is rate-limited hard.

    No join. Everything on this screen is a panel in its own right, and the one
    cross-reference the client draws -- a running task's progress bar, which
    needs Batch for *which* task holds a node and the task log for *how far
    along* it is -- is presentation, not data. Moving it here would mean this
    module deciding what "running" means, which is the line.
    """
    return compose(
        "view-now",
        [
            Part("pool", pool_status.COMMAND),
            Part("jobs", jobs.COMMAND, {"limit": LIVE_LIMIT}),
            Part("tasks", tasks.COMMAND, {"limit": LIVE_LIMIT}),
            Part("autoscale", autoscale_check.COMMAND),
            Part("cost", cost.COMMAND),
        ],
    )


def run(run_id: str) -> dict[str, Any]:
    """Everything about one run: what it is, how it trained, what it scored.

    Five parts, one of which is filtered by the command itself and one of which
    is joined here -- and the difference is worth naming, because it is the rule
    in miniature. `ledger` has a `--run` flag, so asking it for one run's evals
    is the command's own answer to its own question. `tasks` has no such flag,
    so the run's tasks are drawn out of the full log by :func:`_tasks_for`.

    `progress` is fetched with ``last=0`` deliberately. `runinfo` carries a
    progress array too, but truncated to its `--last` default of eight, so the
    console's chart drew 8 of 112 checkpoints and looked like a complete
    history.

    The `tasks` part is answered and then **discarded down to the join**, which
    is the only place in this module a payload does not reach the client whole.
    It has to be: the whole point of joining here is that a run's page stops
    carrying the entire task log, and shipping it under `parts` as well would
    move the filtering off the browser while leaving every byte of it on the
    wire. :func:`_summarised` keeps the part -- and therefore its error, and
    therefore the greyed panel -- while replacing its rows with a count.
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
    )
    composed["parts"]["tasks"] = _summarised(composed["parts"]["tasks"])
    return composed


def runs() -> dict[str, Any]:
    """Every published run, with what is needed to check its claimed status.

    A run's `status` is a CLAIM, not an observation: it lives in the run's own
    event log, written by the training process, so it records what a LIVING
    process did and cannot record how an attempt died. A task killed by OOM,
    `maxWallClockTime`, SIGKILL or node loss is gone before it can write
    `finished`, and the run then claims `running` forever -- four runs on this
    share have.

    Checking that needs three sources and two joins: Batch knows which TASKS are
    live, the task log knows which RUN each task belonged to, and neither can
    answer alone. The console did this itself, which cost it the whole task log
    on a page that shows a table of run names.

    **What this ships is the projection, not the verdict.** `task_runs` is
    `task_id -> run_id` and nothing more. Deciding which Batch states count as
    live -- and whether a run with no live task is "abandoned" or merely
    "abandoned?" because it predates the task log -- stays in the client, for the
    same reason :func:`now` does not draw the progress bar here: that is this
    module deciding what "running" means, which is the line. The client already
    holds the `jobs` part, so it can intersect the two without another request.
    """
    composed = compose(
        "view-runs",
        [
            Part("runs", runs_command.COMMAND, {"limit": 0, "loadable_only": False}),
            Part("jobs", jobs.COMMAND, {"limit": RUN_LIST_JOB_LIMIT}),
            Part("tasks", tasks.COMMAND),
        ],
        join=lambda parts: {"task_runs": _task_runs(parts)},
    )
    composed["parts"]["tasks"] = _summarised(composed["parts"]["tasks"])
    return composed


def experiment(experiment_id: str) -> dict[str, Any]:
    """One experiment's arms, each pinned to the run record behind it.

    `report` already pairs every variant against its control at one knob tier --
    that pairing is the command's, and this view must not second-guess it. What
    it adds is the join `report` cannot make without becoming a different
    command: each arm's row in `runs`, so the page can say what an arm WAS
    (config, iterations, status) beside how it scored.
    """
    return compose(
        "view-experiment",
        [
            Part("report", report.COMMAND, {"experiment": experiment_id}),
            Part("runs", runs_command.COMMAND, {"limit": 0, "loadable_only": False}),
        ],
        join=lambda parts: {"arm_runs": _runs_in(experiment_id, parts)},
    )


def _summarised(part: dict[str, Any]) -> dict[str, Any]:
    """A copy of one part with its `rows` replaced by how many there were.

    For a part fetched only to be joined against. The part stays -- a view that
    dropped it entirely would drop its `error` with it, and the UI needs that to
    grey one panel rather than claim there is nothing to show -- but its rows do
    not go on the wire, because the join is what the client asked for.

    A DIFFERENT TYPE, not the same one with its rows emptied. `Tasks` and
    `TasksSummary` describe two different things, and while one model covered
    both, `parts.tasks.payload.rows` was `[]` on a trimmed part and correct about
    nothing -- a page avoided that only by remembering to read the join instead.
    `TasksSummary` has no `rows` field, so the generated TypeScript cannot offer
    one and the join is all there is.

    **A copy, emphatically.** Trimming the payload in place edits the object the
    command layer returned, which is not this module's to edit: the payload is
    memoised per (command, arguments) and shared by every reader for the TTL, so
    a view that empties `rows` on the way past hands the next caller of
    `/api/tasks` an empty task log. It surfaced here as a run page that was
    correct once and empty afterwards.

    `source_rows` rather than silence, so a run page that shows two tasks out of
    a log of four hundred can say which number is which. A join that quietly
    returns few rows out of many is indistinguishable from a join that is broken.
    """
    payload = part.get("payload")
    if not isinstance(payload, dict) or "rows" not in payload:
        return part
    summary = tasks.TasksSummary(
        source_rows=len(payload["rows"]), reconciled=payload.get("reconciled")
    )
    return {**part, "payload": summary.model_dump()}


def _tasks_for(run_id: str, parts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The task-log rows belonging to one run.

    The join is `task.run_id`, and it deliberately crosses jobs: a run outlives
    the daily job its tasks happen to land in, so grouping by job would split
    one lineage across three headings for a reason that is purely about
    scheduling.

    Read from the task log rather than `runinfo.tasks` because that field is
    empty for runs whose records predate it -- the production run among them --
    and the task log is the durable account either way.

    Returns ``[]`` when the tasks part failed, which the caller must not read as
    "this run has no tasks": the part carries its own error, and that is what
    the UI renders. A join cannot signal failure and must not try.
    """
    available = payloads(parts).get("tasks")
    if available is None:
        return []
    return [row for row in available.get("rows", []) if row.get("run_id") == run_id]


def _task_runs(parts: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Which run each task belonged to: `task_id -> run_id`.

    A projection of the task log, and the reason the run list no longer
    downloads it. Hundreds of short pairs instead of hundreds of full rows, and
    it carries everything the two client-side joins need -- which runs have
    tasks at all, and which of those tasks Batch currently has live.

    Tasks with no `run_id` are dropped rather than mapped to null: they belong to
    no run, so they cannot answer a question about one.
    """
    available = payloads(parts).get("tasks")
    if available is None:
        return {}
    return {
        row["task_id"]: row["run_id"]
        for row in available.get("rows", [])
        if row.get("task_id") and row.get("run_id")
    }


def _runs_in(experiment_id: str, parts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The run records tagged with this experiment, keyed for the arms above."""
    available = payloads(parts).get("runs")
    if available is None:
        return []
    return [row for row in available.get("runs", []) if row.get("experiment_id") == experiment_id]
