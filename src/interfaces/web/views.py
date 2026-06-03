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

# Deliberately NOT `LIVE_LIMIT`. The run list uses jobs to check whether a run
# claiming to be running has a task executing, and a run outlives the daily job
# its tasks land in -- so a smaller limit reads a live run as abandoned.
RUN_LIST_JOB_LIMIT = 50


def now() -> dict[str, Any]:
    """What is happening right now, and did anything die.

    Five questions that were five requests. `cost` rides along because burn rate
    belongs beside node count, and it is nearly free: it derives from the task log
    this view already pays for, and the billing half is memoised server-side.

    No join. Everything here is a panel in its own right, and the one
    cross-reference the client draws -- a running task's progress bar -- is
    presentation, not data.
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
    )
    composed["parts"]["tasks"] = _summarised(composed["parts"]["tasks"])
    return composed


def runs() -> dict[str, Any]:
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
    if not isinstance(payload, dict) or "rows" not in payload:
        return part
    summary = tasks.TasksSummary(
        source_rows=len(payload["rows"]), reconciled=payload.get("reconciled")
    )
    return {**part, "payload": summary.model_dump()}


def _tasks_for(run_id: str, parts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
