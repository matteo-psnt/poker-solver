"""One screen, one request -- composed from commands and nothing else.

The rule this package has always been subordinate to is that it must not grow a
second read path: `fbcf9a8` deleted a browser UI (-4,776 lines) whose
`chart_service.py`, `play_service.py` and `chart/data.py` answered questions the
CLI already answered, drifted from it, and rotted with nothing failing until
someone looked.

That rule was enforced as *every endpoint body is one `Command.invoke`*, which
stopped the disease by forbidding composition. But composition was never the
disease -- DERIVING answers was -- and forbidding it moved the composing onto
the browser, where it cost three things:

latency
    A screen is several questions. Asked as separate requests they are separate
    round trips, each with its own cache slot and its own poll cadence, so the
    Overview's four panels are never the same age as each other.
transfer
    `RunDetail` fetched the ENTIRE task log and filtered it client-side to find
    one run's tasks -- over the wire, into the browser, to discard ~95% of it --
    because `runinfo.tasks` comes back empty for runs whose records predate that
    field, including the production run.
the information architecture
    Fourteen routes, one per command, because a page that can only ask one
    question can only be about one command. `CONSOLE-DESIGN.md` specified five
    destinations; the console grew to fourteen by following the CLI's grouping,
    which is the right organising idea for a reference and the wrong one for a
    console.

So the rule is restated, strictly stronger and still mechanically checked:

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
    runs,
    tasks,
)
from src.interfaces.commands._compose import Part, compose, payloads

# A glanceable screen cannot carry two hundred rows, and the cost of fetching
# them is the point: `tasks` is the slowest read in the console.
LIVE_LIMIT = 10


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
            Part(
                "tasks",
                tasks.COMMAND,
                {"limit": LIVE_LIMIT, "skip_reconcile": False, "tasks_dir": None},
            ),
            Part("autoscale", autoscale_check.COMMAND),
            Part("cost", cost.COMMAND, {"hours": 0.0, "rate": ""}),
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
                {"run": run_id, "limit": 0, "experiment": None, "method": None, "since": None},
            ),
            Part("tasks", tasks.COMMAND, {"limit": 0, "skip_reconcile": False, "tasks_dir": None}),
        ],
        join=lambda parts: {"run_tasks": _tasks_for(run_id, parts)},
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
            Part("runs", runs.COMMAND, {"limit": 0, "loadable_only": False}),
        ],
        join=lambda parts: {"arm_runs": _runs_in(experiment_id, parts)},
    )


def _summarised(part: dict[str, Any]) -> dict[str, Any]:
    """A copy of one part with its `rows` replaced by how many there were.

    For a part fetched only to be joined against. The part stays -- a view that
    dropped it entirely would drop its `error` with it, and the UI needs that to
    grey one panel rather than claim there is nothing to show -- but its rows do
    not go on the wire, because the join is what the client asked for.

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
    return {**part, "payload": {**payload, "rows": [], "source_rows": len(payload["rows"])}}


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


def _runs_in(experiment_id: str, parts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The run records tagged with this experiment, keyed for the arms above."""
    available = payloads(parts).get("runs")
    if available is None:
        return []
    return [row for row in available.get("runs", []) if row.get("experiment_id") == experiment_id]
