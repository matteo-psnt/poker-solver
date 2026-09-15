"""The `wait` subcommand: block until a task or a run reaches a state.

The opposite of `status --watch`, on purpose. There the loop lives in the
RENDERER, so `--json` returns one snapshot and a machine consumer polls on its
own schedule. Here blocking IS the answer -- the caller wants to do nothing
until the condition holds -- so the loop lives in `run`, and `--json` blocks
exactly like the terminal does. Measured across 56 sessions: 1,532 poll calls
to `jobs`/`logs`/`tasks` and 35 refused `sleep && tail` loops, every one a
turn spent asking a question this command answers once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands import tasks
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared import task_history, task_states
from src.shared.task_history import TaskRow

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

# Below this a tick is a hammer on Batch and the record; `status --watch` has
# the same floor for the same reason.
MIN_EVERY = 10
# One call fits the Bash tool's 10-minute cap with a margin for the last tick.
DEFAULT_TIMEOUT = 540

Condition = Literal["started", "finished", "log", "rung"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver wait`."""
    parser.add_argument("--task", default="", help="Task id to wait on.")
    parser.add_argument(
        "--until",
        choices=["started", "finished", "log"],
        default="finished",
        help="For --task: started = holds a node and is running; finished = a terminal "
        "cause (completed, failed, killed, ...); log = its log has been published.",
    )
    parser.add_argument("--run", default="", help="Run id (or fragment) to wait on, with --rung.")
    parser.add_argument(
        "--rung", type=int, default=0, help="For --run: return once a retained rung >= N exists."
    )
    parser.add_argument(
        "--every",
        type=int,
        default=30,
        help=f"Seconds between looks (minimum {MIN_EVERY}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Give up after this many seconds and exit 1 (default {DEFAULT_TIMEOUT}, "
        "which fits one Bash call; 0 = wait forever).",
    )


class WaitPayload(BaseModel):
    """Where the wait stopped, and what it saw there."""

    op: Literal["wait"] = "wait"
    target: str
    condition: Condition
    reached: bool
    waited_seconds: float
    ticks: int
    row: TaskRow | None = None
    rung: int | None = None


@dataclass(frozen=True)
class Seen:
    """One look at the target: whether the condition holds, and the evidence."""

    reached: bool
    row: TaskRow | None = None
    rung: int | None = None

    def describe(self) -> str:
        if self.row is not None:
            return f"last seen {self.row.cause} (attempt {self.row.attempt})"
        if self.rung is not None:
            return f"ladder head at {self.rung}"
        return "nothing recorded yet"


def poll(
    tick: Callable[[], Seen],
    *,
    target: str,
    condition: Condition,
    every: int,
    timeout: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WaitPayload:
    """Look, sleep, look again, until the condition holds or the timeout would pass.

    A timeout is a `CommandError` rather than a payload with `reached=False`:
    exit 1 is what a Monitor until-loop keys on, and a caller that wants the
    partial answer has `--json` on the tick that succeeds.
    """
    started = clock()
    ticks = 0
    while True:
        ticks += 1
        seen = tick()
        waited = clock() - started
        if seen.reached:
            return WaitPayload(
                target=target,
                condition=condition,
                reached=True,
                waited_seconds=round(waited, 1),
                ticks=ticks,
                row=seen.row,
                rung=seen.rung,
            )
        # Do not start a sleep the deadline would cut short: the caller's
        # own timeout kills the process with nothing printed, and the point of
        # this refusal is the line saying how far the target got.
        if timeout and waited + every > timeout:
            raise CommandError(
                f"wait: {target} did not reach {condition} in {waited:.0f}s "
                f"({ticks} look(s)); {seen.describe()}."
            )
        sleep(every)


def _latest(rows: list[TaskRow]) -> TaskRow | None:
    """The attempt Batch describes: an earlier one is settled by construction."""
    if not rows:
        return None
    return max(rows, key=lambda row: row.attempt)


def _task_tick(task_id: str, condition: Condition) -> Callable[[], Seen]:
    """A look at one task, through `tasks` so Batch reconcile runs.

    Reconcile is the point: a task the OOM killer took sits at `started` in the
    node's own account forever, and only Batch can say it is gone. Skipping it
    would wait on a corpse until the timeout.
    """
    if condition == "log":
        from src.interfaces.cloud.config import CloudConfig  # noqa: PLC0415 -- Azure only here
        from src.interfaces.cloud.store import blob  # noqa: PLC0415

        config = CloudConfig.load()

        def look_for_log() -> Seen:
            return Seen(reached=f"{task_id}.log" in blob.task_log_names(config))

        return look_for_log

    def look_at_record() -> Seen:
        payload = tasks.COMMAND.invoke_as(tasks.TasksPayload, task=task_id)
        row = _latest(payload.rows)
        if row is None:
            return Seen(reached=False)
        if condition == "started":
            done = row.phase in {task_states.Phase.RUNNING, task_states.Phase.FINISHED}
        else:
            done = row.cause in task_history.TERMINAL_CAUSES
        return Seen(reached=done, row=row)

    return look_at_record


def _rung_tick(run: str, rung: int) -> Callable[[], Seen]:
    """A look at one run's retained ladder."""
    from src.adapters.postgres import connect, queries  # noqa: PLC0415 -- 1.2s of SQLAlchemy
    from src.interfaces.commands._base import resolve_run_id  # noqa: PLC0415

    engine = connect.engine_from_environment()
    run_id = resolve_run_id(run, engine)

    def look_at_ladder() -> Seen:
        ladder = queries.rung_ladder(engine, run_id)
        head = max(ladder) if ladder else None
        return Seen(reached=head is not None and head >= rung, rung=head)

    return look_at_ladder


def plan(args: argparse.Namespace) -> tuple[str, Condition]:
    """What is being waited on, refusing the combinations that mean nothing."""
    if bool(args.task) == bool(args.run):
        raise CommandError("wait: give exactly one of --task <id> or --run <id> --rung N.")
    if args.task and args.rung:
        raise CommandError("wait: --rung is a run's condition; a task takes --until.")
    if args.run and not args.rung:
        raise CommandError("wait: --run needs --rung N, the retained rung to wait for.")
    if args.run:
        return args.run, "rung"
    return args.task, args.until


def run(args: argparse.Namespace) -> WaitPayload:
    """Block until the target reaches the condition, then report what it saw."""
    target, condition = plan(args)
    tick = _rung_tick(args.run, args.rung) if condition == "rung" else _task_tick(target, condition)
    return poll(
        tick,
        target=target,
        condition=condition,
        every=max(args.every, MIN_EVERY),
        timeout=max(args.timeout, 0),
    )


def render(payload: WaitPayload) -> None:
    print(
        f"{payload.target} reached {payload.condition} after {payload.waited_seconds:.0f}s "
        f"({payload.ticks} look(s))"
    )
    if payload.row is not None:
        print(tasks.format_table([payload.row]))
    elif payload.rung is not None:
        print(f"  ladder head at rung {payload.rung:,}")


COMMAND = Command(
    name="wait",
    help="Block until a task starts, finishes or logs, or a run reaches a rung.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
