"""The `reconcile-runs` subcommand: close runs whose task died without saying so.

A run's terminal status is written by the trainer, in-process: `mark_completed`
and `mark_failed` are the only writers. So when a task is killed -- OOM, the
guard's deadline, a lost node -- nobody writes it, and the run reads as
`running` forever. Nothing else ever closes it, and the count only grows.

That is not cosmetic. `prune-checkpoints` refuses to touch a run it believes is
training, correctly, so a zombie's whole ladder is unreachable disk; and every
reader shows work in flight that stopped days ago.

The evidence to settle it already exists and was simply never joined: a task's
outcome is known from its own exit record, or from Batch's observation of it
when the container died first. This walks that join.

DRY RUN BY DEFAULT. `--apply` is the only thing that writes.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.adapters.postgres import connect
from src.interfaces.commands import tasks as tasks_command
from src.interfaces.commands._base import Command, records_root
from src.shared import task_history

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from src.shared.ports.record import RecordSource

# A run is closable only when EVERY task speaking for it has settled, tested
# POSITIVELY against `task_history.TERMINAL_CAUSES` -- the same set that already
# gates task reconciliation, so there is one declaration of "this has stopped"
# rather than two that can drift.
#
# Positive is also the safe direction. Asking "is it in flight?" and closing
# otherwise would close a run whose cause is `unknown`, which is not evidence of
# death -- it is the absence of evidence.

# What the last task's cause makes the run. Anything not named here is `failed`.
#
# `completed` -> `abandoned` is the distinction worth having, and the dry run is
# what showed it was needed: 15 of 17 closable runs had a last task that exited
# CLEANLY. Nothing failed -- the task finished its chunk, published, and no one
# dispatched the continuation. Recording that as `failed` would put a false
# statement into a record that is the programme's evidence, and "the run is
# incomplete because nobody continued it" is a different fact from "it broke".
#
# It is terminal either way, which is what `prune-checkpoints` needs to know.
_STATUS_BY_CAUSE = {"cancelled": "cancelled", "completed": "abandoned"}

# Only a TRAINING task speaks for a run's status. A `score` task carries the
# run_id of the run it scores, so a finished evaluation would otherwise be read
# as evidence about the training -- and its `cause` would then pick the run's
# status. Scoring a run does say the training has stopped, but it says nothing
# about WHY, and `abandoned` versus `failed` is exactly that question.
_TRAINING_OPS = frozenset({"train", "train-vector", "train-pcs"})


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver reconcile-runs`."""
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the status events. Without it this prints the plan and touches nothing.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=None,
        help="Limit to this run; repeatable. Omit to consider every published run.",
    )


class Closure(BaseModel):
    """One run this would close, and the evidence for closing it."""

    run: str
    status: str
    task_id: str
    cause: str
    # The node's own account or Batch's. A cause Batch inferred is weaker
    # evidence than one the wrapper wrote, and the difference is worth seeing
    # before agreeing to write it down.
    cause_source: str
    ended_at: str | None = None


class ReconcilePlan(BaseModel):
    op: Literal["reconcile-runs"] = "reconcile-runs"
    applied: bool = False
    runs_considered: int = 0
    open_runs: int = 0
    closures: list[Closure] = Field(default_factory=list)
    # In flight, OR carrying a cause nothing can read. Both mean 'not settled',
    # and neither is grounds for writing a terminal status.
    unsettled: list[str] = Field(default_factory=list)
    no_evidence: list[str] = Field(default_factory=list)
    written: int = 0


def _status_of(run_dir: Path, source: RecordSource | None) -> str:
    """The run's own last word, through the fold that reads BOTH layouts.

    Not a raw scan of `run.jsonl`. A run written before the event log has a
    `.run.json` and no log, and reading only the log makes every one of those
    report `running` -- `tail_value` hands back its default on an empty list.
    That is how nine records with a perfectly good `.run.json` came to look like
    nine zombies. `RunMetadata.load` is the one place that knows both layouts,
    and the run-vs-attempt `status` scoping inside it is what stops a dead
    attempt's `died` reading as the run's own.

    `unknown` when nothing can be read, which is not a status any caller acts
    on: absence of evidence protects.
    """
    from src.pipeline.training.run_tracker.metadata import RunMetadata  # noqa: PLC0415

    try:
        return RunMetadata.load(run_dir, source).status or "unknown"
    except (OSError, ValueError, KeyError):
        return "unknown"


def run(args: argparse.Namespace) -> ReconcilePlan:
    """Decide which runs to close, and write only under `--apply`."""
    source = connect.record_source_from_environment()

    # Through `tasks`, not `task_history.read_tasks`, and that is the whole
    # point: `tasks` materialises legs/ AND reconciles the unresolved ones
    # against Batch first. The observer half of that join is exactly the
    # evidence this command needs -- a run whose container died has no exit
    # record of its own, and Batch's account is the only thing that can speak
    # for it. Reading the share directly would see those tasks as unexplained
    # and close nothing, which is what a first draft of this did.
    by_run: dict[str, list[Any]] = defaultdict(list)
    for row in tasks_command.COMMAND.invoke_as(tasks_command.TasksPayload).rows:
        if row.run_id:
            by_run[row.run_id].append(row)

    plan = ReconcilePlan(applied=bool(args.apply))
    with records_root(args) as root:
        wanted = sorted(p for p in root.iterdir() if p.is_dir())
        if args.runs:
            names = set(args.runs)
            wanted = [p for p in wanted if p.name in names]
        plan.runs_considered = len(wanted)

        for run_dir in wanted:
            if _status_of(run_dir, source) != "running":
                continue
            plan.open_runs += 1
            tasks = by_run.get(run_dir.name, [])
            if not tasks:
                # ABSENCE OF EVIDENCE PROTECTS. A run with no task record is not
                # a run that finished -- it is one nothing can speak for, and
                # inventing a terminal status for it would be a guess written
                # into the record as a fact.
                plan.no_evidence.append(run_dir.name)
                continue
            if not all(task.cause in task_history.TERMINAL_CAUSES for task in tasks):
                plan.unsettled.append(run_dir.name)
                continue
            training = [t for t in tasks if t.op in _TRAINING_OPS]
            if not training:
                # Scored but with no training task on record: something stopped
                # it, and nothing here knows what.
                plan.no_evidence.append(run_dir.name)
                continue
            last = max(training, key=lambda t: (t.ended_at or "", t.task_id))
            plan.closures.append(
                Closure(
                    run=run_dir.name,
                    status=_STATUS_BY_CAUSE.get(last.cause, "failed"),
                    task_id=last.task_id,
                    cause=last.cause,
                    cause_source=last.cause_source,
                    ended_at=last.ended_at,
                )
            )

    if not args.apply:
        return plan

    # THROUGH THE SINK, which is where a terminal status lives. This appended
    # the event to `run.jsonl` on the share, and kept doing so after nothing
    # wrote or read that file: 19 runs were reported CLOSED while every reader
    # went on showing them as training. It also skipped any run with no such
    # file, so a run created after the flip could never be reconciled at all --
    # exactly the zombie this command exists to clear.
    with connect.record_sink() as sink:
        for closure in plan.closures:
            sink.closed(
                closure.run,
                closure.status,
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "status": closure.status,
                    # An INFERENCE, and the record says so rather than passing it
                    # off as a first-hand report. A reader that cannot tell the
                    # two apart is one that will eventually trust the wrong one.
                    "reconciled": True,
                    "from_task": closure.task_id,
                    "task_cause": closure.cause,
                    "cause_source": closure.cause_source,
                },
            )
            plan.written += 1
    return plan


def render(payload: ReconcilePlan) -> None:
    verb = "CLOSED" if payload.applied else "would close"
    print(
        f"{payload.runs_considered} runs considered, {payload.open_runs} still open, "
        f"{len(payload.closures)} {verb}"
    )
    for closure in payload.closures:
        print(
            f"  {closure.run[:52]:<54} -> {closure.status:<9}"
            f" ({closure.cause} per {closure.cause_source})"
        )
    if payload.unsettled:
        print(f"\nnot settled, so left open ({len(payload.unsettled)}):")
        for name in payload.unsettled:
            print(f"  {name}")
    if payload.no_evidence:
        print(f"\nno task record, so left open ({len(payload.no_evidence)}):")
        for name in payload.no_evidence:
            print(f"  {name}")
    if payload.applied:
        print(f"\nstatus events written: {payload.written}")
    else:
        print("\nDRY RUN -- nothing was written. Re-run with --apply to execute this plan.")


COMMAND = Command(
    name="reconcile-runs",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Close runs whose task died without recording a terminal status (dry run by default).",
)
