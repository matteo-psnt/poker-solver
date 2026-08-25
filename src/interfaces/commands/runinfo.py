"""The `runinfo` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from src.interfaces.commands._base import (
    Command,
    ledger_for,
    num,
    pct,
    records_root,
    resolve_run_dir,
)
from src.pipeline import services

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver runinfo`."""
    parser.add_argument("--run", required=True, help="Run id (dir name) or path.")
    parser.add_argument(
        "--tier", type=int, default=0, help="Which comparison tier's curve to show."
    )
    parser.add_argument(
        "--tasks-dir",
        default=None,
        help=(
            "Directory CONTAINING a legs/ -- `--tasks-dir data`, not `data/tasks`. "
            "Omit to read the tasks published to the share, which is the normal case."
        ),
    )
    parser.add_argument("--last", type=int, default=8, help="Checkpoints to show (0 = all).")


class RunInfoPayload(services.RunDigest):
    """Everything recorded about one run, joined into one view.

    Subclasses the digest the service already produces. What this adds is what
    the SURFACE needs and the digest does not carry: the op tag, the truncation
    of `progress` to its tail, and the count of what was truncated.

    `attempts` is RENAMED to `training_tasks`, not duplicated: the digest's word
    is about the task log and the reader's question is about the run. The
    inherited field is redeclared with ``exclude=True`` so exactly one of the two
    names crosses the wire -- subclassing would otherwise ship the same number
    twice, which the dict this replaced avoided by popping the old key.
    """

    op: Literal["runinfo"] = "runinfo"
    """TRUNCATED to `--last`. `progress` with `--last 0` is the full series --
    the console's chart drew 8 of 112 checkpoints from this field and looked
    like a complete history, which is why the total travels beside it."""
    total_progress_rows: int = 0
    training_tasks: int | None = None
    attempts: int = Field(default=0, exclude=True)


def run(args: argparse.Namespace) -> RunInfoPayload:
    """Everything recorded about one run, joined into one view.

    The evidence is spread across artifacts written by four subsystems; this is
    the one place that holds the joins so you do not have to.
    """
    with records_root(args) as root:
        digest = services.run_digest(
            resolve_run_dir(args.run, str(root)),
            ledger_path=ledger_for(root),
            tier_index=args.tier,
            tasks_dir=Path(args.tasks_dir) if args.tasks_dir else None,
        )
    fields = digest.model_dump()
    # Trimmed to the tail: a 30M run has thirty checkpoints and the reader wants
    # the shape, not the log.
    fields["progress"] = digest.progress[-args.last :] if args.last > 0 else digest.progress
    return RunInfoPayload(
        **fields,
        total_progress_rows=len(digest.progress),
        training_tasks=digest.attempts,
    )


def render(payload: RunInfoPayload) -> None:
    tag = ""
    if payload.experiment_id:
        tag = f"  {payload.experiment_id}/{payload.arm or '-'}"
    print(f"{payload.run_id}  {payload.config_name}{tag}")

    commit = (payload.git_commit or "unknown")[:8]
    dirty = " (dirty)" if payload.git_dirty else ""
    abstraction = (payload.card_abstraction_hash or "none")[:16]
    print(f"  git {commit}{dirty}   abstraction {abstraction}   status {payload.status}")
    print(
        f"  {payload.iterations:,} iterations over {payload.training_tasks} training task(s), "
        f"{payload.runtime_seconds:.0f}s compute"
    )

    rows = payload.progress or []
    if rows:
        total = payload.total_progress_rows
        print(f"\n  progress  ({total} checkpoints, last {len(rows)})")
        print(f"    {'iteration':>12}{'coverage':>10}{'visits':>9}{'it/s':>8}")
        for row in rows:
            print(
                f"    {row.get('iteration', 0):>12,}"
                f"{pct(row.get('coverage')):>10}"
                f"{num(row.get('mean_visits_per_touched'), '{:.1f}'):>9}"
                f"{num(row.get('iters_per_sec'), '{:.0f}'):>8}"
            )
        flat = payload.coverage_flat_from
        if flat is not None:
            print(f"    coverage flat from {flat:,}")

    points = payload.curve.points
    if points:
        print(f"\n  curve  ({payload.curve.tier or 'untiered'})")
        for point in points:
            print(
                f"    {point.iteration:>12,}  {point.exploitability_mbb:>9.1f} mbb"
                f"  (± {point.std_error_mbb:.1f})"
            )
        # Without this the curve of a run whose eval tree changed simply STOPS and
        # the completed evals below it are invisible -- read once as ten scoring
        # tasks having produced nothing.
        others = (
            i for i in range(len(payload.curve.other_tiers) + 1) if i != payload.curve.tier_index
        )
        for index, other in zip(others, payload.curve.other_tiers, strict=True):
            print(f"    (also recorded, not mixed in -- `--tier {index}`: {other})")

    tasks = payload.tasks or []
    if tasks:
        print(f"\n  tasks  ({len(tasks)})")
        for task in tasks:
            print(f"    {task.task_id:<28} {f'#{task.attempt}':<4} {task.cause}")

    gaps = payload.gaps or []
    print("\n  gaps" if gaps else "\n  no gaps: scored, complete, reproducible")
    for gap in gaps:
        print(f"    - {gap}")


COMMAND = Command(
    name="runinfo",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Everything recorded about a run: provenance, curve, scores, tasks, gaps.",
)
