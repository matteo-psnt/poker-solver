"""The `progress` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.interfaces.commands._base import (
    Command,
    num,
    pct,
    records_root,
    resolve_run_dir,
)
from src.interfaces.errors import CommandError
from src.shared import records, run_events

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver progress`."""
    parser.add_argument("--run", required=True, help="Run id (dir name) or path.")
    parser.add_argument(
        "--last", type=int, default=25, help="Show only the last N checkpoints (0 = all)."
    )


class ProgressRow(BaseModel):
    """One checkpoint's line of `progress.jsonl`.

    LENIENT and almost entirely optional, and both are the shape of the data
    rather than laziness: a resumed run appends across tasks, so one log spans
    code versions and an old row genuinely lacks fields a new one carries.
    Blanking those is the honest rendering; inventing a zero is not.
    """

    model_config = ConfigDict(extra="allow")

    iteration: int
    coverage: float | None = None
    """What the LEG cost, as opposed to what the run has done. Present only on
    rows written by a version that recorded them."""
    task_elapsed_s: float | None = None
    checkpoint_seconds: float | None = None
    """THE convergence diagnostic. Compare against the 1e3-1e4 regret updates
    per infoset CFR needs; coverage saturates early and says nothing about it."""
    mean_visits_per_touched: float | None = None
    iters_per_sec: float | None = None
    schema_version: int | None = None


class ProgressPayload(BaseModel):
    """A run's per-checkpoint history: the only thing that can say how it went.

    `rows` is TRUNCATED by `--last`; `total_rows` is how many there were. A
    chart drawn from a truncated series looks like a complete measurement, which
    is why the count travels beside it.
    """

    op: Literal["progress"] = "progress"
    run_id: str
    total_rows: int
    """The span of record schema versions in the series. A resumed run appends
    across tasks, so one log legitimately spans code versions."""
    schema_version_min: int
    schema_version_max: int
    coverage_plateau_iteration: int | None = None
    rows: list[ProgressRow] = Field(default_factory=list)


def run(args: argparse.Namespace) -> ProgressPayload:
    """A run's per-checkpoint history.

    The run record holds a start and an end; this is the only thing that can say
    where coverage plateaued or whether throughput decayed over wall clock.
    """
    with records_root(args) as root:
        run_dir = resolve_run_dir(args.run, str(root))
        rows = run_events.checkpoints(run_events.read(run_dir))
    if not rows:
        raise CommandError(
            f"No checkpoint history in {run_dir}. It is recorded per checkpoint, so "
            "the run must have reached one under a version that records it."
        )
    low, high = records.version_span(rows)
    return ProgressPayload(
        run_id=run_dir.name,
        total_rows=len(rows),
        schema_version_min=low,
        schema_version_max=high,
        coverage_plateau_iteration=run_events.plateau_iteration(rows),
        rows=[
            ProgressRow.model_validate(row)
            for row in (rows[-args.last :] if args.last > 0 else rows)
        ],
    )


def render(payload: ProgressPayload) -> None:
    rows = payload.rows
    span = (payload.schema_version_min, payload.schema_version_max)
    version = f"v{span[0]}" if span[0] == span[1] else f"v{span[0]}-v{span[1]} (mixed)"
    print(f"Progress for {payload.run_id}")
    print(f"  {payload.total_rows} checkpoints, schema {version}; showing {len(rows)}")
    # it/s and task time are per LEG: a resumed task restarts its clock, so these
    # compare within a task, not across the run.
    print(
        f"    {'iteration':>12}{'coverage':>10}{'visits':>9}{'it/s':>8}{'task time':>10}{'ckpt s':>8}"
    )
    for row in rows:
        print(
            f"    {(row.iteration or 0):>12,}"
            f"{pct(row.coverage):>10}"
            f"{num(row.mean_visits_per_touched, '{:.1f}'):>9}"
            f"{num(row.iters_per_sec, '{:.0f}'):>8}"
            f"{num(row.task_elapsed_s, '{:.0f}s'):>10}"
            f"{num(row.checkpoint_seconds, '{:.1f}'):>8}"
        )
    plateau = payload.coverage_plateau_iteration
    if plateau is not None:
        print(f"  coverage flat from iteration {plateau:,} (2+ intervals under 1% gain)")
    else:
        print("  coverage still climbing at the last checkpoint")


COMMAND = Command(
    name="progress",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Per-checkpoint coverage, visits and throughput for a run.",
)
