"""The `progress` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.cli.commands._base import (
    Command,
    records_root,
    resolve_run_dir,
)
from src.interfaces.errors import CommandError
from src.shared import records, run_events


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run progress`."""
    parser.add_argument("--run", required=True, help="Run id (dir name) or path.")
    parser.add_argument(
        "--last", type=int, default=25, help="Show only the last N checkpoints (0 = all)."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
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
    return {
        "op": "progress",
        "run_id": run_dir.name,
        "total_rows": len(rows),
        "schema_version_min": low,
        "schema_version_max": high,
        "coverage_plateau_iteration": run_events.plateau_iteration(rows),
        "rows": rows[-args.last :] if args.last > 0 else rows,
    }


def render(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    span = (payload["schema_version_min"], payload["schema_version_max"])
    version = f"v{span[0]}" if span[0] == span[1] else f"v{span[0]}-v{span[1]} (mixed)"
    print(f"Progress for {payload['run_id']}")
    print(f"  {payload['total_rows']} checkpoints, schema {version}; showing {len(rows)}")
    # it/s and leg time are per LEG: a resumed leg restarts its clock, so these
    # compare within a leg, not across the run.
    print(
        f"    {'iteration':>12}{'coverage':>10}{'visits':>9}{'it/s':>8}{'leg time':>10}{'ckpt s':>8}"
    )
    for row in rows:
        print(
            f"    {row.get('iteration', 0):>12,}"
            f"{_pct(row.get('coverage')):>10}"
            f"{_num(row.get('mean_visits_per_touched'), '{:.1f}'):>9}"
            f"{_num(row.get('iters_per_sec'), '{:.0f}'):>8}"
            f"{_num(row.get('leg_elapsed_s'), '{:.0f}s'):>10}"
            f"{_num(row.get('checkpoint_seconds'), '{:.1f}'):>8}"
        )
    plateau = payload.get("coverage_plateau_iteration")
    if plateau is not None:
        print(f"  coverage flat from iteration {plateau:,} (2+ intervals under 1% gain)")
    else:
        print("  coverage still climbing at the last checkpoint")


def _pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, int | float) else ""


def _num(value: Any, fmt: str) -> str:
    """Blank rather than a placeholder when a field predates its schema version."""
    return fmt.format(value) if isinstance(value, int | float) else ""


COMMAND = Command(
    name="progress",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Per-checkpoint coverage, visits and throughput for a run.",
)
