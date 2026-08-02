"""The `promote` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands._base import (
    Command,
    resolve_run_dir,
)
from src.pipeline import services


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run promote`."""
    parser.add_argument("--run", required=True, help="Run id to promote.")
    parser.add_argument(
        "--rationale",
        required=True,
        help="Why this run becomes the baseline. Required — a lineage that moved for "
        "an unrecorded reason cannot be audited later.",
    )
    parser.add_argument("--runs-dir", default="data/runs", help="Base runs dir for id resolution.")
    parser.add_argument(
        "--baseline", default=str(services.DEFAULT_BASELINE_PATH), help="Baseline pointer file."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Point the baseline at a run, closing one turn of the base-fork loop."""
    run_dir = resolve_run_dir(args.run, args.runs_dir)
    baseline = services.promote_baseline(
        run_dir.name,
        args.rationale,
        path=Path(args.baseline),
        checkpoint_iteration=services.checkpoint_iteration_of(run_dir),
    )
    return {"op": "promote", "baseline": str(args.baseline), **dataclasses.asdict(baseline)}


def render(payload: dict[str, Any]) -> None:
    print(f"Baseline is now {payload['run_id']}")
    if payload["checkpoint_iteration"] is not None:
        print(f"  Checkpoint:  {payload['checkpoint_iteration']:,}")
    print(f"  Rationale:   {payload['rationale']}")
    print(f"  Recorded in: {payload['baseline']}")


COMMAND = Command(
    name="promote",
    help="Make a run the new baseline (closes one turn of the base-fork loop).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
