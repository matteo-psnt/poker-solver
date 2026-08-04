"""The `train-static` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any

from src.interfaces.commands._base import (
    Command,
    parse_overrides,
)
from src.pipeline import services


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver train-static`."""
    parser.add_argument("--config", required=True, help="Config stem under config/training/.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. Unlike `train`, this does NOT raise memory: the table "
        "is shared and there are no per-worker key maps, so it is a pure throughput knob.",
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Override the config iteration count."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override system.seed.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Nested config override, `__` as the separator. Repeatable.",
    )
    parser.add_argument("--experiment", default=None, help="Experiment id this run is an arm of.")
    parser.add_argument("--arm", default=None, help="Arm within the experiment.")
    parser.add_argument("--parent", default=None, help="Run id this was forked from.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000_000,
        dest="checkpoint_every",
        help="Checkpoint every N iterations (0 = only at the end). This is the bound "
        "on what a killed run loses, traded against disk and write time: a full table "
        "is written each time, and at 250k the writes were ~17%% of a 30M run's wall "
        "clock and left 120 snapshots on the share.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Continue an EXISTING run instead of starting one. --iterations is an "
        "ABSOLUTE target, so re-running past it is a no-op and a retry converges.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.train_static`."""
    out = services.train_static(
        args.config,
        num_workers=args.workers,
        num_iterations=args.iterations,
        seed=args.seed,
        config_overrides=parse_overrides(args.overrides),
        experiment=services.ExperimentTag(
            experiment_id=args.experiment,
            arm=args.arm,
            parent_run_id=args.parent,
        ),
        checkpoint_every=args.checkpoint_every,
        run_id=args.run,
    )
    return {"op": "train-static", **dataclasses.asdict(out)}


def render(payload: dict[str, Any]) -> None:
    print("Static-tree training complete.")
    print(f"  Run ID:      {payload['run_id']}  (under {payload['runs_dir']})")
    print(f"  Config:      {payload['config_name']}")
    print(f"  Iterations:  {payload['iterations']:,}")
    # Coverage is what only this path can report: the table size is known up
    # front, so "how much of the tree did we actually touch" is answerable.
    print(
        f"  Coverage:    {payload['touched_rows']:,} / {payload['num_rows']:,} rows "
        f"({payload['coverage']:.1%})"
    )
    print(f"  Visits/row:  {payload['mean_visits_per_touched']:.1f} mean, on touched rows")
    print(
        f"  Runtime:     {payload['runtime_seconds']:.2f}s "
        f"({payload['iterations_per_second']:.1f} it/s)"
    )
    if payload["dropped_updates"]:
        print(f"  Dropped:     {payload['dropped_updates']:,} updates")
    print(f"  Status:      {payload['status']}")


COMMAND = Command(
    name="train-static",
    help="Train over the statically-enumerated tree (fixed memory, no key maps).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
