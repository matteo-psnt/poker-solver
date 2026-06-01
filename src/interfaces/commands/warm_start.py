"""The `warm-start` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.interfaces.commands._base import Command
from src.pipeline import services

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run warm-start`."""
    parser.add_argument("--config", required=True, help="Config stem under config/training/.")
    parser.add_argument(
        "--source", required=True, help="Run directory whose average strategy seeds the prior."
    )
    parser.add_argument("--run", required=True, help="Run id to create. Must not already exist.")
    parser.add_argument(
        "--at",
        type=int,
        default=None,
        help="Source rung to seed from. Board-free quality is U-shaped, so the LAST "
        "rung is usually not the best one -- score the ladder and pick the minimum.",
    )
    parser.add_argument(
        "--effective-iterations",
        type=int,
        default=services.DEFAULT_EFFECTIVE_ITERATIONS,
        dest="effective_iterations",
        help="How much accumulated regret the prior claims, and so how many real "
        "iterations it takes to overrule it. Too small and it evaporates; too large "
        "and the run argues with a strategy from the wrong chance layer. This is the "
        "experiment's independent variable, not a tuned constant.",
    )
    parser.add_argument("--experiment", default=None, help="Experiment id this run is an arm of.")
    parser.add_argument("--arm", default=None, help="Arm within the experiment.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.warm_start_run`."""
    out = services.warm_start_run(
        args.config,
        source_run=Path(args.source),
        run_id=args.run,
        effective_iterations=args.effective_iterations,
        at_iteration=args.at,
        experiment=services.ExperimentTag(experiment_id=args.experiment, arm=args.arm),
    )
    return {"op": "warm-start", **dataclasses.asdict(out)}


def render(payload: dict[str, Any]) -> None:
    print("Warm start written.")
    print(f"  Run ID:      {payload['run_id']}  (under {payload['runs_dir']})")
    print(f"  Seeded from: {payload['source_run_id']}")
    print(f"  Config:      {payload['config_name']}")
    print(
        f"  Prior:       {payload['effective_iterations']:,} effective iterations of claimed regret"
    )
    print(
        f"  Rows seeded: {payload['seeded_rows']:,} / {payload['num_rows']:,} "
        f"({payload['seeded_fraction']:.1%}); the rest start uniform"
    )
    # The average strategy is deliberately NOT seeded, so the reported blueprint
    # is an average over the real game only.
    print("  Average:     starts empty, so every contribution is earned on the real game")
    print(f"  Status:      {payload['status']}")
    print(f"\n  Continue it with: poker-solver-run train-static --run {payload['run_id']} ...")


COMMAND = Command(
    name="warm-start",
    help="Seed a scalar run from another kernel's strategy (regrets encode it; average does not).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
