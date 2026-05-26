"""The `train-vector` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any

from src.interfaces.commands._base import Command
from src.pipeline import services


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run train-vector`."""
    parser.add_argument("--config", required=True, help="Config stem under config/training/.")
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="ABSOLUTE iteration target, so a retried leg converges rather than "
        "training twice. Hundreds, not millions: every row updates every iteration.",
    )
    parser.add_argument(
        "--universe-boards",
        type=int,
        default=2000,
        dest="universe_boards",
        help="Real boards the bucket-transition and showdown matrices are estimated "
        "from. These matrices ARE the chance layer, so this defines the game.",
    )
    parser.add_argument(
        "--universe-seed",
        type=int,
        default=7,
        dest="universe_seed",
        help="Draws the universe. Part of the game's identity: a resume with a "
        "different value is refused rather than silently blending two games.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        dest="checkpoint_every",
        help="Write a rung every N iterations.",
    )
    parser.add_argument(
        "--retain-every",
        type=int,
        default=1,
        dest="retain_every",
        help="Keep every Nth rung addressable by `evaluate --at`. Real-game quality "
        "is U-shaped, so score the ladder and take the minimum rather than the last.",
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--experiment", default=None, help="Experiment id this run is an arm of.")
    parser.add_argument("--arm", default=None, help="Arm within the experiment.")
    parser.add_argument("--parent", default=None, help="Run id this was forked from.")
    parser.add_argument(
        "--run",
        default=None,
        help="Continue an EXISTING run instead of starting one.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.train_vector_blueprint`."""
    out = services.train_vector_blueprint(
        args.config,
        iterations=args.iterations,
        universe_boards=args.universe_boards,
        universe_seed=args.universe_seed,
        checkpoint_every=args.checkpoint_every,
        retain_every=args.retain_every,
        dtype=args.dtype,
        run_id=args.run,
        experiment=services.ExperimentTag(
            experiment_id=args.experiment,
            arm=args.arm,
            parent_run_id=args.parent,
        ),
    )
    return {"op": "train-vector", **dataclasses.asdict(out)}


def render(payload: dict[str, Any]) -> None:
    print("Board-free vector training complete.")
    print(f"  Run ID:      {payload['run_id']}  (under {payload['runs_dir']})")
    print(f"  Config:      {payload['config_name']}")
    print(f"  Iterations:  {payload['iterations']:,}")
    print(
        f"  Universe:    {payload['universe_boards']:,} boards, seed "
        f"{payload['universe_seed']}, {payload['dtype']}"
    )
    print(
        f"  Coverage:    {payload['touched_rows']:,} / {payload['num_rows']:,} rows "
        f"({payload['coverage']:.1%})"
    )
    # Its own game, not the real one: the averaging in the bucket transitions
    # floors real-game exploitability well above this.
    print(f"  Abstract:    {payload['abstract_exploitability']:.4f} exploitability, in-abstraction")
    print(
        f"  Runtime:     {payload['runtime_seconds']:.2f}s "
        f"({payload['seconds_per_iteration']:.2f}s/iteration)"
    )
    print(f"  Status:      {payload['status']}")


COMMAND = Command(
    name="train-vector",
    help="Train the board-free vector kernel, stored as an ordinary static checkpoint.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
