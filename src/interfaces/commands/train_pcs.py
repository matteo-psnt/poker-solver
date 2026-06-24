"""The `train-pcs` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.interfaces.commands._base import Command, parse_overrides
from src.pipeline import services

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver train-pcs`."""
    parser.add_argument("--config", required=True, help="Config stem under config/training/.")
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="ABSOLUTE target in sampled boards, so a retried leg converges rather "
        "than training twice. Thousands, not millions: one board updates every "
        "live hand at every node it reaches.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes, each sampling its own boards into the shared table. "
        "A CEILING: the hand-space scratch is ~3 GB per worker and private, so the "
        "count is clamped to what the node's RAM holds. Default: the CPU count.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override system.seed.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        dest="checkpoint_every",
        help="Write a rung every N iterations; the bound on what a killed run loses.",
    )
    parser.add_argument(
        "--retain-every",
        type=int,
        default=0,
        dest="retain_every",
        help="Keep one rung per N iterations addressable by `evaluate --at`. 0 keeps "
        "every rung: a sampling trainer's best point is found by scoring the ladder.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Nested config override, `__` as the separator. Repeatable. The sampler's "
        "own knobs live under `pcs__` (alternating, runouts_per_flop, showdown).",
    )
    parser.add_argument("--experiment", default=None, help="Experiment id this run is an arm of.")
    parser.add_argument("--arm", default=None, help="Arm within the experiment.")
    parser.add_argument("--parent", default=None, help="Run id this was forked from.")
    parser.add_argument(
        "--run",
        default=None,
        help="Continue an EXISTING run instead of starting one. --iterations is an "
        "ABSOLUTE target, so re-running past it is a no-op and a retry converges.",
    )
    parser.add_argument(
        "--progress-file",
        default="",
        dest="progress_file",
        help="Write {done,total} iterations here while training. Node-local: a "
        "heartbeat for the task bar between checkpoints, not a record.",
    )


class PcsTrainingPayload(services.PcsTrainingOutput):
    """What a public-chance-sampling leg achieved. NODE-ONLY: no endpoint serves this."""

    op: Literal["train-pcs"] = "train-pcs"


def run(args: argparse.Namespace) -> PcsTrainingPayload:
    """Argparse transport around :func:`services.train_pcs`."""
    out = services.train_pcs(
        args.config,
        iterations=args.iterations,
        num_workers=args.workers,
        seed=args.seed,
        config_overrides=parse_overrides(args.overrides),
        experiment=services.ExperimentTag(
            experiment_id=args.experiment,
            arm=args.arm,
            parent_run_id=args.parent,
        ),
        checkpoint_every=args.checkpoint_every,
        retain_every=args.retain_every,
        run_id=args.run,
        progress_file=Path(args.progress_file) if args.progress_file else None,
    )
    return PcsTrainingPayload(**out.model_dump())


def render(payload: PcsTrainingPayload) -> None:
    print("Public-chance-sampling training complete.")
    print(f"  Run ID:      {payload.run_id}  (under {payload.runs_dir})")
    print(f"  Config:      {payload.config_name}")
    print(f"  Iterations:  {payload.iterations:,} boards ({payload.board_passes:,} passes)")
    print(f"  Workers:     {payload.workers}")
    print(
        f"  Coverage:    {payload.touched_rows:,} / {payload.num_rows:,} rows "
        f"({payload.coverage:.1%})"
    )
    print(
        f"  Runtime:     {payload.runtime_seconds:.2f}s "
        f"({payload.iterations_per_second:.3f} boards/s)"
    )
    print(f"  Status:      {payload.status}")


COMMAND = Command(
    name="train-pcs",
    help="Train by public chance sampling: one board per iteration, every hand at once.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
