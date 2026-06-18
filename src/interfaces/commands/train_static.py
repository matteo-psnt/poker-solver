"""The `train-static` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.interfaces.commands._base import (
    Command,
    parse_overrides,
)
from src.pipeline import services

if TYPE_CHECKING:
    import argparse


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
        default=5_000_000,
        dest="checkpoint_every",
        help="Checkpoint every N iterations (0 = only at the end). This is the bound "
        "on what a killed run loses, traded against write time and a worker respawn "
        "per chunk: at 16 workers a 1M chunk is ~18 s and its checkpoint ~2 s, so "
        "1M paid ~20%% and 5M pays under 5%%.",
    )
    parser.add_argument(
        "--warm-start-from",
        default=None,
        dest="warm_start_from",
        help="Seed a FRESH run from this run's average strategy before training. "
        "Ignored when continuing, so a retried leg cannot lay the prior back over "
        "the progress it already made.",
    )
    parser.add_argument(
        "--warm-start-weight",
        type=int,
        default=services.DEFAULT_EFFECTIVE_ITERATIONS,
        dest="warm_start_weight",
        help="How much accumulated regret the prior CLAIMS, and so how many real "
        "iterations it takes to overrule. The experiment's independent variable.",
    )
    parser.add_argument(
        "--warm-start-at",
        type=int,
        default=None,
        dest="warm_start_at",
        help="Rung of the prior to seed from. Board-free quality is NOT monotone "
        "in iterations, so the last rung is not generally the best -- score the "
        "ladder and name the minimum. Omitted seeds from the manifest's current.",
    )
    parser.add_argument(
        "--warm-start-shape",
        default="flat",
        choices=services.PRIOR_SHAPES,
        dest="warm_start_shape",
        help="How the prior's weight is spread across rows. flat = every row "
        "claims the full weight. confidence = a row claims less the more "
        "indifferent the prior is there, so rows it had no view on stop braking "
        "the solver. Changes resistance, never the strategy a row plays.",
    )
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


class StaticTrainingPayload(services.StaticTrainingOutput):
    """What a training leg achieved. NODE-ONLY: no endpoint serves this."""

    op: Literal["train-static"] = "train-static"


def run(args: argparse.Namespace) -> StaticTrainingPayload:
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
        warm_start_from=Path(args.warm_start_from) if args.warm_start_from else None,
        warm_start_weight=args.warm_start_weight,
        warm_start_at=args.warm_start_at,
        progress_file=Path(args.progress_file) if args.progress_file else None,
        warm_start_shape=args.warm_start_shape,
    )
    return StaticTrainingPayload(**out.model_dump())


def render(payload: StaticTrainingPayload) -> None:
    print("Static-tree training complete.")
    print(f"  Run ID:      {payload.run_id}  (under {payload.runs_dir})")
    print(f"  Config:      {payload.config_name}")
    print(f"  Iterations:  {payload.iterations:,}")
    # Coverage is what only this path can report: the table size is known up
    # front, so "how much of the tree did we actually touch" is answerable.
    print(
        f"  Coverage:    {payload.touched_rows:,} / {payload.num_rows:,} rows "
        f"({payload.coverage:.1%})"
    )
    print(f"  Visits/row:  {payload.mean_visits_per_touched:.1f} mean, on touched rows")
    print(
        f"  Runtime:     {payload.runtime_seconds:.2f}s ({payload.iterations_per_second:.1f} it/s)"
    )
    if payload.dropped_updates:
        print(f"  Dropped:     {payload.dropped_updates:,} updates")
    print(f"  Status:      {payload.status}")


COMMAND = Command(
    name="train-static",
    help="Train over the statically-enumerated tree (fixed memory, no key maps).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
