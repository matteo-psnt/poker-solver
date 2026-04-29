"""The `submit` subcommand: queue a training task on the pool.

**One command covers both starting and continuing**, exactly as
``train-static`` does locally: ``--run`` continues an existing run, and
``--to`` is an ABSOLUTE target, so re-submitting past it is a no-op. That is
what makes a Batch retry converge instead of training twice, and it is why
there is no separate ``resume`` -- a fresh path and a continuing path that
diverge are how one of them stops being exercised.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.cloud import dispatch, spec
from src.interfaces.commands._base import Command


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver submit`."""
    parser.add_argument("--config", default="", help="Training config stem (fresh runs).")
    parser.add_argument(
        "--to",
        type=int,
        required=True,
        help="ABSOLUTE iteration target, not an increment. Re-running past it is a no-op.",
    )
    parser.add_argument("--run", default="", help="Continue this existing run id.")
    parser.add_argument("--experiment", default="", help="Experiment id to tag the run with.")
    parser.add_argument("--arm", default="", help="Arm label within the experiment.")
    parser.add_argument("--parent", default="", help="Parent run id, for a fork lineage.")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable (e.g. --set solver__pruning=true).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count on the node. Default is all CPUs; worth setting below the "
        "core count on a big abstraction, since every worker loads its own copy.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=spec.DEFAULT_CHECKPOINT_EVERY,
        help="Checkpoint interval in iterations.",
    )
    parser.add_argument(
        "--timeout",
        default=spec.DEFAULT_TIMEOUT,
        help="Wall-clock ceiling on the TRAINING process. It fires before the task-level "
        "ceiling so the wrapper's publish trap still runs, losing at most one rung.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Stage the tree and queue one training task."""
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=spec.TRAIN,
                config=args.config,
                to=args.to,
                run_id=args.run,
                experiment=args.experiment,
                arm=args.arm,
                parent=args.parent,
                sets=tuple(args.sets),
                workers=args.workers,
                checkpoint_every=args.checkpoint_every,
                timeout=args.timeout,
            )
        ]
    )
    return {"op": "submit", "target_iteration": args.to, **payload}


def render(payload: dict[str, Any]) -> None:
    print(f"Submitted training to {payload['target_iteration']:,} iterations (absolute).")
    dispatch.render_queued(payload)


COMMAND = Command(
    name="submit",
    help="Queue a training task on the pool (--run continues an existing run).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
