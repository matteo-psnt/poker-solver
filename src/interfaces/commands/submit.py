"""The `submit` subcommand: queue a training task on the pool.

**One command covers both starting and continuing**, exactly as
``train-static`` does locally: ``--run`` continues an existing run, and
``--to`` is an ABSOLUTE target, so re-submitting past it is a no-op. That is
what makes a Batch retry converge instead of training twice, and it is why
there is no separate ``resume`` -- a fresh path and a continuing path that
diverge are how one of them stops being exercised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.interfaces.cloud.tasks import dispatch, spec
from src.interfaces.commands._base import Command
from src.shared import gitinfo
from src.shared.cloudtask.kinds import TaskName

if TYPE_CHECKING:
    import argparse


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
    parser.add_argument(
        "--arm",
        default="",
        help="Arm label within the experiment. Defaults to the branch when "
        "--experiment is given, since that is already the name the work has.",
    )
    parser.add_argument("--parent", default="", help="Parent run id, for a fork lineage.")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable (e.g. --set solver__cfr_plus=true).",
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
        "--kernel",
        choices=("scalar", "board-free"),
        default="scalar",
        help="scalar = external-sampling MCCFR over the real game (train-static). "
        "board-free = the vector kernel, which updates every row every iteration "
        "but solves a bucket-transition approximation of the chance layer.",
    )
    parser.add_argument(
        "--universe-boards",
        type=int,
        default=2000,
        dest="universe_boards",
        help="[board-free] Real boards the bucket-transition matrices are estimated "
        "from. They define the chance layer, so they define the game.",
    )
    parser.add_argument(
        "--universe-seed",
        type=int,
        default=7,
        dest="universe_seed",
        help="[board-free] Draws the universe; part of the game's identity.",
    )
    parser.add_argument(
        "--dtype",
        default="",
        choices=("", "float32", "float64"),
        help="[board-free] Kernel precision. Default float32, measured to cost "
        "nothing in strategy quality.",
    )
    parser.add_argument(
        "--warm-start-from",
        default="",
        dest="warm_start_from",
        help="[scalar] Seed a FRESH run from this run's average strategy. Board-free "
        "reaches a 30M-iteration blueprint's quality in ~100 iterations, so it is "
        "worth a try as a prior even though it is capped as a blueprint.",
    )
    parser.add_argument(
        "--warm-start-weight",
        type=int,
        default=0,
        dest="warm_start_weight",
        help="[scalar] How much regret the prior claims, i.e. how many real "
        "iterations it takes to overrule. The experiment's independent variable.",
    )
    parser.add_argument(
        "--warm-start-at",
        type=int,
        default=0,
        dest="warm_start_at",
        help="[scalar] Rung of the prior to seed from; the last rung is not "
        "generally the best one.",
    )
    parser.add_argument(
        "--warm-start-shape",
        default="",
        dest="warm_start_shape",
        choices=("", "flat", "confidence"),
        help="[scalar] How the prior's weight is spread across rows. flat gives "
        "every row the full weight; confidence damps rows the prior is "
        "indifferent about, so they stop braking the solver.",
    )
    parser.add_argument(
        "--equity-prior",
        type=int,
        default=0,
        dest="equity_prior_weight",
        help="[scalar] Seed a fresh run with a strength-aware opening guess "
        "instead of uniform. Needs no prior run.",
    )
    parser.add_argument(
        "--equity-prior-temperature",
        type=float,
        default=0.0,
        help="[scalar] How sharply the guess commits. High flattens toward "
        "uniform, so a large value is a null arm.",
    )
    parser.add_argument(
        "--equity-prior-fallback",
        type=float,
        default=0.0,
        help="[scalar] Strategy mass the guess claims, which is what makes it "
        "play on rows training never reaches. 0 disables.",
    )
    parser.add_argument(
        "--timeout",
        default=spec.DEFAULT_TIMEOUT,
        help="Wall-clock ceiling on the TRAINING process. It fires before the task-level "
        "ceiling so the wrapper's publish trap still runs, losing at most one rung.",
    )


def _arm(args: argparse.Namespace) -> str:
    """The arm label, defaulting to the branch this was submitted from.

    An arm IS a worktree here — the whole reason for a branch is that it holds
    one idea — so retyping the name as `--arm` is a step that adds nothing and
    can be forgotten. Forgetting it is not a small mistake: an untagged run is
    unaffiliated, `report --experiment` never sees it, and the omission shows up
    as a missing row rather than as an error.

    Only under `--experiment`. Outside one an arm label means nothing, and
    stamping every ordinary run with its branch would turn a field that says
    "this is part of a comparison" into one that is always set.

    An explicit `--arm` always wins, and a detached checkout has no branch to
    fall back on, so both leave this exactly as it was.
    """
    if args.arm or not args.experiment:
        return args.arm
    return gitinfo.get_git_branch() or ""


class SubmitPayload(dispatch.Dispatched):
    """One queued training task, and what it is aiming at."""

    op: Literal["submit"] = "submit"
    """The ABSOLUTE iteration target. `train-static` treats it as such, so
    re-running past it is a no-op -- which is what makes a scheduler retry
    converge instead of training twice."""
    target_iteration: int


def run(args: argparse.Namespace) -> SubmitPayload:
    """Stage the tree and queue one training task."""
    # Imported here, not at module scope, so `--help` does not pay for the
    # Azure SDK -- the same rule `_base.records_root` follows.
    from src.interfaces.cloud.store.workspace import (  # noqa: PLC0415 -- see above
        resolve_published_run,
    )

    # Only a CONTINUE carries a run id, and only then can it be a fragment the
    # node cannot match. A fresh run's id does not exist yet.
    run_id = resolve_published_run(args.run) if args.run else args.run
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=TaskName.TRAIN if args.kernel == "scalar" else TaskName.TRAIN_VECTOR,
                config=args.config,
                to=args.to,
                run_id=run_id,
                experiment=args.experiment,
                arm=_arm(args),
                parent=args.parent,
                sets=tuple(args.sets),
                workers=args.workers,
                checkpoint_every=args.checkpoint_every,
                timeout=args.timeout,
                universe_boards=args.universe_boards if args.kernel == "board-free" else 0,
                universe_seed=args.universe_seed if args.kernel == "board-free" else 0,
                dtype=args.dtype,
                warm_start_from=args.warm_start_from,
                warm_start_weight=args.warm_start_weight,
                warm_start_at=args.warm_start_at,
                warm_start_shape=args.warm_start_shape,
                equity_prior_weight=args.equity_prior_weight,
                equity_prior_temperature=args.equity_prior_temperature,
                equity_prior_fallback=args.equity_prior_fallback,
            )
        ]
    )
    return payload.extend(SubmitPayload, target_iteration=args.to)


def render(payload: SubmitPayload) -> None:
    print(f"Submitted training to {payload.target_iteration:,} iterations (absolute).")
    dispatch.render_queued(payload)


COMMAND = Command(
    name="submit",
    help="Queue a training task on the pool (--run continues an existing run).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
