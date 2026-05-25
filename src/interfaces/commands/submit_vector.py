"""The `submit-vector` subcommand: measure CFR kernels on the pool.

The comparison is a matrix -- kernels x abstractions x derivation sizes -- and
every cell is independent, so they queue as separate tasks and the pool runs as
many at once as it has nodes. Sequentially the work is the sum of the arms; here
it is the slowest one.

Why a node rather than a laptop: the hand-space arm pins a core for tens of
minutes and a laptop throttles under it. A 1,000-iteration run measured that way
drifted 4.1 s to 9.7 s per iteration on heat alone -- which leaves the
exploitability numbers intact, since they are deterministic, and makes every
wall-clock figure beside them meaningless.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.engine.solver.vector import BOARD_FREE, HAND_SPACE, KERNELS, SCALAR
from src.interfaces.cloud import dispatch, spec
from src.interfaces.commands._base import Command
from src.interfaces.commands.vector_sweep import DEFAULT_SCORE_BOARDS
from src.shared.cloudtask.kinds import TaskName

# Longer than a training task's default: the hand-space arm is one full tree
# pass per board per iteration, and it is the arm worth waiting for.
VECTOR_TIMEOUT = "8h"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver submit-vector`."""
    parser.add_argument(
        "--abstraction",
        action="append",
        required=True,
        dest="abstractions",
        metavar="NAME",
        help="Abstraction directory on the share. Repeatable -- one arm per "
        "abstraction per kernel.",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        dest="kernels",
        choices=list(KERNELS),
        help=f"Repeatable. Default: {BOARD_FREE} and {HAND_SPACE}; pass {SCALAR} "
        "to include the shipped kernel.",
    )
    parser.add_argument(
        "--derive-boards",
        type=int,
        action="append",
        dest="derive_boards",
        metavar="N",
        help="Boards the board-free matrices average over. Repeatable, to sweep "
        "it. Ignored by the other kernels, which derive nothing. Default: 6000.",
    )
    parser.add_argument(
        "--train-boards",
        type=int,
        default=0,
        help="Boards hand-space and scalar TRAIN on, drawn from a different "
        "stream than the scoring boards. 0 trains on the scoring boards, which "
        "grades those two on their own training set. Board-free is unaffected.",
    )
    parser.add_argument(
        "--score-boards",
        type=int,
        default=DEFAULT_SCORE_BOARDS,
        help="Boards the exact best response scores against. Genuinely held out "
        "only when --train-boards is set.",
    )
    parser.add_argument(
        "--checkpoints",
        default="",
        help="Comma-separated iteration counts, applied to EVERY arm. Leave "
        "unset unless you mean that: the kernels differ by orders of magnitude "
        "in cost per iteration, so one list sized for one asks days of another.",
    )
    parser.add_argument("--config", default="", help="Config stem for the action model.")
    parser.add_argument(
        "--stack",
        type=int,
        default=20,
        help="Starting stack. Smaller trees keep the hand-space arm affordable; "
        "the comparison is between kernels on ONE tree, so this must not vary "
        "across the arms being compared.",
    )
    parser.add_argument("--timeout", default=VECTOR_TIMEOUT, help="Wall-clock ceiling per arm.")


def _arms(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """Every (abstraction, kernel, derive_boards) cell this submission covers.

    Only the board-free kernel derives anything, so the others get exactly one
    arm per abstraction however many derivation sizes are swept -- otherwise a
    ``--derive-boards`` sweep would silently queue identical duplicate work and
    bill a node-hour for each copy.
    """
    kernels = args.kernels or [BOARD_FREE, HAND_SPACE]
    sizes = args.derive_boards or [6000]
    arms: list[tuple[str, str, int]] = []
    for abstraction in args.abstractions:
        for kernel in kernels:
            if kernel == BOARD_FREE:
                arms.extend((abstraction, kernel, size) for size in sizes)
            else:
                arms.append((abstraction, kernel, 0))
    return arms


def _flags(args: argparse.Namespace, kernel: str, derive: int) -> tuple[str, ...]:
    """The arm's own command line, carried verbatim to the node.

    These ride on ``eval_flags`` rather than on new TaskSpec fields: it is
    already the pass-through for "extra flags this task needs", and six new
    fields would widen a wire three other kinds have to keep threading.
    """
    flags = [
        "--kernel",
        kernel,
        "--score-boards",
        str(args.score_boards),
        "--stack",
        str(args.stack),
    ]
    if args.train_boards:
        flags += ["--train-boards", str(args.train_boards)]
    if kernel == BOARD_FREE:
        flags += ["--derive-boards", str(derive)]
    if args.checkpoints:
        flags += ["--checkpoints", args.checkpoints]
    if args.config:
        flags += ["--config", args.config]
    return tuple(flags)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Stage the tree once and queue one task per arm."""
    arms = _arms(args)
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=TaskName.VECTOR_SWEEP,
                config=abstraction,
                arm=kernel,
                eval_flags=_flags(args, kernel, derive),
                timeout=args.timeout,
            )
            for abstraction, kernel, derive in arms
        ]
    )
    payload["op"] = "submit-vector"
    payload["arms"] = [{"abstraction": a, "kernel": k, "derive_boards": d} for a, k, d in arms]
    return payload


def render(payload: dict[str, Any]) -> None:
    print(f"Queued {len(payload['arms'])} vector-sweep arm(s):")
    for arm in payload["arms"]:
        derive = f", {arm['derive_boards']:,} boards" if arm["derive_boards"] else ""
        print(f"  {arm['kernel']:<11} {arm['abstraction']}{derive}")
    dispatch.render_queued(payload)


COMMAND = Command(
    name="submit-vector",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Queue CFR kernel measurements on the pool, one task per arm.",
)
