"""The `submit-coupling` subcommand: price a board-free abstraction on the pool.

Separate from `submit-vector` because it asks a different question. The sweep
compares KERNELS by training them; this trains nothing and compares
ABSTRACTIONS by what their averaged constants throw away. Folding it into the
sweep's kernel-by-abstraction matrix would put a cell in it that no kernel flag
means anything for.

One task per abstraction: the measurement is quadratic in `--boards` and the
whole point is to read the same curve at several bucket counts, which is the
comparison that says whether the error grows with fineness the way board-free's
exploitability does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from src.interfaces.cloud.tasks import dispatch, spec
from src.interfaces.commands._base import Command
from src.interfaces.commands.abstraction_coupling import DEFAULT_CLASSES
from src.shared.cloudtask.kinds import TaskName

if TYPE_CHECKING:
    import argparse

# Minutes, not hours: one streamed pass over the universe and a Gram matrix.
# The ceiling is here to catch a pathological --boards, not to bound normal work.
COUPLING_TIMEOUT = "2h"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver submit-coupling`."""
    parser.add_argument(
        "--abstraction",
        action="append",
        required=True,
        dest="abstractions",
        metavar="NAME",
        help="Abstraction directory on the share. Repeatable — one task each, "
        "which is how the fineness comparison is made.",
    )
    parser.add_argument(
        "--boards",
        type=int,
        default=2000,
        help="Runouts averaged over. Cost is QUADRATIC in this: it is the Gram "
        "matrix's side length, not a loop bound.",
    )
    parser.add_argument(
        "--classes",
        default=DEFAULT_CLASSES,
        help="Comma-separated public-class counts to sweep the dial over.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Universe seed.")
    parser.add_argument(
        "--board-relative",
        action="store_true",
        help="Price the within-board-rank relabelling instead of the artifact's own bucket ids.",
    )
    parser.add_argument("--timeout", default=COUPLING_TIMEOUT, help="Wall-clock ceiling per task.")


def _flags(args: argparse.Namespace) -> tuple[str, ...]:
    """The task's own command line, carried verbatim on ``eval_flags``."""
    flags = [
        "--boards",
        str(args.boards),
        "--classes",
        args.classes,
        "--seed",
        str(args.seed),
    ]
    if args.board_relative:
        flags.append("--board-relative")
    return tuple(flags)


class SubmitCouplingPayload(dispatch.Dispatched):
    """A dispatch, plus which abstractions it priced."""

    op: Literal["submit-coupling"] = "submit-coupling"
    abstractions: list[str] = Field(default_factory=list)
    boards: int = 0


def run(args: argparse.Namespace) -> SubmitCouplingPayload:
    """Stage the tree once and queue one measurement per abstraction."""
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=TaskName.ABSTRACTION_COUPLING,
                config=abstraction,
                eval_flags=_flags(args),
                timeout=args.timeout,
            )
            for abstraction in args.abstractions
        ]
    )
    return payload.extend(
        SubmitCouplingPayload,
        abstractions=list(args.abstractions),
        boards=args.boards,
    )


def render(payload: SubmitCouplingPayload) -> None:
    print(
        f"Queued {len(payload.abstractions)} coupling measurement(s) "
        f"over {payload.boards:,} boards:"
    )
    for abstraction in payload.abstractions:
        print(f"  {abstraction}")
    dispatch.render_queued(payload)


COMMAND = Command(
    name="submit-coupling",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Queue board-free abstraction-cost measurements on the pool, one per abstraction.",
)
