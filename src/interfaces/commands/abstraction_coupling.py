"""The `abstraction-coupling` subcommand: price the board-free game's averaging.

Board-free is the only kernel that could reach a converged blueprint —
hand-space is 425x too expensive — and it is capped by abstraction error that
gets WORSE as the abstraction gets finer. `bucket_game`'s docstring names the
two suspects: averaged card removal, and per-player transitions that drop the
correlation a shared board induces. This command sizes both, separately, so a
kernel change targets the one that is actually large.

Nothing here trains. The quantities are properties of the abstraction, so they
are array reductions over the same universe `derive` streams — which is what
makes this affordable enough to run before committing to a kernel.

It runs on a node because the fine abstraction is where the answer matters and
a 600-bucket river needs thousands of boards to populate; the universe is
streamed for the same reason `vector-sweep` streams it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from src.core.game.state import Street
from src.engine.solver.vector import coupling
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.pipeline.abstraction.postflop.bucketer import DenseBucketer
from src.pipeline.abstraction.vector_universe import iter_universe

if TYPE_CHECKING:
    import argparse

# The dial the report is about. Stops at 256 because the kernel state that would
# have to be conditioned is ~2.24 GB per class at production shapes, so a
# 64 GB box tops out near 20 -- past that the curve is answering a question no
# affordable kernel can act on, and is here only to show where saturation lies.
DEFAULT_CLASSES = "1,2,4,8,16,32,64,128,256"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run abstraction-coupling`."""
    parser.add_argument(
        "--abstraction",
        required=True,
        help="Abstraction directory name, e.g. buckets-F100T300R600-rexact-a1542e88.",
    )
    parser.add_argument(
        "--abstractions-dir",
        default="data/combo_abstraction",
        help="Where abstraction directories live.",
    )
    parser.add_argument(
        "--boards",
        type=int,
        default=2000,
        help="Runouts the measurement averages over. This is the Gram matrix's "
        "side length, so cost is quadratic in it, not linear.",
    )
    parser.add_argument(
        "--classes",
        default=DEFAULT_CLASSES,
        help="Comma-separated class counts to sweep the conditioning dial over.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Universe seed.")
    parser.add_argument(
        "--progress-file",
        default="",
        help="Write the result as soon as it exists, so a killed task keeps it.",
    )


class ConstantGap(BaseModel):
    """One averaged constant of the board-free game, priced."""

    name: str
    kind: str
    """Error as a fraction of the constant's own norm. Zero means the board
    carries no information and averaging it away costs nothing."""
    relative: float
    recovered: dict[int, float]


class AbstractionCouplingPayload(BaseModel):
    """What board-free's averaging costs, and what conditioning would buy back.

    NODE-ONLY, like `vector-sweep`: the fine abstraction lives on the share.
    """

    op: Literal["abstraction-coupling"] = "abstraction-coupling"
    abstraction: str
    buckets: dict[str, int] = Field(default_factory=dict)
    boards: int
    seed: int
    accumulate_seconds: float
    measure_seconds: float
    gaps: list[ConstantGap] = Field(default_factory=list)


def run(args: argparse.Namespace) -> AbstractionCouplingPayload:
    """Measure every averaged constant in one streamed pass over the universe."""
    path = Path(args.abstractions_dir) / args.abstraction
    if not path.is_dir():
        raise CommandError(f"No such abstraction: {path}")
    abstraction = DenseBucketer.load(path)

    counts = {
        street: abstraction.num_buckets(street)
        for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
    }
    classes = [int(part) for part in args.classes.split(",") if part.strip()]
    if not classes:
        raise CommandError("--classes must name at least one class count.")
    if max(classes) > args.boards:
        raise CommandError(
            f"--classes asks for {max(classes)} classes from {args.boards} boards; "
            "a class per board already recovers everything by construction."
        )

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    stacked = coupling.accumulate(iter_universe(abstraction, args.boards, rng=rng), counts)
    accumulated = time.perf_counter()

    gaps = [
        coupling.measure(
            name,
            "coupling" if name.startswith("transition") else "dispersion",
            matrix,
            classes,
            seed=args.seed,
        )
        for name, matrix in stacked.items()
    ]
    measured = time.perf_counter()

    payload = AbstractionCouplingPayload(
        abstraction=args.abstraction,
        buckets={street.name.lower(): count for street, count in counts.items()},
        boards=args.boards,
        seed=args.seed,
        accumulate_seconds=round(accumulated - started, 1),
        measure_seconds=round(measured - accumulated, 1),
        gaps=[
            ConstantGap(
                name=gap.name, kind=gap.kind, relative=gap.relative, recovered=gap.recovered
            )
            for gap in gaps
        ],
    )
    if args.progress_file:
        Path(args.progress_file).write_text(payload.model_dump_json(indent=2))
    return payload


def render(payload: AbstractionCouplingPayload) -> None:
    buckets = payload.buckets
    print(
        f"abstraction-coupling on {payload.abstraction} "
        f"(F{buckets.get('flop')}/T{buckets.get('turn')}/R{buckets.get('river')}, "
        f"{payload.boards:,} boards, seed {payload.seed})"
    )
    print(f"  accumulated in {payload.accumulate_seconds}s, measured in {payload.measure_seconds}s")
    if not payload.gaps:
        return

    classes = sorted({count for gap in payload.gaps for count in gap.recovered})
    header = "".join(f"{count:>8}" for count in classes)
    print(f"\n  {'':<28} {'':>8}   {'recovered by C public classes':<{len(header)}}")
    print(f"  {'constant':<28} {'error':>8}   {header}")
    for gap in payload.gaps:
        recovered = "".join(f"{gap.recovered.get(count, 0.0):>8.3f}" for count in classes)
        print(f"  {gap.name:<28} {gap.relative:>8.4f}   {recovered}")
    print(
        "\n  error is relative to the constant's own norm; recovered is the "
        "fraction\n  of squared error a public-class partition closes. C=1 is "
        "the shipped game."
    )


COMMAND = Command(
    name="abstraction-coupling",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="What board-free's board averaging costs, and what conditioning would buy back.",
)
