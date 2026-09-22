"""The `match` subcommand: two blueprints, duplicate deals, who wins chips.

The metric this project gated on cannot see the failure it was hiding:
exploitability is a WORST-CASE bound, so it is maximised by refusing to play,
and a blueprint that folds the SB 79.8% scored 722.5 mbb/g against a
fold-every-hand floor of 750. A chip match cannot be won that way -- an
opponent who never folds collects the blinds from a folder every hand.

Read it beside exploitability, never instead of it: one-on-one results can be
INTRANSITIVE (A beats B beats C beats A), which is why the literature pairs the
two rather than swapping one for the other (Johanson et al., AAMAS 2013).

This is the LOCAL door, for two runs already on the box. The pool runs the same
estimator through `score --method blueprint_match`, which is where a match of
published runs belongs: the checkpoints are a local mount on a node and a WAN
download from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.adapters.postgres import connect
from src.interfaces.commands._base import Command, resolve_run_dir
from src.interfaces.commands.evaluate import MATCH_DEALS, render
from src.pipeline import services
from src.shared.config import DEFAULT_RUNS_DIR

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver match`."""
    parser.add_argument("--run", required=True, help="Run A: the arm under test.")
    parser.add_argument("--vs", required=True, help="Run B: the reference it plays.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Where runs live on this box.")
    parser.add_argument(
        "--at", type=int, default=None, help="Rung of --run to play (default: its checkpoint)."
    )
    parser.add_argument(
        "--vs-at",
        type=int,
        default=None,
        help="Rung of --vs to play (default: its checkpoint). Two arms are only "
        "comparable at a matched training budget, so pass this whenever one of them "
        "was continued further than the other.",
    )
    parser.add_argument(
        "--deals",
        type=int,
        default=MATCH_DEALS,
        help="Duplicate deals. Each is played TWICE with the seats swapped off one deck, "
        "so card luck cancels in the pair and the skill difference is what is left -- "
        "~2k deals resolve it, against the ~46k an unpaired match would need.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deck seed. Two arms compared on the SAME seed are paired; vary it across "
        "replicates, never within a comparison.",
    )


def run(args: argparse.Namespace) -> services.EvaluationPayload:
    """Play the two blueprints and report A's edge."""
    return services.evaluate_and_record(
        resolve_run_dir(args.run, args.runs_dir),
        method="blueprint_match",
        vs_run_dir=resolve_run_dir(args.vs, args.runs_dir),
        vs_at=args.vs_at,
        at_iteration=args.at,
        match_deals=args.deals,
        match_seed=args.seed,
        sink=connect.eval_sink_from_environment(),
        record_source=connect.record_source_from_environment(),
    )


COMMAND = Command(
    name="match",
    help="Duplicate-deal head-to-head: which blueprint wins chips off the other.",
    add_arguments=add_arguments,
    run=run,
    # The same renderer `evaluate` uses: one estimator, one way of reading it,
    # whichever door it came through.
    render=render,
)
