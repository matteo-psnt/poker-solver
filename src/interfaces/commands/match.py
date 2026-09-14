"""The `match` subcommand: two blueprints, duplicate deals, who wins chips.

The estimator existed and nothing drove it. That mattered, because the metric
this project gated on cannot see the failure it was hiding: exploitability is a
WORST-CASE bound, so it is maximised by refusing to play, and a blueprint that
folds the SB 79.8% scored 722.5 mbb/g against a fold-every-hand floor of 750.
A chip match cannot be won that way -- an opponent who never folds collects the
blinds from a folder every hand.

Read it beside exploitability, never instead of it: one-on-one results can be
INTRANSITIVE (A beats B beats C beats A), which is why the literature pairs the
two rather than swapping one for the other (Johanson et al., AAMAS 2013).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.adapters.postgres import connect
from src.interfaces.commands._base import Command, resolve_run_dir
from src.pipeline.services import record_blueprint_match

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver match`."""
    parser.add_argument("--run", required=True, help="Run A: the arm under test.")
    parser.add_argument("--vs", required=True, help="Run B: the reference it plays.")
    parser.add_argument("--runs-dir", default=None, help="Where runs live on this box.")
    parser.add_argument(
        "--deals",
        type=int,
        default=2000,
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


class MatchPayload(BaseModel):
    """Chip edge of run A's blueprint over run B's, in mbb per hand."""

    op: Literal["match"] = "match"
    run_a: str
    run_b: str
    deals: int
    a_mbb_per_hand: float
    se_mbb: float
    results: dict[str, Any] = {}


def run(args: argparse.Namespace) -> MatchPayload:
    """Play the two blueprints and report A's edge."""
    source = connect.record_source_from_environment()
    document = record_blueprint_match(
        Path(resolve_run_dir(args.run, args.runs_dir)),
        Path(resolve_run_dir(args.vs, args.runs_dir)),
        num_deals=args.deals,
        seed=args.seed,
        record_source=source,
    )
    # The persisted document embeds BOTH runs' provenance; the numbers live
    # under `results` when it is the full document and at the top level when a
    # caller hands back the bare payload.
    results = dict(document.get("results") or document)
    return MatchPayload(
        run_a=str(results.get("run_a", args.run)),
        run_b=str(results.get("run_b", args.vs)),
        deals=args.deals,
        a_mbb_per_hand=float(results["a_mbb_per_hand"]),
        se_mbb=float(results["se_mbb"]),
        results=results,
    )


def render(payload: MatchPayload) -> None:
    edge, se = payload.a_mbb_per_hand, payload.se_mbb
    sigma = abs(edge) / se if se else 0.0
    winner = payload.run_a if edge > 0 else payload.run_b
    print(f"  A: {payload.run_a}")
    print(f"  B: {payload.run_b}")
    print(f"  Edge (A over B): {edge:+.2f} mbb/hand (± {se:.2f})  {sigma:.1f}σ")
    print(f"  Deals:           {payload.deals:,} duplicate (played twice, seats swapped)")
    # Two mediocre strategies can be far apart head-to-head and level on
    # exploitability, which is the whole reason this command exists.
    print(f"  Reading:         {'inconclusive' if sigma < 2 else f'{winner} wins'}")


COMMAND = Command(
    name="match",
    help="Duplicate-deal head-to-head: which blueprint wins chips off the other.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
