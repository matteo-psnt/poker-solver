"""The `benchmark` subcommand: play hands against GTO Wizard AI and score them.

Two uses, and the baseline one is the gate.

**Probe first, and probe BOTH.** GTO Wizard post their own trivial agents to the
public board, so two agents here have a published right answer. Run them both::

    benchmark --agent always-fold --hands 2000 --expect always-fold
    benchmark --agent check-call  --hands 2000 --expect check-call

One is not enough. Their check-call figure is -184.19 +/- **7.96** -- they only
ran 10,001 hands -- so at 2,000 of ours the combined 3-sigma band is ~28 bb/100
wide, and a wire bug that mis-sized every bet could hide inside it. Always-fold
is published at -63.18 +/- **1.16**, a 7x tighter reference, but folding
exercises almost none of the wire. Together they pin both halves: always-fold
pins the blinds, the seats and the hand accounting against a sharp number;
check-call pins the multi-street path. `--expect` reports the z-score and
refuses past 3, so the strength of the check is legible rather than a boolean.

**Then the blueprint.** `--agent blueprint --run <id>`. Their game is 200 bb and
every blueprint we have is cut for 100, so this refuses until a 200 bb arm
exists; `--allow-depth-mismatch` fields one anyway and the result is a
translation probe, not a score.

There is deliberately no `--key`: it is the whole credential, it cannot be
rotated from here, and a shell history is a bad place for it. Supply it through
``$GTOWIZARD_API_KEY``, never a file in this tree -- a dispatch seals the
working tree and would upload it to the share.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command, resolve_run_dir
from src.interfaces.errors import CommandError
from src.interfaces.gtowizard import agents, session
from src.interfaces.gtowizard.client import DEFAULT_VERSION, KEY_ENV, BenchmarkClient
from src.interfaces.gtowizard.protocol import GAME_NAME

if TYPE_CHECKING:
    import argparse

AGENT_CHOICES = (*agents.BASELINES, "blueprint")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver benchmark`."""
    parser.add_argument(
        "--agent",
        default="check-call",
        choices=AGENT_CHOICES,
        help="Who plays. The default is the PROBE: its score is published, so it "
        "tests the wire rather than the strategy.",
    )
    parser.add_argument("--hands", type=int, default=2000, help="Hands to play.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="How many combined standard errors from the published figure `--expect` "
        "tolerates (default 3).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=session.DEFAULT_CONCURRENT,
        help=f"Hands in flight at once (their cap is {session.MAX_CONCURRENT}; they "
        "recommend fewer so one stuck hand does not stall the run).",
    )
    parser.add_argument(
        "--expect",
        default="",
        choices=("", *agents.PUBLISHED),
        help="Fail unless the score lands within 3 standard errors of this "
        "baseline's published figure. This is the probe's pass/fail.",
    )
    parser.add_argument("--run", default=None, help="Run id, fragment, or path (blueprint only).")
    parser.add_argument(
        "--runs-dir",
        default="/mnt/work/runs",
        help="Where runs live on this box. Local disk, never the share.",
    )
    parser.add_argument(
        "--at", type=int, default=None, help="Seat the checkpoint at this iteration."
    )
    parser.add_argument(
        "--no-resolver",
        action="store_true",
        help="Play the bare blueprint. Otherwise follows `resolver.enabled`.",
    )
    parser.add_argument(
        "--allow-depth-mismatch",
        action="store_true",
        help="Field a blueprint cut for a different depth than their 200 bb table. "
        "The result is a translation probe, not a score.",
    )
    parser.add_argument(
        "--log", default=None, help="Append one JSON object per hand here as it lands."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for `--agent random`.")
    parser.add_argument("--game", default=GAME_NAME, help="Game name.")


def _agent(args: argparse.Namespace) -> agents.Agent:
    if args.agent != "blueprint":
        return agents.baseline(args.agent, args.seed)
    if not args.run:
        raise CommandError("--agent blueprint needs --run.")
    # Imported here, not at module scope: the engine and its numba kernels are
    # the expensive half of start-up, and `--help` and the baseline probe -- the
    # thing this command is FOR until a 200 bb arm exists -- must not pay it.
    from src.adapters.postgres import connect  # noqa: PLC0415 -- see above
    from src.interfaces.gtowizard.player import BlueprintPlayer  # noqa: PLC0415
    from src.pipeline.services.scoring._shared import build_blueprint_for  # noqa: PLC0415
    from src.pipeline.training.run_tracker import RunTracker  # noqa: PLC0415

    run_dir = resolve_run_dir(args.run, args.runs_dir)
    metadata = RunTracker.load(run_dir, connect.record_source_from_environment()).metadata
    blueprint, _storage, _policy = build_blueprint_for(
        run_dir,
        metadata,
        abstraction_hash=metadata.card_abstraction_hash,
        at_iteration=args.at,
    )
    return BlueprintPlayer(
        blueprint,
        use_resolver=False if args.no_resolver else None,
        allow_depth_mismatch=args.allow_depth_mismatch,
    )


class BenchmarkPayload(BaseModel):
    """One run against GTO Wizard AI, and the score it earned.

    ``aivat_bb_per_100`` is the metric; ``raw_bb_per_100`` is the unadjusted
    chip result and is shown only so the gap between them stays visible.
    """

    op: Literal["benchmark"] = "benchmark"
    agent: str
    game: str
    version: int = DEFAULT_VERSION
    hands_played: int
    hands_failed: int
    aivat_bb_per_100: float
    aivat_std_bb_per_100: float
    raw_bb_per_100: float
    off_tree_per_hand: float = 0.0
    truncated_hands: int = 0
    expected: str = ""
    within_expectation: bool | None = None
    #: |ours - published| in combined standard errors. The NUMBER matters more
    #: than the verdict: the band's width is set by their sample as much as
    #: ours, and theirs is only 10,001 hands for check-call.
    z_score: float | None = None


def run(args: argparse.Namespace) -> BenchmarkPayload:
    """Play hands against GTO Wizard AI and report the score."""
    key = os.environ.get(KEY_ENV)
    if not key:
        raise CommandError(
            f"No API key. Set ${KEY_ENV} to the key GTO Wizard approved "
            "(request one at https://benchmark.gtowizard.com/)."
        )
    agent = _agent(args)
    log_path = Path(args.log) if args.log else None
    with BenchmarkClient(key) as client:
        tally = session.run(
            client,
            agent,
            num_hands=args.hands,
            concurrency=args.concurrency,
            log_path=log_path,
        )

    within: bool | None = None
    z_score: float | None = None
    if args.expect:
        published, published_se = agents.PUBLISHED[args.expect]
        combined = (published_se**2 + tally.aivat_std_bb_per_100**2) ** 0.5
        z_score = abs(tally.aivat_bb_per_100 - published) / combined
        within = z_score <= args.sigma
        if not within:
            # A refusal, not a traceback: the run happened and every hand is in
            # `--log`; what failed is the claim that this reproduces a published
            # figure. Raised HERE rather than in the renderer, which is pure
            # formatting and must stay that way.
            raise CommandError(
                f"The {args.expect} probe scored {tally.aivat_bb_per_100:.2f} "
                f"± {tally.aivat_std_bb_per_100:.2f} over {tally.played} hands, "
                f"against a published {published:.2f} ± {published_se:.2f} — "
                f"{z_score:.1f} combined standard errors out. The wire encoding, the "
                "cumulative-bet convention or the hand loop is wrong. Fix that before "
                "reading any blueprint score."
            )

    return BenchmarkPayload(
        agent=args.agent,
        game=args.game,
        hands_played=tally.played,
        hands_failed=tally.failed,
        aivat_bb_per_100=tally.aivat_bb_per_100,
        aivat_std_bb_per_100=tally.aivat_std_bb_per_100,
        raw_bb_per_100=tally.raw_bb_per_100,
        off_tree_per_hand=tally.off_tree_rate,
        truncated_hands=tally.truncated_hands,
        expected=args.expect,
        within_expectation=within,
        z_score=z_score,
    )


def render(payload: BenchmarkPayload) -> None:
    print(f"{payload.agent} vs GTO Wizard AI v{payload.version} — {payload.game}")
    print(f"  hands           {payload.hands_played} played, {payload.hands_failed} failed")
    print(f"  AIVAT bb/100    {payload.aivat_bb_per_100:.2f} ± {payload.aivat_std_bb_per_100:.2f}")
    print(f"  raw bb/100      {payload.raw_bb_per_100:.2f}  (luck, not skill — not the score)")
    if payload.off_tree_per_hand or payload.truncated_hands:
        print(
            f"  off-tree        {payload.off_tree_per_hand:.2f} snapped actions/hand, "
            f"{payload.truncated_hands} truncated replays"
        )
    if payload.expected and payload.z_score is not None:
        published, published_se = agents.PUBLISHED[payload.expected]
        print(
            f"  matches the published {payload.expected} "
            f"({published:.2f} ± {published_se:.2f}) at {payload.z_score:.1f}σ"
        )


COMMAND = Command(
    name="benchmark",
    help="Play hands against GTO Wizard AI and score them in AIVAT bb/100.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
