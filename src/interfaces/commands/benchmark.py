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

**Then the blueprint.** `--agent blueprint --run <id>`. Their game is 200 bb, so
an arm cut for another depth is refused; `--allow-depth-mismatch` fields one
anyway and the result is a translation probe, not a score.

There is deliberately no `--key`: it is the whole credential, it cannot be
rotated from here, and a shell history is a bad place for it. Supply it through
``$GTOWIZARD_API_KEY``, never a file in this tree -- a dispatch seals the
working tree and would carry it to every node.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command, resolve_run_dir
from src.interfaces.errors import CommandError
from src.interfaces.gtowizard import agents, session
from src.interfaces.gtowizard.client import (
    DEFAULT_VERSION,
    BenchmarkClient,
    key_from_environment,
)
from src.interfaces.gtowizard.protocol import GAME_NAME
from src.shared.cloudtask.node.paths import NodePaths

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
        default=str(NodePaths.from_environment().runs),
        help="Where runs live on this box; the node's own runs directory by default.",
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
    # the expensive half of start-up, and `--help` and the keyless
    # baseline probe -- which needs no blueprint at all -- must not pay it.
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
    #: Our own bets their raise range moved. A size the strategy did not choose.
    clamped_per_hand: float = 0.0
    truncated_hands: int = 0
    expected: str = ""
    within_expectation: bool | None = None
    #: |ours - published| in combined standard errors. The NUMBER matters more
    #: than the verdict: the band's width is set by their sample as much as
    #: ours, and theirs is only 10,001 hands for check-call.
    z_score: float | None = None


# Below this an expectation check cannot fail honestly: the standard error is
# estimated off the same few samples, so a run that measured almost nothing
# reports agreement. MEASURED: an empty tally has an INFINITE standard error,
# which drives the z-score to zero -- 2,000 hands that all 409'd printed
# "matches the published check-call at 0.0σ".
MIN_EXPECT_HANDS = 200


def _check_expectation(
    args: argparse.Namespace, tally: session.Tally
) -> tuple[bool | None, float | None]:
    """How far the probe sits from the published figure, or why it cannot say.

    Separate from `run` so both refusals are testable without a live server.
    """
    if not args.expect:
        return None, None
    within: bool | None = None
    z_score: float | None = None
    published, published_se = agents.PUBLISHED[args.expect]
    # A probe that measured nothing must not report agreement. With no
    # hands the standard error is infinite, so the z-score is 0.0 and the
    # run PASSES: 2,000 hands that all failed on a 409 printed "matches the
    # published check-call at 0.0σ". The whole point of this gate is to
    # catch a silent failure, so it cannot have one of its own.
    if tally.played < MIN_EXPECT_HANDS:
        raise CommandError(
            f"The {args.expect} probe played {tally.played} hands; an expectation "
            f"check needs at least {MIN_EXPECT_HANDS}. Below that the standard error "
            "is itself an estimate off a handful of samples, and a few hands that "
            "happen to agree make the band arbitrarily tight. If nothing played at "
            "all, note their cap is 20 hands in progress at once -- abandoned hands "
            "hold slots until they finish."
        )
    combined = (published_se**2 + tally.aivat_std_bb_per_100**2) ** 0.5
    # And it must be able to FAIL. A handful of hands makes the band so wide
    # that a doubled or halved score sits inside it -- agreement then says
    # only that the sample was small. Require the band to be narrower than
    # half the published figure, which is the size of the scale errors this
    # is here to find.
    band = args.sigma * combined
    if not band < abs(published) / 2:
        raise CommandError(
            f"The {args.expect} probe is too noisy to check anything: ±{band:.1f} "
            f"at {args.sigma:g}σ over {tally.played} hands, against a published "
            f"{published:.2f}. A doubled or halved score would sit inside that "
            "band. Run more hands."
        )
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
    return within, z_score


def run(args: argparse.Namespace) -> BenchmarkPayload:
    """Play hands against GTO Wizard AI and report the score."""
    key = key_from_environment()
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

    within, z_score = _check_expectation(args, tally)

    return BenchmarkPayload(
        agent=args.agent,
        game=args.game,
        hands_played=tally.played,
        hands_failed=tally.failed,
        aivat_bb_per_100=tally.aivat_bb_per_100,
        aivat_std_bb_per_100=tally.aivat_std_bb_per_100,
        raw_bb_per_100=tally.raw_bb_per_100,
        off_tree_per_hand=tally.off_tree_rate,
        clamped_per_hand=tally.clamp_rate,
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
    if payload.clamped_per_hand:
        # A size the strategy did not choose. Reported beside off-tree because
        # it is the same kind of drift, pointing the other way: off-tree is
        # THEIR action snapped onto our menu, a clamp is OUR size moved onto
        # theirs. A run carrying many is not measuring what was fielded.
        print(f"  clamped         {payload.clamped_per_hand:.2f} of our bets/hand resized")
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
