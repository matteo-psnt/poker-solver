"""The `evaluate` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.adapters.postgres import connect
from src.interfaces.commands._base import (
    Command,
    CommandError,
    resolve_run_dir,
)
from src.pipeline import services
from src.pipeline.evaluation.estimators.elimination import SELECTION_RULES
from src.pipeline.evaluation.estimators.lbr.config import LBRConfig
from src.pipeline.evaluation.estimators.public_tree_br import PublicBRConfig
from src.shared.cloudtask.rung_spec import parse_ladder
from src.shared.config import DEFAULT_RUNS_DIR

if TYPE_CHECKING:
    import argparse

# The estimators a node can actually run. `score` imports this rather than
# repeating it: a value the submitter accepts but `evaluate` rejects is not
# caught until the node has already been allocated, and the task then retries
# twice on the way to failing.
EVAL_METHODS = (
    "lbr",
    "exact_br",
    "resolver_match",
    "blueprint_match",
    "depth_duel",
    "elimination",
    "settle",
)

# Above this the SB is refusing to play rather than playing badly, and no
# exploitability reading about it means anything. `.claude/rules/
# solver-and-evaluation.md` carries the same line for a human reader.
SB_FOLD_REFUSES = 0.50

# `--deals` serves two estimators whose variance is nothing alike, so neither
# default can be argparse's.
RESOLVER_GATE_DEALS = 1000
MATCH_DEALS = 2000


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver evaluate`."""
    parser.add_argument("--run", required=True, help="Run id (dir name) or path to a run dir.")
    parser.add_argument(
        "--runs-dir", default=DEFAULT_RUNS_DIR, help="Base runs dir for id resolution."
    )
    parser.add_argument(
        "--at",
        type=int,
        default=None,
        help="Score a RETAINED checkpoint at this iteration instead of the run's latest "
        "(requires storage.checkpoint_retain_every at train time). Repeat over the ladder "
        "under one fixed config to build a within-run convergence curve.",
    )
    parser.add_argument(
        "--method",
        choices=EVAL_METHODS,
        default="lbr",
        help="lbr = Local Best Response (trustworthy, default); exact_br = deterministic "
        "exact BR on a sampled public tree (zero eval variance; compare within a "
        "matched board tier); resolver_match = duplicate-deal chip edge of "
        "blueprint+resolver over the bare blueprint (NOT an exploitability bound); "
        "blueprint_match = duplicate-deal chip edge over ANOTHER run (--vs-run), the "
        "one gate folding cannot win; depth_duel = the same against a run cut for "
        "ANOTHER depth, both played at --table-bb; elimination = whole escalating "
        "matches, --ladder-a against --ladder-b, scored as a match win rate; "
        "settle = per-street drift of the average "
        "strategy since --settle-from, which says whether training has stopped "
        "changing anything (NOT whether what it settled on is good).",
    )
    # resolver_match options (--method resolver_match).
    parser.add_argument(
        "--deals",
        type=int,
        default=None,
        help="[resolver_match, blueprint_match, depth_duel] Duplicate deals (each played twice, "
        "seats swapped). Defaults differ because the estimators do: 1,000 for the "
        "resolver gate, whose two sides differ only where the resolver fires, and "
        "2,000 for a match of two different blueprints, which disagree far more often.",
    )
    parser.add_argument(
        "--leaf-continuation",
        type=float,
        default=None,
        help="[resolver_match] Override resolver.leaf_continuation_fraction: what each "
        "player is assumed to commit before showdown at a depth-limit leaf, as a "
        "fraction of the pot. 0 is the shipped check-down. A SENSITIVITY knob -- read "
        "the magnitude of any change, never its sign.",
    )
    parser.add_argument(
        "--resolver-leaf-rollouts",
        type=int,
        default=None,
        help="[resolver_match] Override resolver.leaf_rollouts. Leaf valuation is "
        "most of a solve and loops once per sampled runout, so this is the exchange "
        "rate between leaf accuracy and DEPTH at a fixed budget: measured after the "
        "compiled showdown kernels, 13.1 ms/iteration at 8 rollouts against 4.1 at 1, "
        "i.e. 22 iterations per 300 ms versus 72.",
    )
    parser.add_argument(
        "--resolver-min-street",
        choices=["preflop", "flop", "turn", "river"],
        default=None,
        help="[blueprint_match] The earliest street a deployed side A's resolver may "
        "act on; before it the blueprint answers untouched. `river` is the only arm "
        "that tests SEARCH rather than the leaf guess -- MEASURED at max_depth 6, a "
        "river subgame has 53 leaves and ZERO truncated, so it is the whole "
        "remaining game solved exactly, where flop and turn leave 13 of 53 to an "
        "assumption that is identical at depth 6 and depth 10.",
    )
    parser.add_argument(
        "--resolver-prior-weight",
        type=float,
        default=None,
        help="[resolver_match, lbr] Override resolver.root_prior_weight: the blueprint as a "
        "pseudo-count on the root strategy, in units of CFR iterations. 0 is the "
        "shipped behaviour, where a starved solve walks away from UNIFORM rather "
        "than from the blueprint.",
    )
    parser.add_argument(
        "--resolver-blend-alpha",
        type=float,
        default=None,
        help="[lbr] Override resolver.policy_blend_alpha: the weight the DEPLOYED row "
        "puts on the resolver, the rest going to the blueprint. 0 makes deployed play "
        "exactly the blueprint, which is the control that attributes any gap to the "
        "resolver row rather than to the lookup path around it.",
    )
    parser.add_argument(
        "--resolver-max-iterations",
        type=int,
        default=None,
        help="[resolver_match] Pin subgame-CFR iterations instead of the wall-clock "
        "budget. REQUIRED for a valid A/B: time-budgeted arms differ by how fast the "
        "box was, not only by the knob under test.",
    )
    parser.add_argument(
        "--resolver-allin-runouts",
        type=int,
        default=1,
        help="[resolver_match] Average an all-in board over this many completions "
        "instead of the single dealt one. SAME expectation, less variance -- and "
        "exact (enumerated, zero variance) when few enough cards remain. 1 = the "
        "shipped single-board behaviour.",
    )
    # settle options (--method settle).
    parser.add_argument(
        "--settle-from",
        type=int,
        default=None,
        help="[settle] The EARLIER rung to measure drift from. The GAP is the "
        "instrument: an average moves less the wider the base it covers, so a "
        "settling curve is only a curve at a FIXED gap (e.g. always 400).",
    )
    # blueprint_match options (--method blueprint_match).
    parser.add_argument(
        "--vs-run",
        default=None,
        help="[blueprint_match, depth_duel] The OTHER arm, as a published run id. The chip edge "
        "is reported for --run over this one.",
    )
    parser.add_argument(
        "--vs-at",
        type=int,
        default=None,
        help="[blueprint_match, depth_duel] Rung of --vs-run to play (default: its published "
        "checkpoint). Two arms are only comparable at a matched training budget, so "
        "pass this whenever one of them was continued further than the other.",
    )
    parser.add_argument(
        "--vs-opponent",
        choices=("blueprint", "deployed"),
        default="blueprint",
        help="[blueprint_match] Side B as the bare table or DEPLOYED (through the resolver's "
        "translation, under the same resolver knobs as side A). Deployed is the only way "
        "two different action trees can meet: a bare side raises on a size it lacks.",
    )
    parser.add_argument(
        "--vs-policy-threshold",
        type=float,
        default=0.0,
        help="[blueprint_match, depth_duel] Purify the OPPONENT's table at this probability, the "
        "way the seat purifies what it serves. With --policy-threshold on the same "
        "run and rung this makes a self-match a single-knob A/B over SERVING: the "
        "sampling streams are shared, so the pairs that are not zero are exactly "
        "the decisions the cut changed.",
    )
    # depth_duel options; --vs-run, --vs-at, --deals and both thresholds as above.
    parser.add_argument(
        "--table-bb",
        type=float,
        default=None,
        help="[depth_duel] Table depth in big blinds. Each run plays it through its "
        "OWN tree, rebuilt at the depth it was cut for, as the seat does.",
    )
    # elimination options (--method elimination).
    parser.add_argument(
        "--ladder-a",
        default="",
        help="[elimination] Comma-separated `run[:at[:threshold]]` rungs, deploy.sh's "
        "spelling; a missing threshold plays the table raw. Must include --run.",
    )
    parser.add_argument("--ladder-b", default="", help="[elimination] The opposing ladder.")
    parser.add_argument(
        "--matches",
        type=int,
        default=500,
        help="[elimination] Duplicate match PAIRS: one card sequence played twice, the "
        "ladders swapping seats, so seat and card luck cancel within a pair.",
    )
    parser.add_argument(
        "--starting-bb", type=float, default=100.0, help="[elimination] Opening depth."
    )
    parser.add_argument(
        "--escalate-every",
        type=int,
        default=10,
        help="[elimination] Hands between blind steps, modelled as both stacks "
        "shrinking by --escalate-by. A MODEL of Chipzen's schedule: hold it fixed "
        "across arms, since the win rate moves with it.",
    )
    parser.add_argument("--escalate-by", type=float, default=1.5, help="[elimination] Step.")
    for side in ("a", "b"):
        parser.add_argument(
            f"--select-{side}",
            choices=SELECTION_RULES,
            default="up",
            help=f"[elimination] How ladder {side.upper()} picks a rung: `up` (what the seat "
            "fields: the next rung up within 2x, else log-nearest) or `nearest` in log depth.",
        )
    # LBR options (--method lbr).
    parser.add_argument("--hands", type=int, default=1000, help="[lbr] Number of hands.")
    parser.add_argument("--runouts", type=int, default=12, help="[lbr] Equity runouts per node.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="[lbr, resolver_match] Parallel workers. resolver_match splits DEALS, "
        "which are a pure function of (seed, deal) and so give identical numbers "
        "at any worker count.",
    )
    parser.add_argument(
        "--include-off-tree",
        action="store_true",
        help="[lbr] Add off-tree bet/raise sizes to the exploiter's menu (rigorous via "
        "shadow-state translation; changes the measured completion — re-baseline).",
    )
    parser.add_argument(
        "--allin-runouts",
        type=int,
        default=50,
        help="[lbr] Board runouts averaged at all-in showdown terminals "
        "(variance reduction; same expectation).",
    )
    parser.add_argument(
        "--abstraction-hash",
        default=None,
        help="Pin the card abstraction to this hash (see the abstraction's metadata.json "
        "'config_hash'). Default: the hash recorded on the run.",
    )
    parser.add_argument(
        "--opponent",
        choices=["blueprint", "deployed"],
        default="blueprint",
        help="[lbr, blueprint_match] Strategy under measurement: the raw table, or "
        "blueprint+resolver as deployed. Under blueprint_match it is SIDE A, which "
        "is the only way to measure the resolver against a genuinely off-menu "
        "opponent -- resolver_match plays a blueprint against itself, so every line "
        "it walks is one its own tree has a node for.",
    )
    parser.add_argument(
        "--resolver-iterations",
        type=int,
        default=64,
        help="[lbr] Pinned subgame-CFR iterations per deployed-opponent solve.",
    )
    parser.add_argument(
        "--scorer",
        choices=["myopic", "lookahead"],
        default="myopic",
        help="[lbr] Exploiter action selection: myopic one-step arithmetic, or a "
        "depth-limited best-response lookahead vs the blueprint (stronger exploiter).",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        help="[lbr] Opponent-response levels the lookahead scorer expands.",
    )
    parser.add_argument(
        "--lookahead-top-k",
        type=int,
        default=3,
        help="[lbr] Lookahead-rescore only the top-k myopic candidates (<=0: all).",
    )
    # Exact-BR options (--method exact_br). The board plan defines the comparison
    # tier: evals pair iff flops/turns/rivers and the board seed all match.
    parser.add_argument(
        "--br-flops", type=int, default=8, help="[exact_br] Sampled canonical flops (>=1755: all)."
    )
    parser.add_argument(
        "--progress-file",
        default="",
        help="[exact_br] Write {done,total} flop branches here as they finish. Node-local: "
        "a heartbeat for the task bar, not a record.",
    )
    parser.add_argument(
        "--br-turns", type=int, default=2, help="[exact_br] Turn cards per board node."
    )
    parser.add_argument(
        "--br-rivers", type=int, default=2, help="[exact_br] River cards per board node."
    )
    parser.add_argument(
        "--br-board-seed", type=int, default=7, help="[exact_br] Seed pinning the board sample."
    )
    parser.add_argument(
        "--in-abstraction",
        action="store_true",
        help="[exact_br] Bucket-constrained responder: one action per (node, bucket). The "
        "abstract game's own exploitability; a separate tier from the per-combo default.",
    )
    parser.add_argument(
        "--policy-threshold",
        type=float,
        default=0.0,
        help="[exact_br, blueprint_match, depth_duel] Eval-time thresholding of the blueprint: zero "
        "actions below this probability and renormalise. A separate tier. Under "
        "blueprint_match it purifies side A, which is what the SEAT does to every "
        "rung it serves; see --vs-policy-threshold for the control.",
    )
    parser.add_argument(
        "--purify",
        action="store_true",
        help="[exact_br] Eval-time purification: the blueprint plays its argmax. A separate tier.",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="[exact_br] Attribute the responder's gain to streets, seats, preflop lines and "
        "the top public nodes (recorded in the eval document; `ledger --full` reads it).",
    )
    parser.add_argument(
        "--policy-profile",
        action="store_true",
        help="[exact_br] Record per-street coverage/entropy and the preflop tables of the "
        "checkpoint beside the score.",
    )
    parser.add_argument(
        "--policy-iterate",
        choices=("average", "current"),
        default="average",
        help="[exact_br] Which strategy of the checkpoint to score: the DCFR average, or "
        "the current regret-matched iterate. A separate tier.",
    )
    parser.add_argument(
        "--avg-window-from",
        type=int,
        default=None,
        help="[exact_br] Average only over iterations AFTER this retained rung — exactly the "
        "difference of the two rungs' strategy sums. Combine with --avg-gamma to reweight "
        "WITHIN the window: a fixed-width uniform window at two endpoints holds the averaging "
        "noise constant while the endpoint moves. A separate tier.",
    )
    parser.add_argument(
        "--avg-gamma",
        type=float,
        default=None,
        help="[exact_br] Reweight the average as if training had used this dcfr_gamma, band by "
        "retained ladder band. Costs one rung FETCH and one read per band, per process: pass "
        "--workers 1, and check the run's retention first (a 5M-retention 300M run is 60 "
        "rungs, tens of GB onto the node). --workers 1 then SERIALISES the walk, so pick the "
        "board budget for one worker: measured 09-02, 4/16/16 ran out the 6h task ceiling "
        "twice (rc=124) where 4/2/2 finishes in minutes, and the ladder fetch was 12s of it. "
        "A separate tier.",
    )
    parser.add_argument(
        "--mix-run",
        default=None,
        help="[exact_br, blueprint_match] Blend another RUN's average into this one 50/50 and score the "
        "mixture. Both runs must bucket by the same abstraction (refused otherwise). Tests "
        "whether the solver's residual error is independent ACROSS runs, not just across "
        "training spans. A separate tier.",
    )
    parser.add_argument(
        "--mix-at",
        type=int,
        default=None,
        help="[exact_br, blueprint_match] Rung of --mix-run to blend (default: its published checkpoint).",
    )
    parser.add_argument(
        "--mix-weight",
        type=float,
        default=0.5,
        help="[exact_br, blueprint_match] Weight on --mix-run. 0.5 is the even blend; each side is divided "
        "by its own total mass first, so this weights STRATEGIES, not accumulated mass.",
    )
    parser.add_argument(
        "--br-conditional",
        action="store_true",
        help="[exact_br] Condition chance on the deal: divide each street's branch weights by "
        "the fraction a four-card deal leaves compatible, so a voided branch is not a refund. "
        "Exact at full enumeration; a separate comparison tier from the annulled default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random for lbr, 1 for resolver_match). For "
        "resolver_match it picks the DEALS, so it is how a gate arm is replicated: "
        "several seeds are several independent samples of the same knob tier.",
    )


def run(args: argparse.Namespace) -> services.EvaluationPayload:
    """Argparse transport around :func:`services.evaluate_and_record`.

    All dispatch, payload shaping, and ledger recording live in the orchestrator;
    this function only maps flags onto the params objects. The orchestrator's
    ledger warning prints to stdout, which under ``--json`` is redirected to
    stderr — keeping the machine-readable payload clean.
    """
    run_dir = resolve_run_dir(args.run, args.runs_dir)
    return services.evaluate_and_record(
        run_dir,
        # The composition root, same as the trainers': `pipeline` holds a port
        # and this decides what implements it.
        sink=connect.eval_sink_from_environment(),
        record_source=connect.record_source_from_environment(),
        method=args.method,
        lbr=LBRConfig(
            num_hands=args.hands,
            equity_runouts=args.runouts,
            include_off_tree=args.include_off_tree,
            seed=args.seed,
            num_workers=args.workers,
            allin_runouts=args.allin_runouts,
            opponent=args.opponent,
            scorer=args.scorer,
            lookahead_depth=args.lookahead_depth,
            lookahead_top_k=args.lookahead_top_k,
        ),
        exact_br=PublicBRConfig(
            num_flops=args.br_flops,
            num_turns=args.br_turns,
            num_rivers=args.br_rivers,
            board_seed=args.br_board_seed,
            conditional_chance=args.br_conditional,
            # --workers is shared with lbr; exact_br farms flop subtrees over
            # it, one blueprint per process, so memory caps it before cores do.
            num_workers=args.workers,
            in_abstraction=args.in_abstraction,
            policy_threshold=args.policy_threshold,
            purify=args.purify,
            decompose=args.decompose,
            # `--opponent deployed` already means "the fielded system" for
            # lbr; exact_br now honours it instead of silently scoring the
            # stored rows under a flag that says otherwise.
            deployed=args.opponent == "deployed",
            resolver_iterations=args.resolver_iterations,
            resolver_prior_weight=(
                args.resolver_prior_weight
                if args.resolver_prior_weight is not None
                else PublicBRConfig.resolver_prior_weight
            ),
            policy_iterate=args.policy_iterate,
            avg_window_from=args.avg_window_from,
            avg_gamma=args.avg_gamma,
            mix_run=resolve_run_dir(args.mix_run, args.runs_dir) if args.mix_run else None,
            mix_at=args.mix_at,
            mix_weight=args.mix_weight,
        ),
        resolver_iterations=args.resolver_iterations,
        resolver_gate_deals=args.deals if args.deals is not None else RESOLVER_GATE_DEALS,
        resolver_gate_workers=args.workers,
        # `--seed` means "random" for lbr and "these deals" for the gate, so the
        # gate keeps its deterministic default rather than inheriting None.
        resolver_gate_seed=args.seed if args.seed is not None else 1,
        resolver_root_prior_weight=args.resolver_prior_weight,
        resolver_blend_alpha=args.resolver_blend_alpha,
        resolver_leaf_rollouts=args.resolver_leaf_rollouts,
        leaf_continuation_fraction=args.leaf_continuation,
        resolver_max_iterations=args.resolver_max_iterations,
        resolver_allin_runouts=args.resolver_allin_runouts,
        settle_from=args.settle_from,
        vs_run_dir=resolve_run_dir(args.vs_run, args.runs_dir) if args.vs_run else None,
        vs_at=args.vs_at,
        match_deals=args.deals if args.deals is not None else MATCH_DEALS,
        match_seed=args.seed if args.seed is not None else 1,
        match_threshold_a=args.policy_threshold,
        match_threshold_b=args.vs_policy_threshold,
        match_resolver_a=args.opponent == "deployed",
        match_resolver_b=args.vs_opponent == "deployed",
        table_bb=args.table_bb,
        ladder_a=_ladder(args.ladder_a, args.runs_dir),
        ladder_b=_ladder(args.ladder_b, args.runs_dir),
        elimination_pairs=args.matches,
        starting_bb=args.starting_bb,
        escalate_every=args.escalate_every,
        escalate_by=args.escalate_by,
        select_a=args.select_a,
        select_b=args.select_b,
        resolver_min_street=args.resolver_min_street,
        abstraction_hash=args.abstraction_hash,
        at_iteration=args.at,
        progress_file=Path(args.progress_file) if args.progress_file else None,
        policy_profile=args.policy_profile,
    )


def _ladder(text: str, runs_dir: str) -> list[tuple[Path, int | None, float]]:
    """``--ladder-a``/``-b`` as (run dir, rung, threshold); empty when not given."""
    if not text:
        return []
    try:
        specs = parse_ladder(text)
    except ValueError as exc:
        raise CommandError(str(exc)) from exc
    return [(resolve_run_dir(spec.run, runs_dir), spec.at, spec.threshold) for spec in specs]


def _print_resolver_telemetry(results: dict[str, Any]) -> None:
    """The resolver's stack-off rate and root entropy, when a row carries them.

    Shared by the resolver gate and a deployed chip match, because both measure
    the same two quantities and the chip edge of neither can resolve them: the
    estimator's own docstring puts 50 mbb resolution at ~46,500 deals, while a
    per-aggression rate carries a fraction of a point at a thousand.

    `get`, because every row recorded before 09-30 carries none of these and the
    ledger re-renders those.
    """
    ours = results.get("resolver_stack_off_rate")
    theirs = results.get("blueprint_stack_off_rate")
    if ours is not None and theirs is not None:
        ratio = f"{ours / theirs:.2f}x" if theirs else "no blueprint shove to compare"
        print(
            f"  Stack-offs:    {ours:.2%} of our aggressions vs {theirs:.2%} for the "
            f"bare blueprint ({ratio})"
        )
        # With their DENOMINATORS: a rate printed alone cannot be tested, and the
        # first run of this gate reported three ratios nobody could put a sigma on.
        print(
            f"                 {results['resolver_stack_offs']:,} of "
            f"{results['resolver_aggressions']:,} aggressions vs "
            f"{results['blueprint_stack_offs']:,} of "
            f"{results['blueprint_aggressions']:,}"
        )
    if (entropy := results.get("resolver_root_entropy")) is not None:
        print(
            f"  Root entropy:  {entropy:.3f} resolver, "
            f"{results['played_root_entropy']:.3f} played  (1.0 = UNIFORM over the menu)"
        )


def render(payload: services.EvaluationPayload) -> None:
    results = payload.results
    print("Evaluation complete.")
    print(f"  Run ID:        {payload.run_id}")
    print(f"  Estimator:     {payload.estimator}")
    print(f"  Infosets:      {payload.infosets:,}")
    if not results:
        # An evaluation that computed and recorded nothing has happened before;
        # it must read as that, not as a renderer crash.
        print("  no results recorded")
        return
    # Branch on the METHOD, not on what happens to be in `results`. A resolver
    # gate reports a chip edge and carries no `exploitability_mbb` at all; a
    # renderer that reached for the key would die on it, which is exactly how
    # `checkpoint-profile` once died borrowing this function.
    if payload.method == "settle":
        by_street = results.get("by_street") or {}
        gap = results.get("rung_gap")
        print(f"  Drift since:   rung {results.get('from_iteration')} (gap {gap})")
        print(
            f"  {'street':<9}{'rows':>12}{'median TV':>12}{'reach-wtd':>12}{'pure':>9}{'thin+pure':>11}"
        )
        for street in ("preflop", "flop", "turn", "river"):
            row = by_street.get(street)
            if not row:
                continue
            print(
                f"  {street:<9}{row['rows']:>12,}{row['median_tv']:>12.5f}"
                f"{row['reach_weighted_tv']:>12.5f}{row['pure_fraction']:>8.1%}"
                f"{row['thin_pure_fraction']:>11.1%}"
            )
        # The trap this estimator exists to avoid: drift falls as rows go PURE,
        # and a frozen row is not a settled one.
        print("  reach-wtd is the TV below which half the TRAFFIC sits; a PURE row's")
        print("  drift is exactly 0, so read `pure` beside the drift, and read both")
        print("  beside the exploitability of the same rungs.")
        return
    if payload.method == "depth_duel":
        _render_depth_duel(payload.run_id, results)
        return
    if payload.method == "elimination":
        _render_elimination(results)
        return
    if payload.method == "blueprint_match":
        edge, se = results["a_mbb_per_hand"], results["se_mbb"]
        sigma = abs(edge) / se if se else 0.0
        winner = payload.run_id if edge > 0 else results["run_b"]
        rung_b = results.get("rung_b")
        print(f"  vs:            {results['run_b']} @ {rung_b if rung_b else 'latest'}")
        thr_a, thr_b = (
            results.get("policy_threshold_a", 0.0),
            results.get("policy_threshold_b", 0.0),
        )
        if thr_a or thr_b:
            # Rows changed, not just the knob: a cut that moved nothing is a
            # no-op arm whose exact zero says nothing about serving.
            print(
                f"  Served at:     A {thr_a:.2f} ({results.get('rows_thresholded_a', 0):,} rows"
                f" cut)  B {thr_b:.2f} ({results.get('rows_thresholded_b', 0):,} rows cut)"
            )
        if results.get("opponent") == "deployed":
            # The deployed side's own numbers. Printed here as well as in the
            # resolver-gate branch because a deployed CHIP match is the only place
            # the resolver's off-tree benefit can show, and the first one printed
            # nothing that said the resolver had been in the hand.
            print(
                f"  Side A played: DEPLOYED (blueprint+resolver), "
                f"{results.get('resolver_max_iterations')} iterations, "
                f"prior {results.get('root_prior_weight')}"
            )
            _print_resolver_telemetry(results)
        divergent = results.get("divergent_pairs")
        if divergent is not None:
            # The knob's real sample size: duplicate deals cancel exactly where
            # the two sides agree, so an edge rests only on these pairs.
            share = divergent / results["num_deals"] if results["num_deals"] else 0.0
            print(f"  Diverged on:   {divergent:,} of {results['num_deals']:,} pairs ({share:.1%})")
        print(f"  Edge (A over B): {edge:+.2f} mbb/hand (± {se:.2f})  {sigma:.1f}σ")
        print(
            f"  Deals:         {results['num_deals']:,} duplicate ({results['num_hands']:,} hands)"
        )
        # Two mediocre strategies can be far apart head-to-head and level on
        # exploitability, which is the whole reason this estimator exists.
        print(f"  Reading:       {'inconclusive' if sigma < 2 else f'{winner} wins'}")
        return
    if payload.method == "resolver_match":
        # `confidence_95_mbb` is an INTERVAL (lower, upper), not a half-width --
        # printing it as a scalar is a TypeError, which is how this line first
        # shipped. An interval is also the more honest thing to show: it says
        # whether zero is inside it.
        # Printed FIRST: the chip edge below has never resolved at any deal count
        # this project runs (se ~80 at 4,000 deals), while these two are what the
        # resolver was actually retired on and carry ~0.2 points at the same cost.
        # `get`, because every row recorded before 09-30 carries neither and the
        # ledger re-renders those.
        _print_resolver_telemetry(results)
        low, high = results["confidence_95_mbb"]
        print(
            f"  Resolver edge: {results['resolver_mbb_per_hand']:+.2f} mbb/hand "
            f"(95% CI {low:+.2f}..{high:+.2f}, p={results['p_value']:.4f})"
        )
        print(f"  Deals:         {results['num_deals']:,} ({results['num_hands']:,} hands)")
        print(
            f"  Leaf contin.:  {results['leaf_continuation_fraction']:g} pot"
            f"   iterations: {results['resolver_max_iterations'] or 'wall-clock'}"
        )
        # A high fallback count means the number measures the FALLBACK, not the
        # resolver -- so it is printed beside the edge, not buried in the payload.
        print(
            f"  Decisions:     {results['resolver_decisions']:,} "
            f"({results['resolver_fallbacks']:,} fell back to the blueprint)"
        )
        return
    print(
        f"  Exploitability: {results['exploitability_mbb']:.2f} mbb/g "
        f"(± {results['std_error_mbb']:.2f})"
    )
    # The reference nobody had: production arms read 722-748 against a 750
    # baseline and nobody noticed they were level with folding every hand.
    folding = results.get("fold_everything_mbb")
    if folding:
        margin = folding - results["exploitability_mbb"]
        verdict = "BEATS" if margin > 0 else "WORSE THAN"
        print(
            f"  vs folding:     {folding:.0f} mbb/g is fold-every-hand — "
            f"this {verdict} it by {abs(margin):.0f}"
        )
    # SHAPE beside the score. The gate ranked arms by exploitability alone and
    # r(SB fold, exploitability) = -0.855 across every arm ever profiled: the
    # best-scoring band folded 65-80% of the button where real heads-up folds
    # 10-20%, and the arm that folds MORE wins on a number while losing the
    # chip match 3.23σ. A scalar cannot separate a strategy from an abdication.
    fold = results.get("sb_first_in_fold")
    if fold is not None:
        verdict = "NOT A CANDIDATE" if fold > SB_FOLD_REFUSES else "ok"
        print(
            f"  SB first in:    folds {fold:.1%} of the button "
            f"(real heads-up folds 10-20%; >{SB_FOLD_REFUSES:.0%} {verdict})"
        )
    for seat in results.get("seat_values_mbb") or []:
        if "self_play_mbb" not in seat:
            continue
        print(
            f"    seat {seat['br_seat']} button {seat['button']}: "
            f"BR {seat['value_mbb']:9.2f}  self-play {seat['self_play_mbb']:9.2f}  "
            f"gain {seat['gain_mbb']:9.2f}"
        )
    if decomposition := results.get("decomposition"):
        _render_decomposition(decomposition)
    if profile := results.get("policy_profile"):
        _render_profile(profile)


def _render_depth_duel(run_id: str, results: dict[str, Any]) -> None:
    edge, se = results["a_mbb_per_hand"], results["se_mbb"]
    sigma = abs(edge) / se if se else 0.0
    winner = run_id if edge > 0 else results["run_b"]
    print(f"  vs:            {results['run_b']} @ {results.get('rung_b') or 'latest'}")
    print(
        f"  Table:         {results['table_bb']:g} bb; A cut for "
        f"{results['depth_a_bb']:g} bb, B for {results['depth_b_bb']:g} bb"
    )
    print(
        f"  Served at:     A {results['policy_threshold_a']:.2f} "
        f"({results['rows_thresholded_a']:,} rows cut)  B {results['policy_threshold_b']:.2f} "
        f"({results['rows_thresholded_b']:,} rows cut)"
    )
    # What the depth fiction cost each side: a truncated replay is a forced
    # check/fold, so a high rate is the player's depth failing, not its strategy.
    for side in ("a", "b"):
        print(
            f"  Fiction {side.upper()}:     {results[f'{side}_truncation_rate']:.2%} truncated, "
            f"{results[f'{side}_snap_rate']:.2%} snapped, "
            f"{results[f'{side}_untrained_rate']:.2%} untrained "
            f"of {results[f'{side}_decisions']:,} decisions"
        )
    print(f"  Diverged on:   {results['divergent_pairs']:,} of {results['num_deals']:,} pairs")
    print(f"  Edge (A over B): {edge:+.2f} mbb/hand (± {se:.2f})  {sigma:.1f}σ")
    print(f"  Reading:       {'inconclusive' if sigma < 2 else f'{winner} wins'}")


def _render_elimination(results: dict[str, Any]) -> None:
    rate, se = results["a_win_rate"], results["se"]
    sigma = abs(rate - 0.5) / se if se else 0.0
    print(f"  Ladder A:      {', '.join(results['ladder_a'])}")
    print(f"  Ladder B:      {', '.join(results['ladder_b'])}")
    print(
        f"  Schedule:      {results['starting_bb']:g} bb, stacks / {results['escalate_by']:g} "
        f"every {results['escalate_every']} hands"
    )
    print(
        f"  Matches:       {results['num_matches']:,} ({results['num_pairs']:,} duplicate "
        f"pairs), {results['mean_hands']:.1f} hands each on average"
    )
    print(
        f"  Truncation:    A {results['truncation_rate_a']:.2%}  B "
        f"{results['truncation_rate_b']:.2%}; rung switches A {results['switches_a']:,} "
        f"B {results['switches_b']:,}"
    )
    print(f"  A wins:        {rate:.1%} of matches (± {se:.1%})  {sigma:.1f}σ from even")


def _render_decomposition(decomposition: dict) -> None:
    """The attribution as a table: where the responder's gain comes from."""
    print(f"  Decomposition (identity gap {decomposition['identity']['max_abs_gap_mbb']:.2e} mbb):")
    streets = decomposition["by_street"]
    print("    by street:   " + "  ".join(f"{k} {v:8.1f}" for k, v in streets.items()))
    print(
        "    by position: "
        + "  ".join(f"{k} {v:8.1f}" for k, v in decomposition["by_position"].items())
    )
    for walk in decomposition["identity"]["per_walk"]:
        by_street = "  ".join(f"{v:7.1f}" for v in walk["by_street"].values())
        print(
            f"    seat {walk['br_seat']} button {walk['button']}: gain {walk['gain_mbb']:8.1f}"
            f"  [{by_street}]"
        )
    print("    top nodes (walk-level mbb; blueprint -> best response):")
    for node in decomposition["top_nodes"][:12]:
        blueprint = " ".join(f"{k}:{v:.2f}" for k, v in node["blueprint"].items())
        best = " ".join(f"{k}:{v:.2f}" for k, v in node["best_response"].items())
        print(
            f"      {node['gain_mbb']:8.1f}  s{node['br_seat']}b{node['button']} "
            f"{node['street']:<7} {node['sequence'] or '(root)':<28} reach {node['reach']:.3f}"
        )
        print(f"{'':18}{blueprint}  ->  {best}")
    print(
        f"    self-play fallback mass: {decomposition['selfplay_missing_policy_mass']:.6f}"
        f"  nodes with gain: {decomposition['nodes_with_gain']:,}/{decomposition['responder_nodes']:,}"
    )


def _render_profile(profile: dict) -> None:
    print("  Policy profile (entropy normalised by log(actions); p50 average / current):")
    for street, row in profile["streets"].items():
        if "average_entropy" not in row:
            print(f"    {street:<8} visited {row['visited_fraction']:.4f}")
            continue
        print(
            f"    {street:<8} visited {row['visited_fraction']:.4f}  "
            f"H_avg p50 {row['average_entropy']['p50']:.3f}  "
            f"H_cur p50 {row['current_entropy']['p50']:.3f}  "
            f"pure {row['pure_fraction']:.3f}  no+regret {row['no_positive_regret_fraction']:.3f}  "
            f"spread p50 {row.get('bucket_spread', {}).get('p50', float('nan')):.3f}"
        )
    for node in profile["preflop"]:
        mix = " ".join(f"{k}:{v:.3f}" for k, v in node["combo_weighted_mix"].items())
        print(f"    {node['label']}: {mix}")


COMMAND = Command(
    name="evaluate",
    help="Evaluate a run's exploitability (Local Best Response by default).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
