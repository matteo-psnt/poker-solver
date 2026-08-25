"""The `evaluate` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.interfaces.commands._base import (
    Command,
    resolve_run_dir,
)
from src.pipeline import services
from src.pipeline.evaluation.estimators.lbr.config import LBRConfig
from src.pipeline.evaluation.estimators.public_tree_br import PublicBRConfig
from src.shared.config import DEFAULT_RUNS_DIR

if TYPE_CHECKING:
    import argparse

# The estimators a node can actually run. `score` imports this rather than
# repeating it: a value the submitter accepts but `evaluate` rejects is not
# caught until the node has already been allocated, and the task then retries
# twice on the way to failing.
EVAL_METHODS = ("lbr", "exact_br", "resolver_match")


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
        "blueprint+resolver over the bare blueprint (NOT an exploitability bound).",
    )
    # resolver_match options (--method resolver_match).
    parser.add_argument(
        "--deals",
        type=int,
        default=1000,
        help="[resolver_match] Duplicate deals (each played twice, seats swapped).",
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
        help="[lbr] Strategy under measurement: raw table, or blueprint+resolver as deployed.",
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
        help="[exact_br] Eval-time thresholding of the blueprint: zero actions below this "
        "probability and renormalise. A separate tier.",
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
        "difference of the two rungs' strategy sums. A separate tier.",
    )
    parser.add_argument(
        "--br-conditional",
        action="store_true",
        help="[exact_br] Condition chance on the deal: divide each street's branch weights by "
        "the fraction a four-card deal leaves compatible, so a voided branch is not a refund. "
        "Exact at full enumeration; a separate comparison tier from the annulled default.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random).")


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
            policy_iterate=args.policy_iterate,
            avg_window_from=args.avg_window_from,
        ),
        resolver_iterations=args.resolver_iterations,
        resolver_gate_deals=args.deals,
        resolver_gate_workers=args.workers,
        resolver_root_prior_weight=args.resolver_prior_weight,
        resolver_blend_alpha=args.resolver_blend_alpha,
        resolver_leaf_rollouts=args.resolver_leaf_rollouts,
        leaf_continuation_fraction=args.leaf_continuation,
        resolver_max_iterations=args.resolver_max_iterations,
        resolver_allin_runouts=args.resolver_allin_runouts,
        abstraction_hash=args.abstraction_hash,
        at_iteration=args.at,
        progress_file=Path(args.progress_file) if args.progress_file else None,
        policy_profile=args.policy_profile,
    )


def render(payload: services.EvaluationPayload) -> None:
    results = payload.results
    print("Evaluation complete.")
    print(f"  Run ID:        {payload.run_id}")
    print(f"  Estimator:     {payload.estimator}")
    print(f"  Infosets:      {payload.infosets:,}")
    # Branch on the METHOD, not on what happens to be in `results`. A resolver
    # gate reports a chip edge and carries no `exploitability_mbb` at all; a
    # renderer that reached for the key would die on it, which is exactly how
    # `checkpoint-profile` once died borrowing this function.
    if payload.method == "resolver_match":
        # `confidence_95_mbb` is an INTERVAL (lower, upper), not a half-width --
        # printing it as a scalar is a TypeError, which is how this line first
        # shipped. An interval is also the more honest thing to show: it says
        # whether zero is inside it.
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
            f"pure {row['pure_fraction']:.3f}  no+regret {row['no_positive_regret_fraction']:.3f}"
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
