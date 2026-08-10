"""The `evaluate` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.interfaces.commands._base import (
    Command,
    resolve_run_dir,
)
from src.pipeline import services
from src.pipeline.evaluation.estimators.lbr.config import LBRConfig
from src.pipeline.evaluation.estimators.public_tree_br import PublicBRConfig
from src.shared.config import DEFAULT_RUNS_DIR

# The estimators a node can actually run. `score` imports this rather than
# repeating it: a value the submitter accepts but `evaluate` rejects is not
# caught until the node has already been allocated, and the task then retries
# twice on the way to failing.
EVAL_METHODS = ("lbr", "exact_br")


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
        "matched board tier).",
    )
    # LBR options (--method lbr).
    parser.add_argument("--hands", type=int, default=1000, help="[lbr] Number of hands.")
    parser.add_argument("--runouts", type=int, default=12, help="[lbr] Equity runouts per node.")
    parser.add_argument("--workers", type=int, default=1, help="[lbr] Parallel workers over hands.")
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random).")


def run(args: argparse.Namespace) -> dict[str, Any]:
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
            # --workers is shared with lbr; exact_br splits its four independent
            # (seat, button) walks over it, so 4 saturates the useful range.
            num_workers=args.workers,
        ),
        resolver_iterations=args.resolver_iterations,
        abstraction_hash=args.abstraction_hash,
        at_iteration=args.at,
        progress_file=Path(args.progress_file) if args.progress_file else None,
    )


def render(payload: dict[str, Any]) -> None:
    results = payload["results"]
    print("Evaluation complete.")
    print(f"  Run ID:        {payload['run_id']}")
    print(f"  Estimator:     {payload['estimator']}")
    print(f"  Infosets:      {payload['infosets']:,}")
    print(
        f"  Exploitability: {results['exploitability_mbb']:.2f} mbb/g "
        f"(± {results['std_error_mbb']:.2f})"
    )


COMMAND = Command(
    name="evaluate",
    help="Evaluate a run's exploitability (Local Best Response by default).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
