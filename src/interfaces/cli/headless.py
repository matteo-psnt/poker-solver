"""Non-interactive (headless) entrypoint for training and evaluation.

Unlike the questionary menu in :mod:`src.interfaces.cli.app`, every operation here
is fully specified by CLI flags and emits a machine-readable summary. This is the
surface used by scripts and cloud (Modal) execution — where an interactive prompt is
not an option.

Cloud callers should prefer importing :func:`src.pipeline.training.services.train`
directly (it returns a ``TrainingOutput`` object); this module is the local /
subprocess transport around the same function and additionally writes a
``result.json`` into the run directory.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.statistics import compare_paired_samples
from src.pipeline.training import services
from src.pipeline.training.services import LBR_ESTIMATOR_LABEL, ROLLOUT_ESTIMATOR_LABEL


def _json_default(obj: Any) -> Any:
    """Coerce non-JSON-native values (e.g. numpy scalars) to plain types."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _write_result(run_dir: Path, payload: dict[str, Any]) -> None:
    """Persist a per-operation result file (e.g. ``train_result.json``) in the run dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{payload['op']}_result.json").write_text(
        json.dumps(payload, indent=2, default=_json_default)
    )


def _resolve_run_dir(run: str, runs_dir: str) -> Path:
    """Resolve a run identifier (name under ``runs_dir``) or an explicit path."""
    as_path = Path(run)
    if as_path.is_dir():
        return as_path
    candidate = Path(runs_dir) / run
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"Run not found: '{run}' (looked at {as_path} and {candidate})")


def _cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    out = services.train(
        args.config,
        num_workers=args.workers,
        num_iterations=args.iterations,
        seed=args.seed,
    )
    payload: dict[str, Any] = {"op": "train", **dataclasses.asdict(out)}
    _write_result(Path(out.runs_dir) / out.run_id, payload)
    return payload


def _cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _resolve_run_dir(args.run, args.runs_dir)
    if args.method == "rollout":
        out = services.evaluate_run_rollout(
            run_dir=run_dir,
            num_samples=args.samples,
            num_rollouts=args.rollouts,
            use_average_strategy=not args.current,
            seed=args.seed,
        )
        estimator = ROLLOUT_ESTIMATOR_LABEL
    else:  # "lbr" (default, trustworthy)
        out = services.evaluate_run_lbr(
            run_dir=run_dir,
            num_hands=args.hands,
            equity_runouts=args.runouts,
            include_off_tree=args.include_off_tree,
            seed=args.seed,
            num_workers=args.workers,
            opponent=args.opponent,
            resolver_iterations=args.resolver_iterations,
            scorer=args.scorer,
            lookahead_depth=args.lookahead_depth,
            lookahead_top_k=args.lookahead_top_k,
        )
        estimator = LBR_ESTIMATOR_LABEL
    payload: dict[str, Any] = {
        "op": "evaluate",
        "run_id": run_dir.name,
        "method": args.method,
        "estimator": estimator,
        "infosets": out.infosets,
        "results": out.results,
    }
    _write_result(run_dir, payload)
    _record_to_ledger(run_dir, args, payload, estimator)
    return payload


def _record_to_ledger(
    run_dir: Path, args: argparse.Namespace, payload: dict[str, Any], estimator: str
) -> None:
    """Persist the full eval payload (no clobber) and append a compact ledger row.

    Best-effort: the ledger is a research convenience, so a recording failure warns
    but never fails the evaluation itself. The warning prints to stdout, which under
    ``--json`` is redirected to stderr — keeping the machine-readable payload clean.
    """
    ledger_path = Path(args.ledger)
    try:
        results = payload["results"]
        if args.method == "rollout":
            knobs = eval_ledger.build_rollout_knobs(args, results)
        else:
            knobs = eval_ledger.build_lbr_knobs(args, results)
        result_path, _ = eval_ledger.record_evaluation(
            run_dir=run_dir,
            payload=payload,
            method=args.method,
            estimator=estimator,
            knobs=knobs,
            ledger_path=ledger_path,
        )
        print(f"  Ledger:        appended to {ledger_path} (payload: {result_path})")
    except Exception as exc:
        print(f"  Ledger:        skipped ({type(exc).__name__}: {exc})")


def _cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    """List recent eval-ledger rows as a compact table."""
    records = eval_ledger.read_records(Path(args.ledger))
    if args.run:
        records = [r for r in records if r.get("run_id") == args.run]
    records = records[-args.limit :]
    return {"op": "ledger", "ledger": str(args.ledger), "rows": records}


def _cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Paired (common-random-numbers) comparison of two runs' latest evals."""
    ledger_path = Path(args.ledger)
    rec_a = eval_ledger.latest_record_for_run(args.a, ledger_path)
    rec_b = eval_ledger.latest_record_for_run(args.b, ledger_path)
    if rec_a is None or rec_b is None:
        missing = args.a if rec_a is None else args.b
        raise SystemExit(f"No ledger entry found for run '{missing}' in {ledger_path}")

    reasons = eval_ledger.tier_mismatches(rec_a, rec_b)
    if reasons and not args.force:
        joined = "\n".join(f"  - {r}" for r in reasons)
        raise SystemExit(
            "Refusing to compare: the two evals are not a valid paired comparison:\n"
            f"{joined}\n"
            "Re-run both evals with matching knobs and the same --seed, or pass --force "
            "to override (the resulting p-value will not be trustworthy)."
        )

    payload_a = eval_ledger.load_payload(rec_a)
    payload_b = eval_ledger.load_payload(rec_b)
    comparison = compare_paired_samples(
        payload_a["results"]["pair_samples_mbb"],
        payload_b["results"]["pair_samples_mbb"],
    )
    return {
        "op": "compare",
        "run_a": args.a,
        "run_b": args.b,
        "forced": bool(reasons and args.force),
        "tier_warnings": reasons,
        "comparison": comparison,
    }


def _fmt_commit(commit: str | None, dirty: bool | None) -> str:
    if not commit:
        return "—"
    short = commit[:7]
    if dirty:
        short += "-dirty"
    return short


def _print_ledger(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    if not rows:
        print(f"No eval-ledger entries in {payload['ledger']}.")
        return
    print(f"Eval ledger ({payload['ledger']}): {len(rows)} row(s)")
    header = (
        f"{'run_id':<26} {'commit':<14} {'scorer':<10} {'opp':<10} "
        f"{'seed':>12} {'hands':>6} {'mbb/g':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        knobs = r.get("knobs", {})
        res = r.get("results", {})
        mbb = res.get("exploitability_mbb")
        se = res.get("std_error_mbb")
        score = f"{mbb:.1f}±{se:.1f}" if isinstance(mbb, (int, float)) and se is not None else "—"
        print(
            f"{r.get('run_id', '')[:26]:<26} "
            f"{_fmt_commit(r.get('eval_git_commit'), r.get('eval_git_dirty')):<14} "
            f"{knobs.get('scorer', '')!s:<10} "
            f"{knobs.get('opponent', '')!s:<10} "
            f"{knobs.get('base_seed', '')!s:>12} "
            f"{res.get('num_hands', '')!s:>6} "
            f"{score:>12}"
        )


def _print_compare(payload: dict[str, Any]) -> None:
    c = payload["comparison"]
    print(f"Paired comparison: {payload['run_a']}  vs  {payload['run_b']}")
    if payload["tier_warnings"]:
        print("  ⚠️  FORCED over tier mismatches (p-value not trustworthy):")
        for w in payload["tier_warnings"]:
            print(f"     - {w}")
    print(f"  mean(a):       {c['mean_a']:+.2f} mbb/g")
    print(f"  mean(b):       {c['mean_b']:+.2f} mbb/g")
    print(f"  mean_diff:     {c['mean_diff']:+.2f} mbb/g (± {c['se_diff']:.2f})")
    print(f"  95% CI:        [{c['ci_lower']:+.2f}, {c['ci_upper']:+.2f}]")
    print(
        f"  p-value:       {c['p_value']:.4g}  ({'significant' if c['is_significant'] else 'n.s.'})"
    )
    print(f"  correlation:   {c['correlation']:.3f}  (se unpaired would be {c['se_unpaired']:.2f})")


def _print_human(payload: dict[str, Any]) -> None:
    if payload["op"] == "ledger":
        _print_ledger(payload)
        return
    if payload["op"] == "compare":
        _print_compare(payload)
        return
    if payload["op"] == "train":
        print("Training complete.")
        print(f"  Run ID:      {payload['run_id']}  (under {payload['runs_dir']})")
        print(f"  Config:      {payload['config_name']}")
        print(f"  Iterations:  {payload['iterations']:,}")
        print(f"  Infosets:    {payload['num_infosets']:,}")
        print(
            f"  Runtime:     {payload['runtime_seconds']:.2f}s "
            f"({payload['iterations_per_second']:.1f} it/s)"
        )
        print(f"  Status:      {payload['status']}")
        return

    results = payload["results"]
    print("Evaluation complete.")
    print(f"  Run ID:        {payload['run_id']}")
    print(f"  Estimator:     {payload['estimator']}")
    print(f"  Infosets:      {payload['infosets']:,}")
    print(
        f"  Exploitability: {results['exploitability_mbb']:.2f} mbb/g "
        f"(± {results['std_error_mbb']:.2f})"
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        help="Emit the result payload as JSON only (no human-readable summary).",
    )

    parser = argparse.ArgumentParser(
        prog="poker-solver-run",
        description="Headless training/evaluation entrypoint for scripts and cloud runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train", parents=[common], help="Train a solver from a named training config."
    )
    p_train.add_argument("--config", required=True, help="Config stem under config/training/.")
    p_train.add_argument(
        "--workers", type=int, default=None, help="Parallel workers (default: all CPUs)."
    )
    p_train.add_argument(
        "--iterations", type=int, default=None, help="Override the config iteration count."
    )
    p_train.add_argument("--seed", type=int, default=None, help="Override system.seed.")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser(
        "evaluate",
        parents=[common],
        help="Evaluate a run's exploitability (Local Best Response by default).",
    )
    p_eval.add_argument("--run", required=True, help="Run id (dir name) or path to a run dir.")
    p_eval.add_argument("--runs-dir", default="data/runs", help="Base runs dir for id resolution.")
    p_eval.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Append-only eval ledger path (records provenance + knobs + result).",
    )
    p_eval.add_argument(
        "--method",
        choices=["lbr", "rollout"],
        default="lbr",
        help="lbr = Local Best Response (trustworthy, default); rollout = legacy diagnostic.",
    )
    # LBR options (--method lbr).
    p_eval.add_argument("--hands", type=int, default=1000, help="[lbr] Number of hands.")
    p_eval.add_argument("--runouts", type=int, default=12, help="[lbr] Equity runouts per node.")
    p_eval.add_argument("--workers", type=int, default=1, help="[lbr] Parallel workers over hands.")
    p_eval.add_argument(
        "--include-off-tree",
        action="store_true",
        help="[lbr] Add off-tree bet/raise sizes to the exploiter's menu (rigorous via "
        "shadow-state translation; changes the measured completion — re-baseline).",
    )
    p_eval.add_argument(
        "--opponent",
        choices=["blueprint", "deployed"],
        default="blueprint",
        help="[lbr] Strategy under measurement: raw table, or blueprint+resolver as deployed.",
    )
    p_eval.add_argument(
        "--resolver-iterations",
        type=int,
        default=64,
        help="[lbr] Pinned subgame-CFR iterations per deployed-opponent solve.",
    )
    p_eval.add_argument(
        "--scorer",
        choices=["myopic", "lookahead"],
        default="myopic",
        help="[lbr] Exploiter action selection: myopic one-step arithmetic, or a "
        "depth-limited best-response lookahead vs the blueprint (stronger exploiter).",
    )
    p_eval.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        help="[lbr] Opponent-response levels the lookahead scorer expands.",
    )
    p_eval.add_argument(
        "--lookahead-top-k",
        type=int,
        default=3,
        help="[lbr] Lookahead-rescore only the top-k myopic candidates (<=0: all).",
    )
    # Rollout options (--method rollout).
    p_eval.add_argument("--samples", type=int, default=500, help="[rollout] Number of samples.")
    p_eval.add_argument("--rollouts", type=int, default=50, help="[rollout] Rollouts per infoset.")
    p_eval.add_argument(
        "--current",
        action="store_true",
        help="[rollout] Evaluate the current strategy instead of the average.",
    )
    p_eval.add_argument("--seed", type=int, default=None, help="Random seed (default: random).")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_ledger = sub.add_parser(
        "ledger", parents=[common], help="List recorded evaluations from the eval ledger."
    )
    p_ledger.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Eval ledger path to read.",
    )
    p_ledger.add_argument("--run", default=None, help="Filter to a single run id.")
    p_ledger.add_argument("--limit", type=int, default=25, help="Show only the last N rows.")
    p_ledger.set_defaults(func=_cmd_ledger)

    p_compare = sub.add_parser(
        "compare",
        parents=[common],
        help="Paired (CRN) comparison of two runs' latest evals; refuses mismatched tiers.",
    )
    p_compare.add_argument("--a", required=True, help="First run id (baseline).")
    p_compare.add_argument("--b", required=True, help="Second run id (candidate).")
    p_compare.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Eval ledger path to read.",
    )
    p_compare.add_argument(
        "--force",
        action="store_true",
        help="Compare even if seeds/knob tiers differ (p-value will not be trustworthy).",
    )
    p_compare.set_defaults(func=_cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.json:
        # Training/evaluation log to stdout via print(); redirect those to stderr so the
        # JSON blob is the ONLY thing on stdout and machine consumers can parse it directly.
        with contextlib.redirect_stdout(sys.stderr):
            payload = args.func(args)
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        payload = args.func(args)
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
