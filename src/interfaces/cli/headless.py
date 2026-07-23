"""Non-interactive (headless) entrypoint for training and evaluation.

Unlike the questionary menu in :mod:`src.interfaces.cli.app`, every operation here
is fully specified by CLI flags and emits a machine-readable summary. This is the
surface used by scripts and cloud (Modal) execution — where an interactive prompt is
not an option.

Cloud callers should prefer importing :func:`src.pipeline.services.train`
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

from src.pipeline import services
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.hunl_local_best_response import LBRConfig
from src.pipeline.evaluation.statistics import compare_paired_samples
from src.pipeline.services import RolloutParams
from src.shared import checkpoint_profile
from src.shared.jsonio import json_default
from src.shared.log import configure_logging


def _write_result(run_dir: Path, payload: dict[str, Any]) -> None:
    """Persist a per-operation result file (e.g. ``train_result.json``) in the run dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{payload['op']}_result.json").write_text(
        json.dumps(payload, indent=2, default=json_default)
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
    """Argparse transport around :func:`services.evaluate_and_record`.

    All dispatch, payload shaping, and ledger recording live in the orchestrator;
    this function only maps flags onto the params objects. The orchestrator's
    ledger warning prints to stdout, which under ``--json`` is redirected to
    stderr — keeping the machine-readable payload clean.
    """
    run_dir = _resolve_run_dir(args.run, args.runs_dir)
    payload = services.evaluate_and_record(
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
        rollout=RolloutParams(
            num_samples=args.samples,
            num_rollouts=args.rollouts,
            use_average_strategy=not args.current,
            seed=args.seed,
        ),
        resolver_iterations=args.resolver_iterations,
        abstraction_hash=args.abstraction_hash,
        ledger_path=Path(args.ledger),
    )
    _write_result(run_dir, payload)
    return payload


def _cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    """List recent eval-ledger rows as a compact table."""
    records = eval_ledger.read_records(Path(args.ledger))
    if args.run:
        records = [r for r in records if r.get("run_id") == args.run]
    records = records[-args.limit :]
    return {"op": "ledger", "ledger": str(args.ledger), "rows": records}


def _cmd_checkpoint_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Summarize a run's per-checkpoint phase timings and the Volume commit."""
    run_dir = Path(args.runs_dir) / args.run
    path = run_dir / checkpoint_profile.PROFILE_FILENAME
    if not path.exists():
        raise SystemExit(
            f"No checkpoint profile at {path}. It is written per checkpoint, so the "
            "run must have checkpointed at least once with profiling in place."
        )

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    checkpoints = [r for r in rows if r.get("event") != "volume_commit"]
    commits = [r for r in rows if r.get("event") == "volume_commit"]

    phase_totals: dict[str, float] = {}
    for row in checkpoints:
        for name, secs in row.get("phases", {}).items():
            phase_totals[name] = phase_totals.get(name, 0.0) + secs

    checkpoint_seconds = sum(r["total_seconds"] for r in checkpoints)
    commit_seconds = sum(r["total_seconds"] for r in commits)
    # storage_write wraps the engine-level phases, so counting it alongside them
    # would double-count; collect_keys and storage_write are the top-level split.
    top_level = {k: v for k, v in phase_totals.items() if k in ("collect_keys", "storage_write")}

    return {
        "op": "checkpoint-profile",
        "run": args.run,
        "num_checkpoints": len(checkpoints),
        "checkpoint_seconds": round(checkpoint_seconds, 2),
        "volume_commit_seconds": round(commit_seconds, 2),
        "total_seconds": round(checkpoint_seconds + commit_seconds, 2),
        "commit_share": (
            round(commit_seconds / (checkpoint_seconds + commit_seconds), 3)
            if checkpoint_seconds + commit_seconds > 0
            else None
        ),
        "top_level_phases": {k: round(v, 2) for k, v in sorted(top_level.items())},
        "write_phases": {
            k: round(v, 2)
            for k, v in sorted(phase_totals.items(), key=lambda kv: -kv[1])
            if k not in ("collect_keys", "storage_write")
        },
        "checkpoints": checkpoints,
        "volume_commits": commits,
    }


def _cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Paired (common-random-numbers) comparison of two runs' latest evals."""
    ledger_path = Path(args.ledger)
    rec_a = eval_ledger.latest_record_for_run(args.a, ledger_path, args.a_at)
    rec_b = eval_ledger.latest_record_for_run(args.b, ledger_path, args.b_at)
    if rec_a is None or rec_b is None:
        missing, at = (args.a, args.a_at) if rec_a is None else (args.b, args.b_at)
        at_note = f" at checkpoint iteration {at}" if at is not None else ""
        raise SystemExit(f"No ledger entry found for run '{missing}'{at_note} in {ledger_path}")

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
        # Which checkpoints were actually compared: a run id alone does not say.
        "checkpoint_iteration_a": rec_a.get("checkpoint_iteration"),
        "checkpoint_iteration_b": rec_b.get("checkpoint_iteration"),
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
        "--allin-runouts",
        type=int,
        default=50,
        help="[lbr] Board runouts averaged at all-in showdown terminals "
        "(variance reduction; same expectation).",
    )
    p_eval.add_argument(
        "--abstraction-hash",
        default=None,
        help="Pin the card abstraction to this hash (see the abstraction's metadata.json "
        "'config_hash'). Default: the hash recorded on the run.",
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

    p_profile = sub.add_parser(
        "checkpoint-profile",
        parents=[common],
        help="Per-checkpoint phase timings and Volume-commit cost for a run.",
    )
    p_profile.add_argument("--run", required=True, help="Run id to summarize.")
    p_profile.add_argument(
        "--runs-dir", default="data/runs", help="Directory containing run directories."
    )
    p_profile.set_defaults(func=_cmd_checkpoint_profile)

    p_compare = sub.add_parser(
        "compare",
        parents=[common],
        help="Paired (CRN) comparison of two runs' latest evals; refuses mismatched tiers.",
    )
    p_compare.add_argument("--a", required=True, help="First run id (baseline).")
    p_compare.add_argument("--b", required=True, help="Second run id (candidate).")
    p_compare.add_argument(
        "--a-at",
        type=int,
        default=None,
        help=(
            "Checkpoint iteration to select for --a. Needed when a run has been "
            "evaluated at more than one checkpoint; otherwise the newest row wins."
        ),
    )
    p_compare.add_argument(
        "--b-at",
        type=int,
        default=None,
        help="Checkpoint iteration to select for --b (see --a-at).",
    )
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
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.json:
        # Library layers log to stderr, but third-party writers (numba, zarr) can still
        # print to stdout; redirect so the JSON blob is the ONLY thing on stdout and
        # machine consumers can parse it directly.
        with contextlib.redirect_stdout(sys.stderr):
            payload = args.func(args)
        print(json.dumps(payload, indent=2, default=json_default))
    else:
        payload = args.func(args)
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
