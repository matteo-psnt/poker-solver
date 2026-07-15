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
            seed=args.seed,
            num_workers=args.workers,
            opponent=args.opponent,
            resolver_iterations=args.resolver_iterations,
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
    return payload


def _print_human(payload: dict[str, Any]) -> None:
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
        "--opponent",
        choices=["blueprint", "deployed"],
        default="blueprint",
        help="[lbr] Strategy under measurement: raw table, or blueprint+resolver as deployed.",
    )
    p_eval.add_argument(
        "--resolver-iterations",
        type=int,
        default=32,
        help="[lbr] Pinned subgame-CFR iterations per deployed-opponent solve.",
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
