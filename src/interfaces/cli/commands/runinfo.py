"""The `runinfo` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands._base import (
    Command,
    add_source_argument,
    ledger_for,
    records_root,
    resolve_run_dir,
)
from src.pipeline import services
from src.pipeline.evaluation import ledger as eval_ledger


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run runinfo`."""
    add_source_argument(parser)
    parser.add_argument("--run", required=True, help="Run id (dir name) or path.")
    parser.add_argument(
        "--runs-dir", default="data/runs", help="Directory containing run directories."
    )
    parser.add_argument("--ledger", default=str(eval_ledger.DEFAULT_LEDGER_PATH))
    parser.add_argument(
        "--tier", type=int, default=0, help="Which comparison tier's curve to show."
    )
    parser.add_argument(
        "--legs-dir",
        default=None,
        help=(
            "Directory CONTAINING a local legs/ -- `--legs-dir data`, not "
            "`data/legs` (see `just fetch`). Omit for a purely local run."
        ),
    )
    parser.add_argument("--last", type=int, default=8, help="Checkpoints to show (0 = all).")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Everything recorded about one run, joined into one view.

    The evidence is spread across artifacts written by four subsystems; this is
    the one place that holds the joins so you do not have to.
    """
    with records_root(args) as root:
        digest = services.run_digest(
            resolve_run_dir(args.run, str(root)),
            ledger_path=ledger_for(args, root),
            tier_index=args.tier,
            legs_dir=Path(args.legs_dir) if args.legs_dir else None,
        )
    payload = dataclasses.asdict(digest)
    payload["op"] = "runinfo"
    # Trimmed to the tail: a 30M run has thirty checkpoints and the reader wants
    # the shape, not the log.
    payload["progress"] = digest.progress[-args.last :] if args.last > 0 else digest.progress
    payload["total_progress_rows"] = len(digest.progress)
    return payload


def render(payload: dict[str, Any]) -> None:
    tag = ""
    if payload.get("experiment_id"):
        tag = f"  {payload['experiment_id']}/{payload.get('arm') or '-'}"
    print(f"{payload['run_id']}  {payload['config_name']}{tag}")

    commit = (payload.get("git_commit") or "unknown")[:8]
    dirty = " (dirty)" if payload.get("git_dirty") else ""
    abstraction = (payload.get("card_abstraction_hash") or "none")[:16]
    print(f"  git {commit}{dirty}   abstraction {abstraction}   status {payload['status']}")
    print(
        f"  {payload['iterations']:,} iterations over {payload['attempts']} attempt(s), "
        f"{payload['runtime_seconds']:.0f}s compute"
    )

    rows = payload.get("progress") or []
    if rows:
        total = payload.get("total_progress_rows", len(rows))
        print(f"\n  progress  ({total} checkpoints, last {len(rows)})")
        print(f"    {'iteration':>12}{'coverage':>10}{'visits':>9}{'it/s':>8}")
        for row in rows:
            print(
                f"    {row.get('iteration', 0):>12,}"
                f"{_pct(row.get('coverage')):>10}"
                f"{_num(row.get('mean_visits_per_touched'), '{:.1f}'):>9}"
                f"{_num(row.get('iters_per_sec'), '{:.0f}'):>8}"
            )
        flat = payload.get("coverage_flat_from")
        if flat is not None:
            print(f"    coverage flat from {flat:,}")

    curve = payload.get("curve") or {}
    points = curve.get("points") or []
    if points:
        print(f"\n  curve  ({curve.get('tier') or 'untiered'})")
        for point in points:
            print(
                f"    {point['iteration']:>12,}  {point['exploitability_mbb']:>9.1f} mbb"
                f"  (± {point['std_error_mbb']:.1f})"
            )

    legs = payload.get("legs") or []
    if legs:
        print(f"\n  legs  ({len(legs)})")
        for leg in legs:
            attempt = f"#{leg.get('attempt', 1)}"
            print(f"    {leg['task_id']:<28} {attempt:<4} {leg['cause']}")

    gaps = payload.get("gaps") or []
    print("\n  gaps" if gaps else "\n  no gaps: scored, complete, reproducible")
    for gap in gaps:
        print(f"    - {gap}")


def _pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, int | float) else ""


def _num(value: Any, fmt: str) -> str:
    return fmt.format(value) if isinstance(value, int | float) else ""


COMMAND = Command(
    name="runinfo",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Everything recorded about a run: provenance, curve, scores, legs, gaps.",
)
