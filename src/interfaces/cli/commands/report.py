"""The `report` subcommand: its flags, handler and renderer."""

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
)
from src.pipeline import services
from src.pipeline.evaluation import ledger as eval_ledger
from src.shared.config import DEFAULT_RUNS_DIR


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run report`."""
    add_source_argument(parser)
    parser.add_argument("--experiment", required=True, help="Experiment id to report on.")
    parser.add_argument(
        "--runs-dir", default=DEFAULT_RUNS_DIR, help="Runs dir, for resolving eval payloads."
    )
    parser.add_argument(
        "--ledger", default=str(eval_ledger.DEFAULT_LEDGER_PATH), help="Eval ledger path."
    )
    parser.add_argument(
        "--baseline", default=str(services.DEFAULT_BASELINE_PATH), help="Baseline pointer file."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.experiment_report`."""
    with records_root(args) as root:
        # The baseline travels with the record under --source share: it is the
        # conclusion of the experiment this command reports on.
        baseline = root / "baseline.json" if args.source == "share" else Path(args.baseline)
        out = services.experiment_report(
            args.experiment,
            ledger_path=ledger_for(args, root),
            runs_dir=root,
            baseline_path=baseline,
        )
    return {"op": "report", **dataclasses.asdict(out)}


def render(payload: dict[str, Any]) -> None:
    print(f"Experiment {payload['experiment_id']}")
    if payload["baseline_run_id"]:
        print(f"  Baseline: {payload['baseline_run_id']}")
    for note in payload["notes"]:
        print(f"  ! {note}")
    if not payload["arms"]:
        return

    print(f"  {'arm':<24} {'mbb/g':>9} {'± se':>8} {'vs control':>12} {'p':>8}")
    for arm in payload["arms"]:
        delta = arm["vs_control_mbb"]
        p_value = arm["vs_control_p_value"]
        # Lower exploitability is better, so a negative delta is the idea helping.
        delta_col = "—" if delta is None else f"{delta:+.1f}"
        p_col = "—" if p_value is None else f"{p_value:.3f}"
        if arm["arm"] == services.CONTROL_ARM:
            delta_col, p_col = "(control)", ""
        print(
            f"  {arm['arm']:<24} {arm['exploitability_mbb']:>9.1f} "
            f"{arm['std_error_mbb']:>8.1f} {delta_col:>12} {p_col:>8}"
        )
        for reason in arm["vs_control_blocked"]:
            print(f"      not attributable: {reason}")
    print("  (vs control is variant − control; negative = less exploitable = better)")


COMMAND = Command(
    name="report",
    help="Score every arm of an experiment, each attributed against its control.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
