"""The `ab` subcommand: its flags, handler and renderer.

A paired knob A/B. Every arm trains single-worker at one seed and is scored with
``exact_br``, so a difference between arms is attributable to the knob and
nothing else — see :mod:`src.pipeline.services.ab` for why those two conditions
are the whole point and why neither is optional.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.cli.commands._base import Command, parse_overrides
from src.pipeline import services


def _parse_arm(spec: str) -> services.Arm:
    """Parse ``name:key=value,key=value`` into an :class:`~services.Arm`.

    Values go through the same JSON coercion as ``--set``, so ``true``/``110.0``
    arrive as the types the strict config models require.
    """
    name, sep, overrides = spec.partition(":")
    if not sep or not name.strip():
        raise SystemExit(f"--arm expects NAME:key=value[,key=value], got '{spec}'")
    pairs = [p for p in overrides.split(",") if p.strip()]
    if not pairs:
        raise SystemExit(f"--arm '{name}' has no overrides; an arm with none is the control")
    return services.Arm(name.strip(), parse_overrides(pairs))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run ab`."""
    parser.add_argument("--config", required=True, help="Config stem under config/training/.")
    parser.add_argument(
        "--iterations", type=int, required=True, help="Iterations per arm (all arms equal)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Training seed, shared by every arm. Required: without a fixed seed the "
            "arms are not comparable."
        ),
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME:key=value[,key=value]",
        help=(
            "An arm to compare against the control, e.g. --arm "
            "'prune110:solver__enable_pruning=true,solver__pruning_threshold=110.0'. "
            "Repeatable. The control is generated automatically."
        ),
    )
    parser.add_argument("--runs-dir", default="data/runs", help="Base runs dir.")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help=(
            "Train and score the control TWICE and require an exact match before "
            "trusting the arms. Roughly doubles the control's cost; worth it the "
            "first time a config, machine, or code version is used."
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run a paired knob A/B; every arm is single-worker at one seed."""
    from pathlib import Path

    arms = [_parse_arm(spec) for spec in args.arm]
    result = services.run_ab(
        args.config,
        arms,
        iterations=args.iterations,
        seed=args.seed,
        runs_dir=Path(args.runs_dir),
        verify_determinism=args.verify_determinism,
    )
    return {
        "op": "ab",
        "config_name": result.config_name,
        "iterations": result.iterations,
        "seed": result.seed,
        "determinism_verified": result.determinism_verified,
        # Preformatted by the service so the CLI and a library caller cannot
        # drift in how the same comparison reads.
        "table": services.format_ab_table(result),
        "arms": [
            {
                "name": a.name,
                "run_id": a.run_id,
                "overrides": a.overrides,
                "iterations": a.iterations,
                "touched_rows": a.touched_rows,
                "num_rows": a.num_rows,
                "coverage": a.coverage,
                "runtime_seconds": a.runtime_seconds,
                "exploitability_mbb": a.exploitability_mbb,
            }
            for a in result.arms
        ],
    }


def render(payload: dict[str, Any]) -> None:
    print(payload["table"])


COMMAND = Command(
    name="ab",
    help=(
        "Paired knob A/B: trains a control plus each --arm single-worker at one "
        "seed and scores every one with exact_br, so a difference is attributable "
        "to the knob and nothing else."
    ),
    add_arguments=add_arguments,
    run=run,
    render=render,
)
