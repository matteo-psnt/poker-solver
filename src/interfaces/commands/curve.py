"""The `curve` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any

from src.interfaces.commands._base import (
    Command,
    ledger_for,
    records_root,
    resolve_run_dir,
)
from src.pipeline import services


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver curve`."""
    parser.add_argument("--run", required=True, help="Run id (dir name) or path to a run dir.")
    parser.add_argument(
        "--tier",
        type=int,
        default=0,
        help="Which comparison tier to plot when a run was scored by more than one "
        "(0 = best-covered). Tiers are never merged — see the listing in the output.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.exploitability_curve`."""
    with records_root(args) as root:
        out = services.exploitability_curve(
            resolve_run_dir(args.run, str(root)),
            ledger_path=ledger_for(root),
            tier_index=args.tier,
        )
    return {"op": "curve", "decay_ratio": out.decay_ratio, **dataclasses.asdict(out)}


def render(payload: dict[str, Any]) -> None:
    points = payload["points"]
    print(f"Convergence curve for {payload['run_id']}")
    if not points:
        print("  No placeable evaluations for this run.")
        if payload["unplaceable_records"]:
            print(
                f"  {payload['unplaceable_records']} recorded eval(s) carry no "
                "checkpoint_iteration (pre-provenance) — they cannot be placed on an axis."
            )
        if payload["retained_iterations"]:
            rungs = ", ".join(f"{i:,}" for i in payload["retained_iterations"])
            print(f"  Ladder on disk: {rungs}")
            print("  Score them with: evaluate --run <id> --at <iteration>")
        else:
            print(
                "  No retained checkpoint ladder either — train with "
                "storage.checkpoint_retain_every set to build one."
            )
        return

    print(f"  Tier: {payload['tier']}")
    print(f"  {'iteration':>12}  {'mbb/g':>10}  {'± se':>8}  {'hands':>8}")
    for point in points:
        print(
            f"  {point['iteration']:>12,}  {point['exploitability_mbb']:>10.1f}  "
            f"{point['std_error_mbb']:>8.1f}  {point['num_hands']:>8,}"
        )

    if payload["decay_ratio"] is not None:
        first, last = points[0], points[-1]
        budget_ratio = last["iteration"] / first["iteration"] if first["iteration"] else 0
        print(
            f"  Decay:       {payload['decay_ratio']:.2f}x over {budget_ratio:.0f}x budget "
            f"(O(1/sqrt(T)) predicts ~{budget_ratio**0.5:.2f}x)"
        )
    if payload["missing_iterations"]:
        gaps = ", ".join(f"{i:,}" for i in payload["missing_iterations"])
        print(f"  Unscored rungs: {gaps}")
    for other in payload["other_tiers"]:
        print(f"  (also recorded, not mixed in: {other})")


COMMAND = Command(
    name="curve",
    help="Within-run exploitability vs iteration, from the retained checkpoint ladder.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
