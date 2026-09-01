"""The `arms` subcommand: one experiment's arms, side by side, per instrument.

`--experiment`/`--arm` is the dispatch backbone -- 55 of 60 submits carry it --
and until now nothing read it back. The `report` command that used to attributed
arms only through `compare_paired_samples`, which REFUSES `exact_br` rows: they
carry no per-hand samples, so every exact_br A/B in this project has been
subtracted by hand. It does not need pairing. The estimator is deterministic,
so two arms in a matched tier differ by exactly the number printed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command, CommandError, ledger_for, records_root
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.services.experiments import ArmsOutput, experiment_arms

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver arms`."""
    parser.add_argument("--experiment", required=True, help="Experiment id to report on.")
    parser.add_argument(
        "--control",
        default=None,
        help="Arm every other arm is differenced against, at matched iterations.",
    )


class ArmsPayload(BaseModel):
    """What `arms` answers. The console can read this unchanged."""

    op: Literal["arms"] = "arms"
    ledger: str
    result: ArmsOutput


def run(args: argparse.Namespace) -> ArmsPayload:
    """Read the published record and group one experiment's evals by tier and arm."""
    with records_root(args) as root:
        ledger_path = ledger_for(root)
        records = eval_ledger.read_records(ledger_path)
        result = experiment_arms(records, args.experiment, control=args.control)
    if not result.tiers:
        known = sorted({str(r["experiment_id"]) for r in records if r.get("experiment_id")})
        raise CommandError(
            f"No scored evaluations tagged experiment={args.experiment!r}. "
            f"Recorded experiments: {', '.join(known) or 'none'}."
        )
    if args.control and all(tier.control is None for tier in result.tiers):
        raise CommandError(
            f"No arm named {args.control!r} in this experiment. "
            f"Arms: {', '.join(sorted({a for t in result.tiers for a in t.arms}))}."
        )
    return ArmsPayload(ledger=str(ledger_path), result=result)


def render(payload: ArmsPayload) -> None:
    result = payload.result
    print(f"experiment {result.experiment_id}: {len(result.tiers)} tier(s)")
    for index, tier in enumerate(result.tiers):
        print(f"\n[{index}] {tier.tier}")
        head = f"  {'iteration':>10}  {'arm':<24}{'mbb/hand':>12}"
        print(head + (f"{'vs ' + tier.control:>16}" if tier.control else ""))
        for point in tier.points:
            error = f" ±{point.std_error_mbb:.1f}" if point.std_error_mbb else ""
            line = (
                f"  {point.iteration:>10,}  {point.arm:<24}{point.exploitability_mbb:>12.1f}{error}"
            )
            if tier.control and point.vs_control_mbb is not None:
                # Signed and explicit: a bare number here reads as a level.
                # Negative is LESS exploitable, which is the arm that won.
                combined = (
                    f" ±{point.vs_control_stderr_mbb:.1f}" if point.vs_control_stderr_mbb else ""
                )
                line += f"{point.vs_control_mbb:>+16.1f}{combined}"
            print(line)
        if tier.unmatched_iterations:
            rungs = ", ".join(f"{i:,}" for i in tier.unmatched_iterations[:6])
            print(f"  not every arm is scored at: {rungs} — those rows compare unequal training")
    if result.unplaceable_records:
        print(
            f"\n  {result.unplaceable_records} row(s) carry no checkpoint iteration "
            "and cannot be placed against an arm."
        )


COMMAND = Command(
    name="arms",
    help="Compare an experiment's arms, grouped by the instrument each was measured with.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
