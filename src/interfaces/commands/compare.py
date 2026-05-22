"""The `compare` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.interfaces.commands._base import (
    Command,
    ledger_for,
    records_root,
)
from src.interfaces.errors import CommandError
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.statistics import compare_paired_samples


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver compare`."""
    parser.add_argument("--a", required=True, help="First run id (baseline).")
    parser.add_argument("--b", required=True, help="Second run id (candidate).")
    parser.add_argument(
        "--a-at",
        type=int,
        default=None,
        help=(
            "Checkpoint iteration to select for --a. Needed when a run has been "
            "evaluated at more than one checkpoint; otherwise the newest row wins."
        ),
    )
    parser.add_argument(
        "--b-at",
        type=int,
        default=None,
        help="Checkpoint iteration to select for --b (see --a-at).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compare even if seeds/knob tiers differ (p-value will not be trustworthy).",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Paired (common-random-numbers) comparison of two runs' latest evals."""
    with records_root(args) as root:
        return _compare(args, root)


def _compare(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    ledger_path = ledger_for(root)
    rec_a = eval_ledger.latest_record_for_run(args.a, ledger_path, args.a_at)
    rec_b = eval_ledger.latest_record_for_run(args.b, ledger_path, args.b_at)
    if rec_a is None or rec_b is None:
        missing, at = (args.a, args.a_at) if rec_a is None else (args.b, args.b_at)
        at_note = f" at checkpoint iteration {at}" if at is not None else ""
        raise CommandError(f"No ledger entry found for run '{missing}'{at_note} in {ledger_path}")

    reasons = eval_ledger.tier_mismatches(rec_a, rec_b)
    if reasons and not args.force:
        joined = "\n".join(f"  - {r}" for r in reasons)
        raise CommandError(
            "Refusing to compare: the two evals are not a valid paired comparison:\n"
            f"{joined}\n"
            "Re-run both evals with matching knobs and the same --seed, or pass --force "
            "to override (the resulting p-value will not be trustworthy)."
        )

    runs_dir = root
    payload_a = eval_ledger.load_payload(rec_a, runs_dir)
    payload_b = eval_ledger.load_payload(rec_b, runs_dir)

    # Checked AFTER --force, deliberately: --force overrides a judgement about
    # whether a comparison is meaningful, but it cannot conjure per-hand samples
    # that were never recorded. exact_br is deterministic and stores none, so the
    # forced path would otherwise die on a bare KeyError.
    samples_a = payload_a["results"].get("pair_samples_mbb")
    samples_b = payload_b["results"].get("pair_samples_mbb")
    if not samples_a or not samples_b:
        missing, rec_missing = (args.a, rec_a) if not samples_a else (args.b, rec_b)
        raise CommandError(
            f"Cannot pair: the eval for '{missing}' recorded no per-hand samples "
            f"(method '{rec_missing.get('method')}'). Deterministic estimators like "
            "exact_br have nothing to pair — within a matched board tier compare "
            "their exploitability_mbb directly; no p-value applies. --force does "
            "not help here."
        )

    comparison = compare_paired_samples(samples_a, samples_b)
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


def render(payload: dict[str, Any]) -> None:
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


COMMAND = Command(
    name="compare",
    help="Paired (CRN) comparison of two runs' latest evals; refuses mismatched tiers.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
