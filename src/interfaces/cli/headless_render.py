"""Human-readable rendering of the headless entrypoint's result payloads.

Every command returns the same payload it would emit under ``--json``; this
module is the only thing that turns one into text. Keeping it apart from
:mod:`~src.interfaces.cli.headless` keeps that module about *what runs* and
this one about *what it looks like*, and makes the JSON contract the single
source both agree on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.pipeline import services


def _fmt_commit(commit: str | None, dirty: bool | None) -> str:
    if not commit:
        return "—"
    short = commit[:7]
    if dirty:
        short += "-dirty"
    return short


def _render_train(payload: dict[str, Any]) -> None:
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


def _render_resume(payload: dict[str, Any]) -> None:
    if payload["no_op"]:
        print(
            f"Nothing to do: {payload['run_id']} is at "
            f"{payload['resumed_from_iteration']:,}, target was "
            f"{payload['target_iteration']:,}."
        )
        return
    print("Resume complete.")
    print(f"  Run ID:      {payload['run_id']}  (under {payload['runs_dir']})")
    print(
        f"  Iterations:  {payload['resumed_from_iteration']:,} -> "
        f"{payload['iterations']:,}  (target {payload['target_iteration']:,})"
    )
    print(f"  Infosets:    {payload['num_infosets']:,}")
    print(f"  Status:      {payload['status']}")


def _render_precompute(payload: dict[str, Any]) -> None:
    print("Precompute complete.")
    print(f"  Abstraction: {payload['abstraction_config']}")
    print(f"  Output:      {payload['output_dir']}")


def _render_promote(payload: dict[str, Any]) -> None:
    print(f"Baseline is now {payload['run_id']}")
    if payload["checkpoint_iteration"] is not None:
        print(f"  Checkpoint:  {payload['checkpoint_iteration']:,}")
    print(f"  Rationale:   {payload['rationale']}")
    print(f"  Recorded in: {payload['baseline']}")


def _render_curve(payload: dict[str, Any]) -> None:
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


def _render_report(payload: dict[str, Any]) -> None:
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


def _render_ledger(payload: dict[str, Any]) -> None:
    # Printed BEFORE the empty-rows early return: `just fetch` runs `--rebuild`
    # without --json, and the recovery counts are the entire point of that call.
    # A rebuild that found nothing and one that recovered 200 rows must not look
    # identical.
    rebuilt = payload.get("rebuilt")
    if rebuilt:
        print(
            f"Rebuilt {payload['ledger']}: {rebuilt['recovered']} row(s) recovered "
            f"from per-run records, {rebuilt['preserved']} preserved (no record to "
            "rebuild from — pre-dating per-run records)."
        )

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


def _render_compare(payload: dict[str, Any]) -> None:
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


def _render_evaluate(payload: dict[str, Any]) -> None:
    results = payload["results"]
    print("Evaluation complete.")
    print(f"  Run ID:        {payload['run_id']}")
    print(f"  Estimator:     {payload['estimator']}")
    print(f"  Infosets:      {payload['infosets']:,}")
    print(
        f"  Exploitability: {results['exploitability_mbb']:.2f} mbb/g "
        f"(± {results['std_error_mbb']:.2f})"
    )


def _render_checkpoint_profile(payload: dict[str, Any]) -> None:
    print(f"Checkpoint profile for {payload['run']}")
    print(f"  Checkpoints: {payload['num_checkpoints']}")
    print(f"  Writing:     {payload['checkpoint_seconds']:.2f}s")
    share = payload["commit_share"]
    commit = f"  Committing:  {payload['volume_commit_seconds']:.2f}s"
    print(commit if share is None else f"{commit}  ({share:.1%} of total)")
    print(f"  Total:       {payload['total_seconds']:.2f}s")
    for label, phases in (
        ("Top-level", payload["top_level_phases"]),
        ("Write phases", payload["write_phases"]),
    ):
        if phases:
            print(f"  {label}:")
            for name, secs in phases.items():
                print(f"    {name:<24} {secs:>8.2f}s")


def _render_train_static(payload: dict[str, Any]) -> None:
    print("Static-tree training complete.")
    print(f"  Run ID:      {payload['run_id']}  (under {payload['runs_dir']})")
    print(f"  Config:      {payload['config_name']}")
    print(f"  Iterations:  {payload['iterations']:,}")
    # Coverage is what only this path can report: the table size is known up
    # front, so "how much of the tree did we actually touch" is answerable.
    print(
        f"  Coverage:    {payload['touched_rows']:,} / {payload['num_rows']:,} rows "
        f"({payload['coverage']:.1%})"
    )
    print(f"  Visits/row:  {payload['mean_visits_per_touched']:.1f} mean, on touched rows")
    print(
        f"  Runtime:     {payload['runtime_seconds']:.2f}s "
        f"({payload['iterations_per_second']:.1f} it/s)"
    )
    if payload["dropped_updates"]:
        print(f"  Dropped:     {payload['dropped_updates']:,} updates")
    print(f"  Status:      {payload['status']}")


RENDERERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "train": _render_train,
    "train-static": _render_train_static,
    "resume": _render_resume,
    "precompute": _render_precompute,
    "promote": _render_promote,
    "curve": _render_curve,
    "report": _render_report,
    "ledger": _render_ledger,
    "compare": _render_compare,
    "evaluate": _render_evaluate,
    "checkpoint-profile": _render_checkpoint_profile,
}


def print_human(payload: dict[str, Any]) -> None:
    """Render a result payload for a terminal.

    A table rather than an if/elif chain so that adding a command cannot again
    silently inherit another command's renderer: ``checkpoint-profile`` used to
    fall through to the evaluate branch and die on a missing ``results`` key.
    An op with no entry now says so instead of raising.
    """
    render = RENDERERS.get(payload["op"])
    if render is None:
        print(f"(no human-readable renderer for '{payload['op']}'; re-run with --json)")
        return
    render(payload)
