"""Shared reporting for paired (common-random-numbers) LBR comparisons.

Refusing unpaired evals, computing stats on per-hand differences, and printing
the standard report/verdict is evaluation protocol, not deploy orchestration.
It lives here — inside the layer contract and the test suite — so the Modal
entrypoints that drive it (``run_compare``, ``run_deployed_gate``) stay thin.
"""

from __future__ import annotations

from typing import Any

from src.pipeline.evaluation.statistics import compare_paired_samples


def print_variance_decomposition(results: dict[str, Any]) -> None:
    """Print the terminal-type variance decomposition of an LBR eval, if present."""
    decomposition = results.get("variance_decomposition")
    if not decomposition:
        return
    print("  variance by terminal type (within-group share of total):")
    for label, group in decomposition["groups"].items():
        print(
            f"    {label:>9}: {group['variance_share']:>5.1%} of variance "
            f"({group['n']} deals, {group['share_of_samples']:.1%})"
        )
    print(f"    (between-group: {decomposition['between_group_share']:.1%})")


def report_paired_lbr(
    arm_a: tuple[str, dict[str, Any]],
    arm_b: tuple[str, dict[str, Any]],
    *,
    diff_label: str,
    better_labels: tuple[str, str],
    show_pairing_gain: bool = False,
) -> dict[str, float]:
    """Print the standard paired-LBR report for two arms; return the comparison.

    ``arm_*`` is ``(title, results)`` where ``results`` is one eval payload's
    ``results`` dict. Refuses arms whose ``base_seed`` differs — without a
    common seed the per-hand differences do not cancel deal luck and the
    comparison is not paired. ``better_labels`` names the verdict winner:
    ``[0]`` when ``mean_diff > 0`` (arm B less exploitable), ``[1]`` otherwise.
    """
    title_a, results_a = arm_a
    title_b, results_b = arm_b
    if results_a["base_seed"] != results_b["base_seed"]:
        raise RuntimeError(
            f"base_seed mismatch ({results_a['base_seed']} vs {results_b['base_seed']}): "
            "the evals are not paired."
        )

    comparison = compare_paired_samples(
        results_a["pair_samples_mbb"], results_b["pair_samples_mbb"]
    )

    for title, results in ((title_a, results_a), (title_b, results_b)):
        print(f"\n{title}:")
        print(f"  {results['exploitability_mbb']:.1f} mbb/g (± {results['std_error_mbb']:.1f})")
        print_variance_decomposition(results)

    print(f"\nPAIRED DIFFERENCE ({diff_label}, {comparison['n']} common deals):")
    print(
        f"  {comparison['mean_diff']:.1f} mbb/g (± {comparison['se_diff']:.1f}; "
        f"95% CI [{comparison['ci_lower']:.1f}, {comparison['ci_upper']:.1f}])"
    )
    print(f"  p-value: {comparison['p_value']:.4f} | correlation: {comparison['correlation']:.3f}")
    if show_pairing_gain:
        print(
            f"  pairing gain: SE {comparison['se_diff']:.1f} vs {comparison['se_unpaired']:.1f} "
            f"unpaired ({comparison['se_unpaired'] / max(comparison['se_diff'], 1e-12):.1f}x tighter)"
        )
    if comparison["is_significant"]:
        better = better_labels[0] if comparison["mean_diff"] > 0 else better_labels[1]
        print(f"  VERDICT: {better} is significantly less exploitable (95% level).")
    else:
        print("  VERDICT: no significant difference at the 95% level.")
    return comparison
