"""
Statistical analysis for poker evaluation results.

Provides sample summaries, paired comparisons, and variance decomposition
for match results.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats


def summarize_samples(
    samples: Sequence[float],
    confidence_level: float = 0.95,
) -> dict:
    """One-sample summary: mean, SE, t-based CI, and t-test p-value against zero.

    The single home for the degenerate zero-variance convention (p = 1 when the
    mean is exactly 0, else 0) — callers must not re-implement it.

    Raises:
        ValueError: If fewer than two samples are given.
    """
    n = len(samples)
    if n < 2:
        raise ValueError("summarize_samples requires at least 2 samples.")

    values = np.asarray(samples, dtype=np.float64)
    mean = float(values.mean())
    se = float(stats.sem(values))
    t_value = float(stats.t.ppf((1 + confidence_level) / 2, n - 1))
    margin = t_value * se

    if se > 0:
        t_stat, p_value = stats.ttest_1samp(values, 0.0)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat = 0.0
        p_value = 1.0 if mean == 0.0 else 0.0

    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": bool(p_value < 1 - confidence_level),
    }


def compare_paired_samples(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    confidence_level: float = 0.95,
) -> dict:
    """Paired (common-random-numbers) comparison of two matched sample sequences.

    Both sequences must come from evaluations that shared their randomness index
    by index (e.g. two LBR evals with the same ``base_seed``: hand ``i`` sees the
    same deal in both). The per-index differences then cancel the shared luck, so
    the confidence interval on ``mean(a) - mean(b)`` is far tighter than what two
    independent intervals could resolve.

    Returns a dict with the paired difference (``mean_diff``, ``se_diff``,
    ``ci_lower``/``ci_upper``, ``t_statistic``, ``p_value``, ``is_significant``),
    the per-run means, the cross-run ``correlation``, and ``se_unpaired`` — the
    standard error an unpaired comparison would have had, so the variance
    reduction bought by pairing is visible in the output.

    Raises:
        ValueError: If lengths differ or fewer than two samples are given.
    """
    if len(samples_a) != len(samples_b):
        raise ValueError(
            f"Paired comparison requires equal-length samples (CRN hand-for-hand "
            f"matching); got {len(samples_a)} vs {len(samples_b)}."
        )
    n = len(samples_a)
    if n < 2:
        raise ValueError("Paired comparison requires at least 2 samples.")

    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    # A paired test IS the one-sample test on the per-index differences.
    diff_summary = summarize_samples(a - b, confidence_level)

    if float(a.std(ddof=1)) > 0 and float(b.std(ddof=1)) > 0:
        correlation = float(stats.pearsonr(a, b)[0])
    else:
        correlation = 0.0
    se_unpaired = float(np.sqrt(stats.sem(a) ** 2 + stats.sem(b) ** 2))

    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": diff_summary["mean"],
        "se_diff": diff_summary["se"],
        "ci_lower": diff_summary["ci_lower"],
        "ci_upper": diff_summary["ci_upper"],
        "t_statistic": diff_summary["t_statistic"],
        "p_value": diff_summary["p_value"],
        "is_significant": diff_summary["is_significant"],
        "correlation": correlation,
        "se_unpaired": se_unpaired,
    }


def variance_decomposition(
    samples: Sequence[float],
    groups: Sequence[str],
) -> dict:
    """Decompose sample variance by group label (law of total variance).

    For each group: sample count, share of samples, mean, std, and
    ``variance_share`` — the fraction of total (population) variance contributed
    by within-group spread. ``between_group_share`` covers the rest (group means
    differing from the grand mean). Shares sum to 1, which is why population
    (``ddof=0``) variances are used throughout.

    Used to attribute LBR eval noise to terminal types (fold / showdown / all-in)
    so variance-reduction work targets the component that actually dominates.

    Raises:
        ValueError: If lengths differ or no samples are given.
    """
    if len(samples) != len(groups):
        raise ValueError(f"samples ({len(samples)}) and groups ({len(groups)}) must align.")
    if not samples:
        raise ValueError("variance_decomposition requires at least one sample.")

    values = np.asarray(samples, dtype=np.float64)
    labels = np.asarray(groups)
    n = len(values)
    total_variance = float(values.var(ddof=0))

    group_stats: dict[str, dict] = {}
    between = 0.0
    grand_mean = float(values.mean())
    for label in sorted(set(labels.tolist())):
        member = values[labels == label]
        share = len(member) / n
        group_variance = float(member.var(ddof=0))
        within = share * group_variance
        between += share * (float(member.mean()) - grand_mean) ** 2
        group_stats[str(label)] = {
            "n": len(member),
            "share_of_samples": share,
            "mean": float(member.mean()),
            "std": float(member.std(ddof=0)),
            "variance_share": within / total_variance if total_variance > 0 else 0.0,
        }

    return {
        "total_variance": total_variance,
        "between_group_share": between / total_variance if total_variance > 0 else 0.0,
        "groups": group_stats,
    }
