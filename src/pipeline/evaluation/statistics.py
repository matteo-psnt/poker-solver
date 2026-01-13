"""
Statistical analysis for poker evaluation results.

Provides confidence intervals, significance tests, and variance calculations
for match results.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats


def compute_confidence_interval(
    values: list[float],
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute confidence interval for sample mean.

    Args:
        values: List of sample values
        confidence_level: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0

    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of mean

    # t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)

    margin_of_error = t_value * std_err
    lower = mean - margin_of_error
    upper = mean + margin_of_error

    return float(mean), float(lower), float(upper)


def compute_win_rate_confidence_interval(
    wins: int,
    total: int,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute confidence interval for win rate using Wilson score interval.

    Args:
        wins: Number of wins
        total: Total number of games
        confidence_level: Confidence level

    Returns:
        Tuple of (win_rate, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p = wins / total
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return p, lower, upper


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
    diffs = a - b

    mean_diff = float(diffs.mean())
    se_diff = float(stats.sem(diffs))
    t_value = float(stats.t.ppf((1 + confidence_level) / 2, n - 1))
    margin = t_value * se_diff

    if se_diff > 0:
        t_stat, p_value = stats.ttest_rel(a, b)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        # Identical difference on every sample: no within-pair noise at all.
        t_stat = 0.0
        p_value = 1.0 if mean_diff == 0.0 else 0.0

    if float(a.std(ddof=1)) > 0 and float(b.std(ddof=1)) > 0:
        correlation = float(stats.pearsonr(a, b)[0])
    else:
        correlation = 0.0
    se_unpaired = float(np.sqrt(stats.sem(a) ** 2 + stats.sem(b) ** 2))

    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_lower": mean_diff - margin,
        "ci_upper": mean_diff + margin,
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": bool(p_value < 1 - confidence_level),
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


def t_test_difference(
    values1: Sequence[float],
    values2: Sequence[float],
    confidence_level: float = 0.95,
) -> tuple[float, float, bool]:
    """
    Perform t-test to determine if two samples are significantly different.

    Args:
        values1: First sample
        values2: Second sample
        confidence_level: Confidence level

    Returns:
        Tuple of (t_statistic, p_value, is_significant)
    """
    if len(values1) < 2 or len(values2) < 2:
        return 0.0, 1.0, False

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(values1, values2)

    # Check if significant at given confidence level
    alpha = 1 - confidence_level
    is_significant = bool(p_value < alpha)

    return float(t_stat), float(p_value), is_significant


def compute_variance(values: list[float]) -> tuple[float, float]:
    """
    Compute variance and standard deviation.

    Args:
        values: List of values

    Returns:
        Tuple of (variance, std_dev)
    """
    if not values:
        return 0.0, 0.0

    variance = float(np.var(values, ddof=1))  # Sample variance
    std_dev = float(np.std(values, ddof=1))  # Sample std dev

    return variance, std_dev


def compute_percentiles(
    values: Sequence[float],
    percentiles: Sequence[int] = (25, 50, 75),
) -> list[float]:
    """
    Compute percentiles of sample.

    Args:
        values: List of values
        percentiles: Percentile ranks (0-100)

    Returns:
        List of percentile values
    """
    if not values:
        return [0.0] * len(percentiles)

    return [float(np.percentile(values, p)) for p in percentiles]


def estimate_required_sample_size(
    values: list[float],
    target_margin_of_error: float,
    confidence_level: float = 0.95,
) -> int:
    """
    Estimate required sample size for given margin of error.

    Args:
        values: Pilot sample values
        target_margin_of_error: Desired margin of error
        confidence_level: Confidence level

    Returns:
        Estimated required sample size
    """
    if not values:
        return 100  # Default

    std_dev = np.std(values, ddof=1)
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Formula: n = (z * sigma / margin)^2
    required_n = (z * std_dev / target_margin_of_error) ** 2

    return max(30, int(np.ceil(required_n)))  # Minimum 30 samples


class MatchStatisticsAnalyzer:
    """
    Analyzes statistics from head-to-head matches.

    Provides confidence intervals, significance tests, and variance analysis.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize analyzer.

        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.confidence_level = confidence_level

    def analyze_payoffs(
        self,
        payoffs: list[int],
        big_blind: int = 2,
    ) -> dict:
        """
        Analyze payoff distribution.

        Args:
            payoffs: List of payoffs
            big_blind: Big blind size for normalization

        Returns:
            Dictionary of statistics
        """
        if not payoffs:
            return {
                "mean": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "std_dev": 0.0,
                "variance": 0.0,
                "bb_per_hand": 0.0,
            }

        # Convert to bb/hand
        bb_payoffs = [p / big_blind for p in payoffs]

        mean, ci_lower, ci_upper = compute_confidence_interval(bb_payoffs, self.confidence_level)
        variance, std_dev = compute_variance(bb_payoffs)

        return {
            "mean": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std_dev": std_dev,
            "variance": variance,
            "bb_per_hand": mean,
        }

    def compare_strategies(
        self,
        payoffs1: list[int],
        payoffs2: list[int],
    ) -> dict:
        """
        Compare two strategies statistically.

        Args:
            payoffs1: Payoffs for first strategy
            payoffs2: Payoffs for second strategy

        Returns:
            Dictionary of comparison statistics
        """
        if not payoffs1 or not payoffs2:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "mean_difference": 0.0,
            }

        t_stat, p_value, is_sig = t_test_difference(payoffs1, payoffs2, self.confidence_level)

        mean_diff = np.mean(payoffs1) - np.mean(payoffs2)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_sig,
            "mean_difference": mean_diff,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"MatchStatisticsAnalyzer(confidence={self.confidence_level})"
