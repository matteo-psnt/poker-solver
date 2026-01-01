"""
Statistical analysis for poker evaluation results.

Provides confidence intervals, significance tests, and variance calculations
for match results.
"""

from typing import Sequence

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
    margin = z * np.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return p, lower, upper


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
