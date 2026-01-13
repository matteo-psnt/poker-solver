"""Tests for statistical analysis."""

import numpy as np
import pytest

from src.pipeline.evaluation.statistics import (
    MatchStatisticsAnalyzer,
    compare_paired_samples,
    compute_confidence_interval,
    compute_percentiles,
    compute_variance,
    compute_win_rate_confidence_interval,
    estimate_required_sample_size,
    t_test_difference,
    variance_decomposition,
)


class TestComparePairedSamples:
    """Paired CRN comparison: the difference CI must beat the unpaired one."""

    def test_constant_shift_is_certain(self):
        a = [1.0, 5.0, 3.0, 9.0, 2.0]
        b = [x - 2.0 for x in a]

        result = compare_paired_samples(a, b)

        assert result["mean_diff"] == pytest.approx(2.0)
        assert result["se_diff"] == 0.0
        assert result["p_value"] == 0.0
        assert result["is_significant"]
        assert result["ci_lower"] == result["ci_upper"] == pytest.approx(2.0)

    def test_identical_samples_are_null(self):
        a = [1.0, 5.0, 3.0]

        result = compare_paired_samples(a, list(a))

        assert result["mean_diff"] == 0.0
        assert result["se_diff"] == 0.0
        assert result["p_value"] == 1.0
        assert not result["is_significant"]

    def test_pairing_beats_unpaired_on_correlated_samples(self):
        # Shared per-index "luck" dominates; the paired difference cancels it.
        rng = np.random.default_rng(0)
        luck = rng.normal(0.0, 10.0, size=200)
        a = luck + rng.normal(1.0, 1.0, size=200)
        b = luck + rng.normal(0.0, 1.0, size=200)

        result = compare_paired_samples(a.tolist(), b.tolist())

        assert result["correlation"] > 0.9
        assert result["se_diff"] < result["se_unpaired"] / 3
        # The ~1.0 gap is invisible to unpaired CIs at this noise level but
        # decisively resolved by the paired test.
        assert result["is_significant"]
        assert result["ci_lower"] < 1.0 < result["ci_upper"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            compare_paired_samples([1.0, 2.0], [1.0])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compare_paired_samples([1.0], [2.0])


class TestVarianceDecomposition:
    """Within/between-group shares must follow the law of total variance."""

    def test_shares_sum_to_one(self):
        samples = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 0.0, 0.0]
        groups = ["a", "a", "a", "b", "b", "b", "c", "c"]

        result = variance_decomposition(samples, groups)

        within = sum(g["variance_share"] for g in result["groups"].values())
        assert within + result["between_group_share"] == pytest.approx(1.0)
        assert result["groups"]["c"]["variance_share"] == 0.0
        assert result["groups"]["b"]["variance_share"] > result["groups"]["a"]["variance_share"]

    def test_group_stats(self):
        result = variance_decomposition([1.0, 3.0, 100.0], ["x", "x", "y"])

        assert result["groups"]["x"]["n"] == 2
        assert result["groups"]["x"]["mean"] == pytest.approx(2.0)
        assert result["groups"]["x"]["share_of_samples"] == pytest.approx(2 / 3)
        assert result["groups"]["y"]["std"] == 0.0

    def test_constant_samples_have_zero_shares(self):
        result = variance_decomposition([5.0, 5.0, 5.0], ["a", "a", "b"])

        assert result["total_variance"] == 0.0
        assert result["between_group_share"] == 0.0
        assert all(g["variance_share"] == 0.0 for g in result["groups"].values())

    def test_misaligned_lengths_raise(self):
        with pytest.raises(ValueError, match="must align"):
            variance_decomposition([1.0, 2.0], ["a"])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            variance_decomposition([], [])


class TestConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_confidence_interval_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lower, upper = compute_confidence_interval(values)

        assert mean == pytest.approx(3.0)
        assert lower < mean < upper

    def test_confidence_interval_empty(self):
        values = []
        mean, lower, upper = compute_confidence_interval(values)

        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_confidence_interval_95_vs_99(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        _, lower_95, upper_95 = compute_confidence_interval(values, 0.95)
        _, lower_99, upper_99 = compute_confidence_interval(values, 0.99)

        # 99% CI should be wider than 95% CI
        assert (upper_99 - lower_99) > (upper_95 - lower_95)


class TestWinRateConfidenceInterval:
    """Tests for win rate confidence interval."""

    def test_win_rate_ci_basic(self):
        wins = 60
        total = 100

        win_rate, lower, upper = compute_win_rate_confidence_interval(wins, total)

        assert win_rate == pytest.approx(0.6)
        assert 0 <= lower <= win_rate <= upper <= 1

    def test_win_rate_ci_perfect_wins(self):
        wins = 100
        total = 100

        win_rate, lower, upper = compute_win_rate_confidence_interval(wins, total)

        assert win_rate == pytest.approx(1.0)
        assert lower > 0.9  # Should be high
        assert upper == pytest.approx(1.0)

    def test_win_rate_ci_no_wins(self):
        wins = 0
        total = 100

        win_rate, lower, upper = compute_win_rate_confidence_interval(wins, total)

        assert win_rate == pytest.approx(0.0)
        assert lower == pytest.approx(0.0)
        assert upper < 0.1  # Should be low

    def test_win_rate_ci_zero_total(self):
        wins = 0
        total = 0

        win_rate, lower, upper = compute_win_rate_confidence_interval(wins, total)

        assert win_rate == 0.0
        assert lower == 0.0
        assert upper == 0.0


class TestTTest:
    """Tests for t-test."""

    def test_t_test_different_samples(self):
        values1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        values2 = [6.0, 7.0, 8.0, 9.0, 10.0]

        t_stat, p_value, is_sig = t_test_difference(values1, values2)

        assert t_stat != 0
        assert p_value < 0.05  # Should be significant
        assert is_sig

    def test_t_test_same_samples(self):
        values1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        values2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        t_stat, p_value, is_sig = t_test_difference(values1, values2)

        assert t_stat == pytest.approx(0.0, abs=1e-10)
        assert p_value > 0.05  # Should not be significant
        assert not is_sig

    def test_t_test_insufficient_samples(self):
        values1 = [1.0]
        values2 = [2.0]

        t_stat, p_value, is_sig = t_test_difference(values1, values2)

        assert t_stat == 0.0
        assert p_value == 1.0
        assert not is_sig


class TestVariance:
    """Tests for variance computation."""

    def test_variance_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        variance, std_dev = compute_variance(values)

        assert variance > 0
        assert std_dev == pytest.approx(variance**0.5)

    def test_variance_empty(self):
        values = []

        variance, std_dev = compute_variance(values)

        assert variance == 0.0
        assert std_dev == 0.0

    def test_variance_constant(self):
        values = [3.0, 3.0, 3.0, 3.0]

        variance, std_dev = compute_variance(values)

        assert variance == pytest.approx(0.0, abs=1e-10)
        assert std_dev == pytest.approx(0.0, abs=1e-10)


class TestPercentiles:
    """Tests for percentile computation."""

    def test_percentiles_basic(self):
        values = list(range(1, 101))  # 1 to 100

        percentiles = compute_percentiles(values, [25, 50, 75])

        assert percentiles[0] == pytest.approx(25.75, abs=1)
        assert percentiles[1] == pytest.approx(50.5, abs=1)
        assert percentiles[2] == pytest.approx(75.25, abs=1)

    def test_percentiles_empty(self):
        values = []

        percentiles = compute_percentiles(values, [25, 50, 75])

        assert all(p == 0.0 for p in percentiles)


class TestSampleSize:
    """Tests for sample size estimation."""

    def test_sample_size_estimation(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        required_n = estimate_required_sample_size(values, target_margin_of_error=0.5)

        assert required_n >= 30  # Minimum
        assert isinstance(required_n, int)

    def test_sample_size_empty(self):
        values = []

        required_n = estimate_required_sample_size(values, target_margin_of_error=0.5)

        assert required_n == 100  # Default


class TestMatchStatisticsAnalyzer:
    """Tests for MatchStatisticsAnalyzer."""

    def test_create_analyzer(self):
        analyzer = MatchStatisticsAnalyzer(confidence_level=0.95)

        assert analyzer.confidence_level == 0.95

    def test_analyze_payoffs(self):
        analyzer = MatchStatisticsAnalyzer()

        payoffs = [10, 20, 30, 40, 50]  # In chips
        analysis = analyzer.analyze_payoffs(payoffs, big_blind=2)

        assert "mean" in analysis
        assert "ci_lower" in analysis
        assert "ci_upper" in analysis
        assert "std_dev" in analysis
        assert "variance" in analysis
        assert "bb_per_hand" in analysis

        # Mean should be 30 chips = 15 bb
        assert analysis["mean"] == pytest.approx(15.0)

    def test_analyze_payoffs_empty(self):
        analyzer = MatchStatisticsAnalyzer()

        payoffs = []
        analysis = analyzer.analyze_payoffs(payoffs)

        assert analysis["mean"] == 0.0
        assert analysis["bb_per_hand"] == 0.0

    def test_compare_strategies(self):
        analyzer = MatchStatisticsAnalyzer()

        payoffs1 = [10, 20, 30, 40, 50]
        payoffs2 = [5, 15, 25, 35, 45]

        comparison = analyzer.compare_strategies(payoffs1, payoffs2)

        assert "t_statistic" in comparison
        assert "p_value" in comparison
        assert "is_significant" in comparison
        assert "mean_difference" in comparison

        # payoffs1 should be higher
        assert comparison["mean_difference"] > 0

    def test_compare_strategies_empty(self):
        analyzer = MatchStatisticsAnalyzer()

        payoffs1 = []
        payoffs2 = []

        comparison = analyzer.compare_strategies(payoffs1, payoffs2)

        assert comparison["t_statistic"] == 0.0
        assert comparison["p_value"] == 1.0
        assert not comparison["is_significant"]

    def test_str_representation(self):
        analyzer = MatchStatisticsAnalyzer()

        s = str(analyzer)
        assert "MatchStatisticsAnalyzer" in s
