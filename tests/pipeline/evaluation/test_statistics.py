"""Tests for statistical analysis."""

import numpy as np
import pytest

from src.pipeline.evaluation.statistics import (
    compare_paired_samples,
    summarize_samples,
    variance_decomposition,
)


class TestSummarizeSamples:
    """One-sample summary: t-based CI and the degenerate zero-variance convention."""

    def test_basic_summary(self):
        result = summarize_samples([1.0, 2.0, 3.0, 4.0, 5.0])

        assert result["mean"] == pytest.approx(3.0)
        assert result["ci_lower"] < 3.0 < result["ci_upper"]
        assert 0.0 < result["p_value"] < 1.0

    def test_zero_variance_conventions(self):
        null = summarize_samples([0.0, 0.0, 0.0])
        assert null["p_value"] == 1.0 and not null["is_significant"]

        shifted = summarize_samples([2.0, 2.0, 2.0])
        assert shifted["p_value"] == 0.0 and shifted["is_significant"]
        assert shifted["ci_lower"] == shifted["ci_upper"] == pytest.approx(2.0)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            summarize_samples([1.0])


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
