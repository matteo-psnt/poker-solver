"""
Unit tests for numba-compiled operations.

Tests the DCFR weight calculation functions.
"""

from src.utils.numba_ops import (
    compute_dcfr_strategy_weight,
    compute_dcfr_weight,
)


class TestDCFRWeights:
    """Tests for DCFR weight calculation functions."""

    def test_dcfr_weight_iteration_1(self):
        """First iteration should have weight 1.0."""
        weight = compute_dcfr_weight(1, alpha=1.5, beta=0.0, is_positive=True)
        assert weight == 1.0

        weight_neg = compute_dcfr_weight(1, alpha=1.5, beta=0.0, is_positive=False)
        assert weight_neg == 1.0

    def test_dcfr_weight_increases_with_iteration(self):
        """Weight should increase with iteration number for alpha > 0."""
        w1 = compute_dcfr_weight(10, alpha=1.5, beta=0.0, is_positive=True)
        w2 = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=True)
        w3 = compute_dcfr_weight(1000, alpha=1.5, beta=0.0, is_positive=True)

        # Higher iteration → higher weight (with alpha = 1.5)
        assert w2 > w1
        assert w3 > w2

    def test_dcfr_weight_alpha_vs_beta(self):
        """Positive regrets should use alpha, negative use beta."""
        w_pos = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=True)
        w_neg = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=False)

        # With beta=0, negative regrets get weight 1.0
        assert w_neg == 1.0
        assert w_pos > w_neg

        # With non-zero beta (beta < 1 means discounting negative regrets)
        w_neg_beta = compute_dcfr_weight(100, alpha=1.5, beta=0.5, is_positive=False)
        # 100^(0.5-1) = 100^(-0.5) = 0.1 < 1.0
        assert w_neg_beta < 1.0  # Beta < 1 means discounting

    def test_dcfr_weight_zero_exponent(self):
        """Zero exponent should give uniform weight 1.0."""
        weight = compute_dcfr_weight(100, alpha=0.0, beta=0.0, is_positive=True)
        assert weight == 1.0

        weight_neg = compute_dcfr_weight(100, alpha=0.0, beta=0.0, is_positive=False)
        assert weight_neg == 1.0

    def test_dcfr_weight_formula(self):
        """Test that the formula t^(exponent-1) is correctly applied."""
        # For alpha=1.5 and iteration=100, weight should be 100^0.5 = 10
        weight = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=True)
        expected = 100.0**0.5
        assert abs(weight - expected) < 1e-6

        # For alpha=2.0 and iteration=100, weight should be 100^1.0 = 100
        weight = compute_dcfr_weight(100, alpha=2.0, beta=0.0, is_positive=True)
        expected = 100.0**1.0
        assert abs(weight - expected) < 1e-6

    def test_dcfr_strategy_weight(self):
        """Test strategy weight calculation."""
        # First iteration
        w1 = compute_dcfr_strategy_weight(1, gamma=2.0)
        assert w1 == 1.0

        # Later iterations
        w10 = compute_dcfr_strategy_weight(10, gamma=2.0)
        w100 = compute_dcfr_strategy_weight(100, gamma=2.0)

        # Should increase with iteration
        assert w100 > w10 > w1

    def test_dcfr_strategy_weight_formula(self):
        """Test that the strategy weight formula t^(gamma-1) is correctly applied."""
        # For gamma=2.0 and iteration=100, weight should be 100^1.0 = 100
        weight = compute_dcfr_strategy_weight(100, gamma=2.0)
        expected = 100.0**1.0
        assert abs(weight - expected) < 1e-6

        # For gamma=3.0 and iteration=100, weight should be 100^2.0 = 10000
        weight = compute_dcfr_strategy_weight(100, gamma=3.0)
        expected = 100.0**2.0
        assert abs(weight - expected) < 1e-6

    def test_dcfr_strategy_weight_zero_gamma(self):
        """Zero gamma should give uniform weight 1.0."""
        weight = compute_dcfr_strategy_weight(100, gamma=0.0)
        assert weight == 1.0

    def test_dcfr_weight_typical_values(self):
        """Test with typical DCFR parameter values from literature."""
        # Typical values: alpha=1.5, beta=0.0, gamma=2.0
        iteration = 1000

        # Positive regret weight
        w_pos = compute_dcfr_weight(iteration, alpha=1.5, beta=0.0, is_positive=True)
        expected_pos = 1000.0**0.5  # ~31.6
        assert abs(w_pos - expected_pos) < 1e-6

        # Negative regret weight (beta=0)
        w_neg = compute_dcfr_weight(iteration, alpha=1.5, beta=0.0, is_positive=False)
        assert w_neg == 1.0

        # Strategy weight
        w_strategy = compute_dcfr_strategy_weight(iteration, gamma=2.0)
        expected_strategy = 1000.0**1.0  # 1000
        assert abs(w_strategy - expected_strategy) < 1e-6
