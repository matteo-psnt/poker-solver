"""
Unit tests for numba-compiled operations.

Tests the DCFR weight calculation functions.
"""

from src.shared.numba_ops import (
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

    def test_dcfr_weight_approaches_one(self):
        """Discount factor should approach 1.0 as iterations increase."""
        w1 = compute_dcfr_weight(10, alpha=1.5, beta=0.0, is_positive=True)
        w2 = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=True)
        w3 = compute_dcfr_weight(1000, alpha=1.5, beta=0.0, is_positive=True)

        # Higher iteration → discount factor closer to 1.0 (less discounting)
        assert w2 > w1
        assert w3 > w2
        assert w3 < 1.0  # Should still be < 1.0 (some discount applied)

    def test_dcfr_weight_alpha_vs_beta(self):
        """Positive regrets should use alpha, negative use beta."""
        w_pos = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=True)
        w_neg = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=False)

        # With beta=0, negative regrets get weight 1.0 (no discount)
        assert w_neg == 1.0
        # With alpha=1.5, positive regrets get discounted
        assert w_pos < 1.0  # Discount applied
        assert w_pos > 0.9  # But close to 1 for iteration 100

        # With non-zero beta
        w_neg_beta = compute_dcfr_weight(100, alpha=1.5, beta=0.5, is_positive=False)
        # Should also be discounted (but differently than alpha)
        assert w_neg_beta < 1.0

    def test_dcfr_weight_zero_exponent(self):
        """Zero exponent should give uniform weight 1.0."""
        weight = compute_dcfr_weight(100, alpha=0.0, beta=0.0, is_positive=True)
        assert weight == 1.0

        weight_neg = compute_dcfr_weight(100, alpha=0.0, beta=0.0, is_positive=False)
        assert weight_neg == 1.0

    def test_dcfr_weight_formula(self):
        """Test that the standard DCFR formula t^exp / (t^exp + 1) is correctly applied."""
        # For alpha=1.5 and iteration=100
        # Formula: 100^1.5 / (100^1.5 + 1) = 1000 / 1001 ≈ 0.999
        weight = compute_dcfr_weight(100, alpha=1.5, beta=0.0, is_positive=True)
        t_exp = 100.0**1.5
        expected = t_exp / (t_exp + 1.0)
        assert abs(weight - expected) < 1e-6

        # For alpha=2.0 and iteration=100
        # Formula: 100^2.0 / (100^2.0 + 1) = 10000 / 10001 ≈ 0.9999
        weight = compute_dcfr_weight(100, alpha=2.0, beta=0.0, is_positive=True)
        t_exp = 100.0**2.0
        expected = t_exp / (t_exp + 1.0)
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
        """Test that the strategy weight formula t^gamma is correctly applied."""
        # For gamma=2.0 and iteration=100, weight should be 100^2.0 = 10000
        weight = compute_dcfr_strategy_weight(100, gamma=2.0)
        expected = 100.0**2.0
        assert abs(weight - expected) < 1e-6

        # For gamma=3.0 and iteration=100, weight should be 100^3.0 = 1000000
        weight = compute_dcfr_strategy_weight(100, gamma=3.0)
        expected = 100.0**3.0
        assert abs(weight - expected) < 1e-6

    def test_dcfr_strategy_weight_zero_gamma(self):
        """Zero gamma should give uniform weight 1.0."""
        weight = compute_dcfr_strategy_weight(100, gamma=0.0)
        assert weight == 1.0

    def test_dcfr_weight_typical_values(self):
        """Test with typical DCFR parameter values from literature (Brown & Sandholm 2019)."""
        # Typical values: alpha=1.5, beta=0.0, gamma=2.0
        iteration = 1000

        # Positive regret discount factor
        # Formula: 1000^1.5 / (1000^1.5 + 1) ≈ 0.999968
        w_pos = compute_dcfr_weight(iteration, alpha=1.5, beta=0.0, is_positive=True)
        t_exp = 1000.0**1.5
        expected_pos = t_exp / (t_exp + 1.0)
        assert abs(w_pos - expected_pos) < 1e-6
        assert w_pos > 0.999  # Should be very close to 1.0 for large t

        # Negative regret discount factor (beta=0 means no discount)
        w_neg = compute_dcfr_weight(iteration, alpha=1.5, beta=0.0, is_positive=False)
        assert w_neg == 1.0

        # Strategy weight (t^gamma)
        w_strategy = compute_dcfr_strategy_weight(iteration, gamma=2.0)
        expected_strategy = 1000.0**2.0  # 1,000,000
        assert abs(w_strategy - expected_strategy) < 1e-6
