"""
Unit tests for numba-compiled operations.

Covers the unified regret-update kernel (the single implementation of the
CFR/CFR+/linear/DCFR regret math) and the DCFR strategy weight.
"""

import numpy as np
import pytest

from src.engine.solver.numba_ops import (
    WEIGHTING_CODES,
    apply_regret_updates,
    compute_dcfr_strategy_weight,
)


def _dcfr_discount(iteration: int, regret_value: float, alpha: float, beta: float) -> float:
    """Observe the kernel's DCFR discount factor on a single-slot row.

    A zero-utility update (utilities=0, node_utility=0, reach=1) leaves only
    the discount: result = regret_value * factor.
    """
    row = np.array([regret_value], dtype=np.float64)
    apply_regret_updates(
        row,
        np.array([0], dtype=np.int64),
        np.array([0.0]),
        0.0,
        1.0,
        False,
        iteration,
        WEIGHTING_CODES["dcfr"],
        alpha,
        beta,
    )
    return float(row[0]) / regret_value


def _reference_update(
    regrets: list[float],
    targets: list[int],
    utilities: list[float],
    node_utility: float,
    opponent_reach: float,
    cfr_plus: bool,
    iteration: int,
    weighting: str,
    alpha: float = 1.5,
    beta: float = 0.0,
) -> list[float]:
    """Scalar per-action reference: the pre-unification InfoSet.update_regret math."""
    out = list(regrets)
    for j, i in enumerate(targets):
        if weighting == "dcfr" and iteration > 1:
            exponent = alpha if out[i] > 0 else beta
            factor = 0.5 if exponent == 0.0 else (iteration**exponent / (iteration**exponent + 1.0))
            out[i] *= factor
        weighted = (utilities[j] - node_utility) * opponent_reach
        if weighting == "linear":
            weighted *= iteration
        updated = out[i] + weighted
        out[i] = max(0.0, updated) if cfr_plus else updated
    return out


class TestApplyRegretUpdates:
    """The unified kernel against the scalar reference, including index subsets."""

    @pytest.mark.parametrize("weighting", ["none", "linear", "dcfr"])
    @pytest.mark.parametrize("cfr_plus", [False, True])
    def test_masked_subset_matches_scalar_reference(self, weighting, cfr_plus):
        """Updating an index SUBSET (the pruned/partial-legal shape) is exactly
        the old per-action loop over those indices."""
        initial = [5.0, -2.0, 0.0, 3.25, -0.75]
        targets = [0, 2, 4]
        utilities = [2.1875, -3.0625, 1.0]
        node_utility = -0.4375
        opponent_reach = 0.3125

        row = np.array(initial, dtype=np.float64)
        expected = list(initial)
        for iteration in (1, 2, 7):
            apply_regret_updates(
                row,
                np.array(targets, dtype=np.int64),
                np.array(utilities),
                node_utility,
                opponent_reach,
                cfr_plus,
                iteration,
                WEIGHTING_CODES[weighting],
                1.5,
                0.0,
            )
            expected = _reference_update(
                expected,
                targets,
                utilities,
                node_utility,
                opponent_reach,
                cfr_plus,
                iteration,
                weighting,
            )
        assert row.tolist() == expected  # bit-identical, no tolerance

    def test_full_row_identity_indices(self):
        """Identity indices update every slot like the old full-row kernel."""
        row = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        apply_regret_updates(
            row,
            np.arange(3, dtype=np.int64),
            np.array([1.0, 2.0, 3.0]),
            2.0,
            0.5,
            False,
            1,
            WEIGHTING_CODES["none"],
            1.5,
            0.0,
        )
        assert row.tolist() == [1.0 + -0.5, -1.0 + 0.0, 0.5 + 0.5]

    def test_empty_targets_is_noop(self):
        """All-pruned nodes pass empty targets; the row must be untouched."""
        row = np.array([1.0, -2.0], dtype=np.float64)
        apply_regret_updates(
            row,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            1.0,
            1.0,
            True,
            5,
            WEIGHTING_CODES["dcfr"],
            1.5,
            0.0,
        )
        assert row.tolist() == [1.0, -2.0]

    def test_untargeted_slots_untouched(self):
        """Slots outside target_indices see neither discount nor add."""
        row = np.array([4.0, -4.0, 4.0], dtype=np.float64)
        apply_regret_updates(
            row,
            np.array([1], dtype=np.int64),
            np.array([1.0]),
            0.0,
            1.0,
            False,
            10,
            WEIGHTING_CODES["dcfr"],
            1.5,
            0.0,
        )
        assert row[0] == 4.0 and row[2] == 4.0  # no DCFR decay on untouched slots

    def test_single_action_delegation_form(self):
        """node_utility=0, reach=1 reduces to a plain regret add (the
        InfoSet.update_regret delegation contract)."""
        row = np.array([1.5], dtype=np.float64)
        apply_regret_updates(
            row,
            np.array([0], dtype=np.int64),
            np.array([-2.25]),
            0.0,
            1.0,
            False,
            1,
            WEIGHTING_CODES["none"],
            1.5,
            0.0,
        )
        assert row[0] == 1.5 - 2.25


class TestDCFRWeights:
    """DCFR discount behavior of the unified kernel, and the strategy weight."""

    def test_dcfr_no_discount_at_iteration_1(self):
        assert _dcfr_discount(1, 8.0, alpha=1.5, beta=0.0) == 1.0
        assert _dcfr_discount(1, -8.0, alpha=1.5, beta=0.0) == 1.0

    def test_dcfr_discount_approaches_one(self):
        w1 = _dcfr_discount(10, 8.0, alpha=1.5, beta=0.0)
        w2 = _dcfr_discount(100, 8.0, alpha=1.5, beta=0.0)
        w3 = _dcfr_discount(1000, 8.0, alpha=1.5, beta=0.0)
        assert w1 < w2 < w3 < 1.0

    def test_dcfr_alpha_vs_beta(self):
        """Positive regrets discount by alpha, negative by beta; beta=0 halves."""
        assert _dcfr_discount(100, -8.0, alpha=1.5, beta=0.0) == 0.5
        w_pos = _dcfr_discount(100, 8.0, alpha=1.5, beta=0.0)
        assert 0.9 < w_pos < 1.0
        assert _dcfr_discount(100, -8.0, alpha=1.5, beta=0.5) < 1.0

    def test_dcfr_zero_exponent_halves(self):
        """t^0 / (t^0 + 1) = 1/2 for either sign (Brown & Sandholm 2019)."""
        assert _dcfr_discount(100, 8.0, alpha=0.0, beta=0.0) == 0.5
        assert _dcfr_discount(100, -8.0, alpha=0.0, beta=0.0) == 0.5

    @pytest.mark.parametrize("alpha", [1.5, 2.0])
    def test_dcfr_discount_formula(self, alpha):
        """Standard DCFR formula t^alpha / (t^alpha + 1)."""
        t_exp = 100.0**alpha
        assert _dcfr_discount(100, 8.0, alpha=alpha, beta=0.0) == pytest.approx(
            t_exp / (t_exp + 1.0), abs=1e-12
        )

    def test_dcfr_strategy_weight(self):
        w1 = compute_dcfr_strategy_weight(1, gamma=2.0)
        w10 = compute_dcfr_strategy_weight(10, gamma=2.0)
        w100 = compute_dcfr_strategy_weight(100, gamma=2.0)
        assert w1 == 1.0
        assert w100 > w10 > w1

    @pytest.mark.parametrize("gamma,expected", [(2.0, 100.0**2.0), (3.0, 100.0**3.0)])
    def test_dcfr_strategy_weight_formula(self, gamma, expected):
        assert compute_dcfr_strategy_weight(100, gamma=gamma) == pytest.approx(expected, abs=1e-6)

    def test_dcfr_strategy_weight_zero_gamma(self):
        assert compute_dcfr_strategy_weight(100, gamma=0.0) == 1.0
