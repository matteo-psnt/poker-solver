"""The load-time policy threshold, pinned against the evaluator's own transform.

The number that justifies this knob (940.1 -> 854.0 on the gate) was measured
through `transform_policy_rows`. Fielding it through a different implementation
is only worth anything if the two agree, so that agreement is the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.solver.policy.threshold import threshold_strategy_sum
from src.pipeline.evaluation.estimators.public_tree_br import (
    PublicBRConfig,
    transform_policy_rows,
)

THRESHOLD = 0.02


def _normalise(strategy_sum: np.ndarray, starts: np.ndarray, widths: np.ndarray) -> np.ndarray:
    totals = np.add.reduceat(strategy_sum, starts)
    out = strategy_sum / np.repeat(np.where(totals > 0, totals, 1.0), widths)
    return out.reshape(len(starts), int(widths[0]))


class TestAgainstTheEvaluator:
    """Same rows, same threshold, same answer."""

    @pytest.mark.parametrize("threshold", [0.02, 0.05, 0.2])
    def test_the_two_implementations_agree(self, threshold: float) -> None:
        rng = np.random.default_rng(7)
        buckets, actions = 64, 5
        raw = rng.random((buckets, actions)).astype(np.float32) ** 4
        # A row of near-equal tiny mass: every action below the cut, so both
        # implementations have to fall back rather than emit NaN.
        raw[3] = np.float32(1e-6)
        missing = np.zeros(buckets, dtype=bool)
        missing[:4] = True  # untrained rows must survive untouched, uniform.

        rows = raw / raw.sum(axis=1, keepdims=True)
        rows[missing] = 1.0 / actions
        theirs = transform_policy_rows(
            rows.astype(np.float64), missing, PublicBRConfig(policy_threshold=threshold)
        )

        flat = raw.reshape(-1).copy()
        widths = np.full(buckets, actions, dtype=np.int64)
        starts = np.arange(buckets, dtype=np.int64) * actions
        threshold_strategy_sum(flat, (~missing).astype(np.uint8), starts, widths, threshold)
        ours = _normalise(flat, starts, widths)
        ours[missing] = 1.0 / actions

        np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-7)


class TestThresholdStrategySum:
    def test_a_zero_threshold_is_a_no_op(self) -> None:
        flat = np.array([0.1, 0.9, 0.5, 0.5], dtype=np.float32)
        before = flat.copy()
        changed = threshold_strategy_sum(
            flat, np.ones(2, np.uint8), np.array([0, 2]), np.array([2, 2]), 0.0
        )
        assert changed == 0
        np.testing.assert_array_equal(flat, before)

    def test_an_untrained_row_is_never_touched(self) -> None:
        # Its uniform fallback is what the blueprint plays there; purifying it
        # would field the first action, which is fold.
        flat = np.array([0.001, 0.999], dtype=np.float32)
        threshold_strategy_sum(flat, np.zeros(1, np.uint8), np.array([0]), np.array([2]), THRESHOLD)
        np.testing.assert_allclose(flat, [0.001, 0.999], rtol=1e-6)

    def test_a_row_below_the_cut_everywhere_keeps_its_largest(self) -> None:
        # 100 equal actions are all at 0.01 < 0.02. Emptying the row would make
        # the blueprint unable to act at all here.
        flat = np.full(100, 0.01, dtype=np.float32)
        flat[42] = np.float32(0.0101)
        threshold_strategy_sum(
            flat, np.ones(1, np.uint8), np.array([0]), np.array([100]), THRESHOLD
        )
        assert flat[42] > 0.0
        assert np.count_nonzero(flat) == 1

    def test_a_row_with_no_mass_is_left_alone(self) -> None:
        flat = np.zeros(3, dtype=np.float32)
        changed = threshold_strategy_sum(
            flat, np.ones(1, np.uint8), np.array([0]), np.array([3]), THRESHOLD
        )
        assert changed == 0

    def test_ragged_rows_threshold_independently(self) -> None:
        # The real table is ragged -- widths follow each node's action count --
        # so a boundary read off the wrong row would silently mix two infosets.
        flat = np.array([0.99, 0.01, 0.5, 0.49, 0.01], dtype=np.float32)
        threshold_strategy_sum(
            flat, np.ones(2, np.uint8), np.array([0, 2]), np.array([2, 3]), THRESHOLD
        )
        np.testing.assert_allclose(flat, [0.99, 0.0, 0.5, 0.49, 0.0])
