"""Shaping the prior must change how hard a row resists, never what it plays.

The flat shape seeds every row at full strength, including rows where the prior
is near-uniform -- and those units then have to be overcome before the solver can
move away from uniform at all. That is not a neutral prior; it is a brake applied
hardest to the rows the prior had no opinion about.

The confidence shape scales each row by how decisive the prior is there. The
danger is that a change to the WEIGHTS becomes a change to the STRATEGY, which
would make every warm-start comparison measure two things at once. That is the
first thing pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.services.warm_start import PRIOR_SHAPES, regrets_encoding, row_confidence

# Three rows of widths 3, 2, 4, laid out contiguously as the ragged slots are.
WIDTHS = np.array([3, 2, 4], dtype=np.int64)
STARTS = np.array([0, 3, 5], dtype=np.int64)

# Row 0 decisive (90/5/5), row 1 indifferent (50/50), row 2 mildly tilted.
STRATEGY = np.array([90.0, 5.0, 5.0, 50.0, 50.0, 40.0, 30.0, 20.0, 10.0])


def _match(regrets: np.ndarray, start: int, width: int) -> np.ndarray:
    positive = np.maximum(regrets[start : start + width], 0.0)
    total = positive.sum()
    return positive / total if total > 0 else np.full(width, 1.0 / width)


class TestShapingDoesNotMoveTheStrategy:
    """The whole method rests on this: shape changes resistance, not policy."""

    def test_every_shape_plays_the_same_strategy(self):
        flat, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="flat")
        conf, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="confidence")
        for start, width in zip(STARTS, WIDTHS, strict=True):
            np.testing.assert_allclose(
                _match(flat, start, width), _match(conf, start, width), rtol=1e-12
            )

    def test_an_unknown_shape_is_refused(self):
        with pytest.raises(ValueError, match="unknown prior shape"):
            regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="entropy")

    def test_flat_is_the_default(self):
        """Every measured result so far used flat; changing the default would
        silently reinterpret them."""
        explicit, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="flat")
        implied, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0)
        np.testing.assert_array_equal(explicit, implied)
        assert PRIOR_SHAPES[0] == "flat"


class TestConfidence:
    def test_a_decisive_row_scores_higher_than_an_indifferent_one(self):
        c = row_confidence(
            np.array([0.9, 0.05, 0.05, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1]), STARTS, WIDTHS
        )
        assert c[0] > 0.4, c
        assert c[1] == pytest.approx(0.0, abs=1e-9), "a 50/50 row has no opinion to assert"
        assert c[0] > c[2] > c[1]

    def test_a_forced_action_is_never_damped(self):
        """Width-1 rows have nothing to be uncertain about; damping them would
        weaken a prior that is not even a choice."""
        c = row_confidence(np.array([1.0]), np.array([0]), np.array([1]))
        assert c[0] == pytest.approx(1.0)

    def test_confidence_is_bounded(self):
        rng = np.random.default_rng(0)
        raw = rng.random(9)
        totals = np.add.reduceat(raw, STARTS)
        norm = raw / np.repeat(totals, WIDTHS)
        c = row_confidence(norm, STARTS, WIDTHS)
        assert ((c >= 0.0) & (c <= 1.0)).all(), c


class TestWhatShapingActuallyChanges:
    def test_an_indifferent_row_stops_braking(self):
        """The point of the exercise: a row the prior has no view on should not
        resist the solver, and under flat it resists exactly as hard as a row the
        prior is certain about."""
        flat, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="flat")
        conf, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="confidence")
        indifferent = slice(3, 5)
        assert flat[indifferent].sum() == pytest.approx(1000.0)
        assert conf[indifferent].sum() < 1.0, "a 50/50 row should assert almost nothing"

    def test_a_decisive_row_keeps_most_of_its_weight(self):
        conf, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="confidence")
        assert conf[0:3].sum() > 400.0, "a 90/5/5 row should still speak up"

    def test_evidence_overrides_a_damped_row_far_sooner(self):
        """Concretely: the same real regret moves an indifferent row much further
        under confidence shaping, which is the whole intended effect."""
        evidence = np.zeros(9)
        evidence[4] = 200.0  # the solver learns the second action is better
        flat, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="flat")
        conf, _ = regrets_encoding(STRATEGY, STARTS, WIDTHS, 1000.0, shape="confidence")
        moved_flat = _match(flat + evidence, 3, 2)[1]
        moved_conf = _match(conf + evidence, 3, 2)[1]
        assert moved_conf > moved_flat + 0.2, (moved_flat, moved_conf)
