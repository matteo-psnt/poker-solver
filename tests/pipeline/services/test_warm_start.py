"""Does the warm start actually encode the strategy it claims to?

The whole method rests on one identity: regret matching reads
``sigma(a) = R+(a) / sum(R+)``, so regrets proportional to a strategy reproduce
that strategy exactly. If that fails the run does not crash — it starts from some
other policy and every downstream number is measuring the wrong thing.

These are on the pure encoder, so they need no tree, config or disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.pipeline.services.warm_start import regrets_encoding
from tests.test_helpers import make_test_config

# Three rows of widths 3, 2, 4 laid out contiguously, as the ragged slot layout does.
WIDTHS = np.array([3, 2, 4], dtype=np.int64)
STARTS = np.array([0, 3, 5], dtype=np.int64)


class _Buckets:
    """Bucket counts only; the tree never asks this for an assignment here."""

    def __init__(self, counts):
        self.counts = counts

    def get_bucket(self, hole_cards, board, street):
        return 0

    def num_buckets(self, street):
        return self.counts.get(street, 169)


def test_row_slot_starts_agree_with_the_tree_accessors():
    """Pin the boundaries to the TREE's own layout, not to a rebuilt copy.

    The seeding path once rebuilt row order with ``np.repeat`` — node-major,
    the retired layout — and every test fed it hand-made arrays, so the drift
    was invisible while the seeded prior landed on the wrong infosets. A real
    tree makes the agreement the assertion.
    """
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    tree = BettingTree(
        GameRules(1, 2),
        ActionModel(config),
        starting_stack=20,
        buckets_per_street={Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5},
    )
    starts = tree.row_slot_starts
    for node in tree.nodes:
        for bucket in range(tree.num_buckets(node.street)):
            row = tree.row(node.node_id, bucket)
            lo, hi = tree.slots(node.node_id, bucket)
            assert starts[row] == lo, (node.node_id, bucket)
            assert tree.row_widths[row] == hi - lo


def _match(regrets: np.ndarray, start: int, width: int) -> np.ndarray:
    """Regret matching over one row, the way the solver reads it."""
    positive = np.maximum(regrets[start : start + width], 0.0)
    total = positive.sum()
    return positive / total if total > 0 else np.full(width, 1.0 / width)


class TestEncoding:
    def test_matching_the_seeded_regrets_reproduces_the_strategy(self):
        strategy = np.array([2.0, 1.0, 1.0, 3.0, 9.0, 4.0, 4.0, 4.0, 8.0])
        regrets, seeded = regrets_encoding(strategy, STARTS, np.repeat(WIDTHS, 1), weight=1000.0)

        assert seeded.tolist() == [True, True, True]
        for start, width in zip(STARTS, WIDTHS, strict=True):
            expected = strategy[start : start + width] / strategy[start : start + width].sum()
            np.testing.assert_allclose(_match(regrets, start, width), expected, rtol=1e-12)

    def test_the_weight_scales_regrets_without_moving_the_strategy(self):
        """The weight is how much the prior CLAIMS, not what it plays. Two weights
        must give the same policy and different magnitudes, or the knob is
        secretly changing the strategy as well as its confidence."""
        strategy = np.array([2.0, 1.0, 1.0, 3.0, 9.0, 4.0, 4.0, 4.0, 8.0])
        light, _ = regrets_encoding(strategy, STARTS, WIDTHS, weight=10.0)
        heavy, _ = regrets_encoding(strategy, STARTS, WIDTHS, weight=10_000.0)

        for start, width in zip(STARTS, WIDTHS, strict=True):
            np.testing.assert_allclose(
                _match(light, start, width), _match(heavy, start, width), rtol=1e-12
            )
        assert heavy.sum() > light.sum() * 100

    def test_every_seeded_row_claims_the_same_confidence(self):
        """The row normalisation is what makes ``effective_iterations`` mean one
        thing everywhere, and nothing else in this file pins it.

        Regret matching normalises per row, so seeding raw ``strategy_sum``
        reproduces the policy just as well — the tests above would all pass. What
        it would NOT do is give every row the same weight: a row the source
        visited often carries far more mass than a sparse one, so the prior would
        be arbitrarily stronger there and the knob would mean a different number
        of iterations per row.
        """
        # Row 0 accumulated 4 units of mass, row 1 twelve, row 2 twenty.
        strategy = np.array([2.0, 1.0, 1.0, 3.0, 9.0, 4.0, 4.0, 4.0, 8.0])
        regrets, _ = regrets_encoding(strategy, STARTS, WIDTHS, weight=1000.0)

        for start, width in zip(STARTS, WIDTHS, strict=True):
            assert regrets[start : start + width].sum() == pytest.approx(1000.0), (
                "each seeded row must claim exactly `weight`, independent of how "
                "much mass the source happened to accumulate there"
            )

    def test_regrets_are_never_negative(self):
        """Regret matching only reads the positive part, so a negative entry would
        silently drop that action from the prior."""
        strategy = np.array([2.0, 1.0, 1.0, 3.0, 9.0, 4.0, 4.0, 4.0, 8.0])
        regrets, _ = regrets_encoding(strategy, STARTS, WIDTHS, weight=1000.0)
        assert (regrets >= 0).all()

    def test_an_unvisited_row_is_left_uniform_not_forced(self):
        """A row the source never answered gets zeros, which regret matching reads
        as uniform. Inventing a preference there would assert knowledge the source
        did not have."""
        strategy = np.array([2.0, 1.0, 1.0, 0.0, 0.0, 4.0, 4.0, 4.0, 8.0])
        regrets, seeded = regrets_encoding(strategy, STARTS, WIDTHS, weight=1000.0)

        assert seeded.tolist() == [True, False, True]
        assert regrets[3:5].tolist() == [0.0, 0.0]
        np.testing.assert_allclose(_match(regrets, 3, 2), [0.5, 0.5])

    @pytest.mark.parametrize("weight", [0.5, 1.0, 7.5])
    def test_fractional_weights_still_encode_the_same_policy(self, weight):
        strategy = np.array([1.0, 3.0, 0.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0])
        regrets, _ = regrets_encoding(strategy, STARTS, WIDTHS, weight=weight)
        np.testing.assert_allclose(_match(regrets, 0, 3), [0.25, 0.75, 0.0], rtol=1e-12)
