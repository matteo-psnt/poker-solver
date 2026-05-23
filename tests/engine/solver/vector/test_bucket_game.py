"""Removing the board dimension has to preserve the things CFR relies on.

The derivation in ``bucket_game`` averages over boards, and averaging is exactly
where a quantity can stop meaning what its name says. Three properties are
load-bearing downstream and are pinned here:

* transitions are row-stochastic, so range mass is conserved rather than
  quietly leaking as play moves down the streets;
* the showdown matrix is antisymmetric, which is what carries zero-sum from hand
  space into bucket space — the bucket kernel's zero-sum check is only
  meaningful because this holds;
* the compatibility rate is a probability and is symmetric, since "these two
  holdings collide" does not depend on who is asked.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.state import Street
from src.engine.solver.vector import bucket_game
from src.engine.solver.vector.hand_context import (
    HandContext,
    blocking_matrix,
    enumerate_live_hands,
)
from tests.engine.solver.vector.contexts import ordered_context

COUNTS = {Street.FLOP: 4, Street.TURN: 5, Street.RIVER: 6}
ALL_COUNTS = {Street.PREFLOP: 169, **COUNTS}


def _context(rng):
    """Strength-ordered, so showdowns carry signal — see ``contexts``."""
    return ordered_context(rng, ALL_COUNTS)


@pytest.fixture(scope="module")
def game():
    rng = np.random.default_rng(19)
    return bucket_game.derive([_context(rng) for _ in range(4)], ALL_COUNTS)


class TestInvariants:
    def test_transitions_conserve_mass(self, game):
        for step, matrix in game.transitions.items():
            rows = matrix.sum(axis=1)
            occupied = rows > 0
            assert occupied.any(), f"{step} has no occupied source bucket"
            assert np.allclose(rows[occupied], 1.0, atol=1e-4)

    def test_showdown_is_antisymmetric(self, game):
        assert np.allclose(game.showdown, -game.showdown.T, atol=1e-6)

    def test_showdown_diagonal_is_zero(self, game):
        """A bucket against itself is symmetric, so it wins as often as it loses."""
        assert np.allclose(np.diag(game.showdown), 0.0, atol=1e-6)

    def test_compatibility_is_a_symmetric_probability(self, game):
        for street, rate in game.compatible.items():
            assert np.allclose(rate, rate.T, atol=1e-6), street
            assert (rate >= 0).all(), street
            assert (rate <= 1.0 + 1e-6).all(), street

    def test_validate_rejects_a_broken_showdown_matrix(self, game):
        broken = bucket_game.BucketGame(
            buckets_per_street=game.buckets_per_street,
            transitions=game.transitions,
            compatible=game.compatible,
            showdown=np.abs(game.showdown) + 0.5,
        )
        with pytest.raises(ValueError, match="antisymmetric"):
            broken.validate()


class TestDerivation:
    def test_transitions_match_a_direct_count(self):
        """The matrix is P(next | this) over (runout, hand) pairs — count it."""
        rng = np.random.default_rng(5)
        contexts = [_context(rng) for _ in range(3)]
        game = bucket_game.derive(contexts, ALL_COUNTS)

        counts = np.zeros((ALL_COUNTS[Street.FLOP], ALL_COUNTS[Street.TURN]))
        for context in contexts:
            flop = context.buckets_for(Street.FLOP)
            turn = context.buckets_for(Street.TURN)
            for source, destination in zip(flop.tolist(), turn.tolist(), strict=True):
                counts[source, destination] += 1

        totals = counts.sum(axis=1, keepdims=True)
        expected = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        assert np.allclose(game.transitions[(Street.FLOP, Street.TURN)], expected, atol=1e-5)

    def test_a_single_bucket_per_street_collapses_to_the_whole_range(self):
        """With one bucket everywhere, every hand is the same hand.

        The transition is then the 1x1 identity and the showdown must be exactly
        zero: a range against itself wins as often as it loses, whatever the
        cards are. Any nonzero here is a sign error in the aggregation.
        """
        rng = np.random.default_rng(3)
        hand_cards = enumerate_live_hands(rng.choice(52, 5, replace=False))
        num_hands = hand_cards.shape[0]
        blocks = blocking_matrix(hand_cards)
        context = HandContext(
            hand_cards,
            np.zeros((4, num_hands), dtype=np.int64),
            rng.permutation(num_hands),
            blocks,
        )
        single = dict.fromkeys(ALL_COUNTS, 1)

        game = bucket_game.derive([context], single)
        assert game.showdown.shape == (1, 1)
        assert game.showdown[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert game.transitions[(Street.FLOP, Street.TURN)][0, 0] == pytest.approx(1.0)
