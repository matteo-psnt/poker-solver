"""The opening guess must be poker-shaped, not just non-uniform.

Replacing uniform with something arbitrary is not an improvement; the guess is
only worth seeding if it says the things a competent player would. So these pin
DIRECTION -- strong hands lean aggressive, weak hands lean passive, folding is
for weak hands -- rather than exact probabilities, which are a tuning choice.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.core.game.actions import Action, ActionType
from src.pipeline.services.equity_prior import (
    action_aggression,
    bucket_strength,
    strength_policy,
)

POT = 100.0
# A typical facing-a-bet menu: fold, call, half pot, pot, jam.
MENU = (
    Action(ActionType.FOLD),
    Action(ActionType.CALL),
    Action(ActionType.RAISE, 50),
    Action(ActionType.RAISE, 100),
    Action(ActionType.ALL_IN, 400),
)


def _policy(strength: float) -> np.ndarray:
    return strength_policy(strength, action_aggression(MENU, POT))


class TestAggressionAxis:
    def test_it_is_ordered_fold_to_allin(self):
        a = action_aggression(MENU, POT)
        assert list(a) == sorted(a), a
        assert a[0] == 0.0
        assert a[-1] == 1.0

    def test_a_bigger_bet_is_more_aggressive(self):
        small, big = action_aggression(
            (Action(ActionType.BET, 33), Action(ActionType.BET, 100)), POT
        )
        assert big > small

    def test_a_huge_overbet_does_not_outrank_all_in(self):
        """Capped at one pot, so a 4x pot bet and a jam stay distinguishable."""
        a = action_aggression((Action(ActionType.BET, 400), Action(ActionType.ALL_IN, 900)), POT)
        assert a[1] > a[0]


class TestTheGuessIsPokerShaped:
    def test_a_strong_hand_prefers_aggression(self):
        p = _policy(0.95)
        assert p.argmax() >= 3, p
        assert p[0] < 0.05, "a strong hand should not be folding"

    def test_a_weak_hand_prefers_folding_or_calling(self):
        p = _policy(0.05)
        assert p.argmax() <= 1, p

    def test_a_middling_hand_sits_in_the_middle(self):
        p = _policy(0.5)
        assert 1 <= p.argmax() <= 3, p

    def test_aggression_rises_monotonically_with_strength(self):
        """The single property that makes this a poker prior rather than noise."""
        axis = action_aggression(MENU, POT)
        means = [float(_policy(s) @ axis) for s in (0.05, 0.25, 0.5, 0.75, 0.95)]
        assert all(a < b for a, b in itertools.pairwise(means)), means


class TestItIsAProbabilityDistribution:
    @pytest.mark.parametrize("strength", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_it_sums_to_one_and_is_non_negative(self, strength):
        p = _policy(strength)
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()

    def test_a_single_action_gets_all_the_mass(self):
        p = strength_policy(0.5, action_aggression((Action(ActionType.CHECK),), POT))
        assert p == pytest.approx([1.0])

    def test_a_high_temperature_degrades_toward_uniform(self):
        """The thing being replaced is uniform, so the knob must be able to
        reach it -- otherwise a bad temperature cannot be diagnosed."""
        p = strength_policy(0.9, action_aggression(MENU, POT), temperature=1e6)
        np.testing.assert_allclose(p, np.full(len(MENU), 1 / len(MENU)), atol=1e-3)


class TestBucketStrength:
    def test_it_spans_the_unit_interval_in_order(self):
        s = [bucket_strength(b, 600) for b in (0, 300, 599)]
        assert 0.0 < s[0] < s[1] < s[2] < 1.0

    def test_the_extremes_are_not_certainties(self):
        """The top river bucket is a strong class, not the nuts. Claiming 1.0
        there would seed maximum aggression in the spots that cost most."""
        assert bucket_strength(599, 600) < 1.0
        assert bucket_strength(0, 600) > 0.0

    def test_a_single_bucket_street_is_neutral(self):
        assert bucket_strength(0, 1) == 0.5
