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
    DEFAULT_TEMPERATURE,
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


def _test_tree(config, abstraction):
    from src.core.actions.action_model import ActionModel
    from src.core.game.rules import GameRules
    from src.engine.solver.betting_tree import build_betting_tree

    return build_betting_tree(
        GameRules(config.game.small_blind, config.game.big_blind),
        ActionModel(config),
        abstraction,
        starting_stack=config.game.starting_stack,
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
        # 0.10, not 0.05: at the MEASURED-best temperature a strong hand still
        # folds ~5%. The optimum is deliberately less poker-correct than a
        # sharper prior -- its job is to bias search, not to play well, and
        # t=0.10 plays far better poker while scoring WORSE than no prior at all.
        assert p[0] < 0.10, "a strong hand should not be folding often"

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


class TestComposition:
    """Equity and a trained prior answer different questions, so they add.

    The equity guess covers rows nothing has reached; the trained prior covers
    rows it has a real opinion about. Replacing one with the other would discard
    the guess exactly where the trained prior is weakest, which is the overlap
    worth keeping.
    """

    def test_adding_a_base_raises_every_row_it_touches(self):
        from src.pipeline.services.warm_start import regrets_encoding

        strategy = np.array([2.0, 1.0, 1.0, 3.0, 1.0])
        starts, widths = np.array([0, 3]), np.array([3, 2])
        alone, _ = regrets_encoding(strategy, starts, widths, 1000.0)
        base = np.full(5, 100.0)
        with_base, _ = regrets_encoding(strategy, starts, widths, 1000.0, base_regrets=base)
        assert (with_base > alone).all()
        assert with_base.sum() == pytest.approx(alone.sum() + base.sum())

    def test_a_row_the_trained_prior_missed_still_carries_the_guess(self):
        """The whole point of composing: a row with no trained opinion is not
        left uniform just because the trained prior never reached it."""
        from src.pipeline.services.warm_start import regrets_encoding

        strategy = np.array([2.0, 1.0, 1.0, 0.0, 0.0])  # row 1 unseen by the prior
        starts, widths = np.array([0, 3]), np.array([3, 2])
        alone, seeded_alone = regrets_encoding(strategy, starts, widths, 1000.0)
        base = np.array([0.0, 0.0, 0.0, 70.0, 30.0])
        composed, seeded_both = regrets_encoding(
            strategy, starts, widths, 1000.0, base_regrets=base
        )
        assert alone[3:].sum() == 0.0, "unseen row is empty without a base"
        assert composed[3:].tolist() == [70.0, 30.0]
        assert not seeded_alone[1]
        assert seeded_both[1], "composing must mark the row as carrying a prior"


class TestTheChannelThatMatters:
    """Evaluation reads ``strategy_sum`` and never ``regrets``.

    That is why a seeded prior plays uniform on a row training never reached --
    and, measured, why it does not matter: seeding ``strategy_sum`` to fix it
    bought -4.3 +/- 8.9 mbb at 30M, because a cold run already covers 99.2% of
    rows by then. The prior earns its keep by steering training, so what these
    pin is that the seeded regrets are one distribution at one scale.
    """

    def test_seeding_scales_one_guess_rather_than_making_two(self):
        from src.pipeline.services.equity_prior import seed_regrets
        from tests.test_helpers import DummyCardAbstraction, make_test_config

        tree = _test_tree(make_test_config(), DummyCardAbstraction())
        unit, _ = seed_regrets(tree, 1.0)
        scaled, _ = seed_regrets(tree, 250.0)
        np.testing.assert_allclose(scaled, unit * 250.0, rtol=1e-12)

    def test_a_regret_only_row_is_read_as_uniform_by_the_evaluator(self):
        """The behaviour the fallback existed to change, kept as documentation of
        why seeding regrets alone is the whole mechanism."""
        from src.engine.solver.numba_ops import average_strategy

        played = average_strategy(np.zeros(len(MENU)))
        np.testing.assert_allclose(played, np.full(len(MENU), 1 / len(MENU)))


class TestTheMeasuredDefaults:
    """The temperature default is a MEASURED optimum, not a taste.

    3 training seeds x 6 board seeds at 30M, mbb/g vs a cold control:
    0.10 +17.9 | 0.25 -80.8 | 0.50 -101.7 | 0.75 -73.3 | 1.50 -57.0 | uniform +18.5.
    Both extremes lose to not seeding at all, so a drift in either direction
    turns a 100 mbb gain into a loss.
    """

    def test_the_default_is_the_measured_peak(self):
        assert DEFAULT_TEMPERATURE == 0.50

    def test_the_default_is_softer_than_the_sharp_arm_that_lost(self):
        """t=0.10 scored +17.9 vs cold -- worse than no prior at all."""
        assert DEFAULT_TEMPERATURE > 0.10

    def test_the_default_still_carries_information(self):
        """A guess this flat must stay far from uniform, which scored +18.5."""
        p = strength_policy(0.95, action_aggression(MENU, POT), DEFAULT_TEMPERATURE)
        assert p.max() > 1.5 / len(MENU), p
