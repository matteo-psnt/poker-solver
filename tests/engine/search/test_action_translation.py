"""Tests for off-tree action translation."""

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.actions import raises
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.engine.search.action_translation import translate_action_distribution
from src.shared.config import Config


def _initial_state():
    rules = GameRules(small_blind=1, big_blind=2)
    hole_cards = (
        (Card.new("As"), Card.new("Kh")),
        (Card.new("Qd"), Card.new("Jc")),
    )
    return rules.create_initial_state(starting_stack=200, hole_cards=hole_cards, button=0), rules


def test_nearest_translation_is_deterministic():
    state, rules = _initial_state()
    config = Config.default().merge({"action_model": {"off_tree_mapping": "nearest"}})
    model = ActionModel(config)

    dist = translate_action_distribution(state, raises(7), model, rules)
    assert len(dist) == 1
    mapped, prob = dist[0]
    assert prob == 1.0
    assert mapped in rules.get_legal_actions(state, action_model=model)


def test_probabilistic_translation_interpolates_between_sizes():
    state, rules = _initial_state()
    config = Config.default().merge({"action_model": {"off_tree_mapping": "probabilistic"}})
    model = ActionModel(config)

    dist = translate_action_distribution(state, raises(5), model, rules)
    assert len(dist) == 2
    assert abs(sum(prob for _, prob in dist) - 1.0) < 1e-9
    assert all(action in rules.get_legal_actions(state, action_model=model) for action, _ in dist)

    # Pin the pseudo-harmonic value (Ganzfried & Sandholm): raise-to-5 between
    # sizes 4 and 6 at pot 3 gives a=4/3, b=2, x=5/3 in pot units, so
    # P(lower) = (b-x)(1+a) / ((b-a)(1+x)) = 7/16. Linear interpolation would
    # give 0.5 — without this assertion a regression to linear is silent.
    (lower, lower_prob), (upper, upper_prob) = sorted(dist, key=lambda d: d[0].amount)
    assert (lower.amount, upper.amount) == (4, 6)
    assert lower_prob == pytest.approx(7 / 16)
    assert upper_prob == pytest.approx(9 / 16)
