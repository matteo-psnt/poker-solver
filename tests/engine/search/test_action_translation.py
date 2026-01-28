"""Tests for off-tree action translation."""

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
