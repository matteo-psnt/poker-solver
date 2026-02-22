"""Tests for the node-template ActionModel."""

from src.core.actions.action_model import ActionModel
from src.core.game.actions import ActionType
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.shared.config import Config


def _initial_state():
    rules = GameRules(small_blind=1, big_blind=2)
    hole_cards = (
        (Card.new("As"), Card.new("Kh")),
        (Card.new("Qd"), Card.new("Jc")),
    )
    return rules.create_initial_state(starting_stack=200, hole_cards=hole_cards, button=0), rules


def test_action_model_preflop_legal_actions():
    state, _ = _initial_state()
    model = ActionModel(Config.default())
    actions = model.get_legal_actions(state)
    types = {a.type for a in actions}

    assert ActionType.FOLD in types
    assert ActionType.CALL in types
    assert ActionType.RAISE in types


def test_action_model_hash_stable_for_same_config():
    model1 = ActionModel(Config.default())
    model2 = ActionModel(Config.default())
    assert model1.get_config_hash() == model2.get_config_hash()


def test_preflop_open_sizes_exposed():
    model = ActionModel(Config.default())
    opens = model.get_preflop_open_sizes_bb()
    assert opens
    assert all(size > 0 for size in opens)
