"""Tests for the node-template ActionModel."""

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType
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
    state, rules = _initial_state()
    model = ActionModel(Config.default())
    actions = rules.get_legal_actions(state, action_model=model)
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


def _three_bets_to_bb(token: str) -> list[float]:
    """The big blind's 3-bet totals over a 2 bb open, in big blinds."""
    config = Config.default().merge(
        {"action_model": {"preflop_templates": {"bb_vs_open": ["fold", "call", token]}}}
    )
    model = ActionModel(config)
    state, rules = _initial_state()
    state = state.apply_action(Action(ActionType.RAISE, 2), rules)
    return [
        (200 - state.apply_action(action, rules).stacks[1]) / 2
        for action in rules.get_legal_actions(state, action_model=model)
        if action.type is ActionType.RAISE
    ]


def test_x_open_scales_the_call_owed():
    """What every fielded tree carries: 3.5x over a 2 bb open is to 4.5 bb."""
    assert _three_bets_to_bb("3.5x_open") == [4.5]


def test_x_to_scales_the_wager_faced():
    assert _three_bets_to_bb("3.5x_to") == [7.0]
    assert _three_bets_to_bb("5x_to") == [10.0]
