"""An oversized preflop raise against the fielded tree, in big blinds.

Built on `production_opens` at 100 bb because the bug is in its sizes: the tree's
3-bets over a 2 bb open are to 4.5 and 5.5 bb, so the sizes the live field uses
either sit near them (2x, 3x) or are a pot it never trained (5x).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.interfaces.chipzen.adapter import reconstruct, wire_action
from src.interfaces.chipzen.protocol import GameConfig, TurnState
from src.interfaces.table import oversized, table_scale
from src.shared.config.loader import load_training_config

DEPTH = 100


@pytest.fixture(scope="module")
def tree():
    config = load_training_config("production_opens")
    config = config.merge({"game": {"starting_stack": DEPTH * config.game.big_blind}})
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    return SimpleNamespace(config=config, rules=rules, action_model=ActionModel(config))


@pytest.fixture(scope="module")
def facing_open(tree):
    """The big blind facing a small-blind open to 2 bb, and its menu."""
    bb = tree.config.game.big_blind
    hole = ((Card.new("As"), Card.new("Kd")), (Card.new("7c"), Card.new("2h")))
    state = tree.rules.create_initial_state(
        starting_stack=tree.config.game.starting_stack, hole_cards=hole, button=0
    )
    state = state.apply_action(Action(ActionType.RAISE, bb), tree.rules)
    return state, tree.rules.get_legal_actions(state, action_model=tree.action_model)


@pytest.mark.parametrize(
    ("to_bb", "expected"), [(4, False), (6, False), (7, False), (8, True), (10, True), (25, True)]
)
def test_only_a_raise_well_past_the_tree_reads_as_oversized(tree, facing_open, to_bb, expected):
    """Up to 3.5x the open stays a raise: our largest is to 5.5 bb (3.5 bb on top)."""
    state, legal = facing_open
    bb = tree.config.game.big_blind
    assert oversized(Action(ActionType.RAISE, (to_bb - 2) * bb), state, legal) is expected


def _scale(tree):
    bb = tree.config.game.big_blind
    table = GameConfig.parse(
        {
            "variant": "nlhe",
            "starting_stack": DEPTH * bb,
            "small_blind": bb // 2,
            "big_blind": bb,
            "num_players": 2,
        }
    )
    return table_scale(table, tree)


def _turn(bb: int, three_bet_to: int) -> TurnState:
    """Our small blind, having opened to 2 bb, facing their 3-bet."""
    stack = DEPTH * bb
    entry = {"phase": "preflop", "is_timeout": False}
    return TurnState.parse(
        {
            "hand_number": 1,
            "phase": "preflop",
            "board": [],
            "your_hole_cards": ["Ah", "Kd"],
            "pot": 2 * bb + three_bet_to,
            "your_stack": stack - 2 * bb,
            "opponent_stacks": [stack - three_bet_to],
            "to_call": three_bet_to - 2 * bb,
            "min_raise": 2 * three_bet_to - 2 * bb,
            "max_raise": stack,
            "action_history": [
                {**entry, "seat": 0, "action": "post_small_blind", "amount": bb // 2},
                {**entry, "seat": 1, "action": "post_big_blind", "amount": bb},
                {**entry, "seat": 0, "action": "raise", "amount": 2 * bb},
                {**entry, "seat": 1, "action": "raise", "amount": three_bet_to},
            ],
        }
    )


def test_a_5x_3bet_is_answered_as_a_jam_and_calling_it_ships(tree):
    bb = tree.config.game.big_blind
    turn = _turn(bb, 10 * bb)
    scale = _scale(tree)
    spot = reconstruct(tree, turn, seat=0, scale=scale)
    assert spot.read_as_jam
    assert not spot.truncated
    legal = tree.rules.get_legal_actions(spot.state, action_model=tree.action_model)
    calls = [action for action in legal if action.type is not ActionType.FOLD]
    assert [action.type for action in calls] == [ActionType.ALL_IN]
    assert wire_action(calls[0], turn, spot) == {
        "action": "raise",
        "params": {"amount": turn.max_raise},
    }
    assert wire_action(Action(ActionType.FOLD), turn, spot)["action"] == "fold"


def test_a_3x_3bet_replays_as_before(tree):
    bb = tree.config.game.big_blind
    turn = _turn(bb, 6 * bb)
    scale = _scale(tree)
    spot = reconstruct(tree, turn, seat=0, scale=scale)
    assert not spot.read_as_jam
    assert wire_action(Action(ActionType.CALL), turn, spot) == {"action": "call", "params": {}}
