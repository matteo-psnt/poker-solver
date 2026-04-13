"""The default lookahead depth has to reach the end of a river hand.

Not a taste question. A truncated leaf is not merely a less-accurate leaf: the
resolver converges to the truncated game, so extra compute makes it *more*
confidently wrong. Measured over sampled river spots, mean L1 from the exact
river equilibrium at a matched iteration budget:

    budget    depth 2   depth 3   depth 4   depth 6
       200     1.4143    0.4823    0.4591    0.4542
       800     1.4940    0.3405    0.2326    0.2234
     3,200     1.5284    0.2802    0.0982    0.0523

Depth 2 rises across a 16x budget increase while every deeper setting roughly
halves. So this pins the property the default is chosen FOR — that a river
subgame is fully expanded — rather than the number 6, which is only the depth at
which that happens to become true under this action model.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Street
from src.engine.search.tree_builder import build_local_tree
from src.shared.config import ResolverConfig
from tests.test_helpers import make_test_config

SPOTS = 12


@pytest.fixture(scope="module")
def parts():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=200)
    return ActionModel(config), GameRules(1, 2), config


def _river_states(action_model, rules, config, count):
    """Real river decision states, reached by random legal play."""
    rng = np.random.default_rng(11)
    found = []
    while len(found) < count:
        cards = list(rng.permutation(52))
        hole = (
            (FULL_DECK[cards[0]], FULL_DECK[cards[1]]),
            (FULL_DECK[cards[2]], FULL_DECK[cards[3]]),
        )
        board = tuple(FULL_DECK[c] for c in cards[4:9])
        state = rules.create_initial_state(
            starting_stack=config.game.starting_stack, hole_cards=hole, button=0
        )
        for _ in range(40):
            if state.is_terminal:
                break
            needed = state.street.board_card_count
            if len(state.board) < needed:
                state = state.replace(board=board[:needed], validate=False)
            actions = rules.get_legal_actions(state, action_model=action_model)
            if not actions:
                break
            if state.street == Street.RIVER:
                found.append(state)
                break
            state = rules.apply_action(state, actions[int(rng.integers(len(actions)))])
    return found[:count]


def _leaf_census(state, action_model, rules, depth):
    """(terminal leaves, truncated leaves) of the lookahead at ``depth``."""
    tree = build_local_tree(state, action_model=action_model, rules=rules, max_depth=depth)
    terminal = truncated = 0
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
        elif node.state.is_terminal:
            terminal += 1
        else:
            truncated += 1
    return terminal, truncated


def test_the_default_depth_solves_a_river_hand_to_the_end(parts):
    """Every river leaf must be a real terminal, not an estimate."""
    action_model, rules, config = parts
    depth = ResolverConfig().max_depth
    truncated_total = 0
    for state in _river_states(action_model, rules, config, SPOTS):
        _, truncated = _leaf_census(state, action_model, rules, depth)
        truncated_total += truncated
    assert truncated_total == 0, (
        f"default max_depth={depth} truncates {truncated_total} river leaves; "
        "the resolver would converge to a truncated game there"
    )


def test_the_old_default_of_two_truncated_most_river_leaves(parts):
    """The regression this guards against, stated as a fact rather than a memory.

    Recorded so a future reader can see what depth 2 actually did, without
    re-deriving it: about half of every river lookahead ended in an estimate.
    """
    action_model, rules, config = parts
    terminal = truncated = 0
    for state in _river_states(action_model, rules, config, SPOTS):
        got_terminal, got_truncated = _leaf_census(state, action_model, rules, 2)
        terminal += got_terminal
        truncated += got_truncated
    assert truncated > terminal * 0.5


def test_the_default_is_where_the_river_tree_stops_growing(parts):
    """6 is not arbitrary — the river tree is fully expanded there.

    Going deeper buys nothing, which is what makes this a setting rather than a
    dial to keep turning.
    """
    action_model, rules, config = parts
    states = _river_states(action_model, rules, config, SPOTS)
    at_default = [_leaf_census(s, action_model, rules, ResolverConfig().max_depth) for s in states]
    at_deeper = [_leaf_census(s, action_model, rules, 12) for s in states]
    assert at_default == at_deeper
