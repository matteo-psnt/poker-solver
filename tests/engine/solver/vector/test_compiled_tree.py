"""The compiled tree has to agree with the real game, not merely with itself.

``BettingTree`` records decision nodes; this compilation adds the edges and
terminals a vector pass walks instead of rediscovering them by re-applying
actions. Every one of those additions is a new opportunity to disagree with
``GameRules`` — a mislabelled fold, a payoff of the wrong sign, an edge pointing
at the wrong child — and none of those would crash. They would train.

So the checks here are against the live rules engine: walk real states, and
assert the compiled arrays say what the game says.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.vector.compiled_tree import (
    EDGE_TO_NODE,
    EDGE_TO_TERMINAL,
    TerminalKind,
    compile_tree,
)
from tests.test_helpers import make_test_config

BOARD = (Card("2c"), Card("7d"), Card("9h"), Card("4s"), Card("Ts"))
HOLE = ((Card("As"), Card("Kd")), (Card("Qh"), Card("Jc")))
BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}
STACK = 20


@pytest.fixture(scope="module")
def rules():
    return GameRules(small_blind=1, big_blind=2)


@pytest.fixture(scope="module")
def tree(rules):
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    return BettingTree(
        rules,
        ActionModel(config),
        starting_stack=STACK,
        buckets_per_street=BUCKETS,
    )


@pytest.fixture(scope="module")
def compiled(tree, rules):
    return compile_tree(tree, rules)


def _settle(state):
    needed = state.street.board_card_count
    if len(state.board) >= needed:
        return state
    return state.replace(board=BOARD[:needed], validate=False)


def _root(rules):
    return rules.create_initial_state(starting_stack=STACK, hole_cards=HOLE, button=0)


class TestStructure:
    def test_every_edge_is_assigned_and_in_range(self, compiled):
        assert (compiled.edge_kind != -1).all()
        nodes = compiled.edge_kind == EDGE_TO_NODE
        terminals = compiled.edge_kind == EDGE_TO_TERMINAL
        assert (compiled.edge_target[nodes] < compiled.num_nodes).all()
        assert (compiled.edge_target[terminals] < compiled.num_terminals).all()
        assert (compiled.edge_target >= 0).all()

    def test_edge_count_matches_the_action_counts(self, compiled):
        assert compiled.num_edges == int(compiled.tree.num_actions.sum())

    def test_the_betting_tree_is_a_tree_not_a_dag(self, compiled):
        """Reach can be assigned to a child rather than accumulated into it.

        The forward pass writes child ranges instead of adding them, which is
        only sound while no node has two parents. If this ever fails the kernel
        needs ``np.add.at`` there, so the assertion is the guard, not a
        curiosity.
        """
        assert not compiled.is_dag
        assert compiled.parent_count[0] == 0
        assert (compiled.parent_count[1:] == 1).all()

    def test_depth_strictly_increases_along_every_edge(self, compiled):
        for node_id in range(compiled.num_nodes):
            start, end = compiled.edges_of(node_id)
            for edge in range(start, end):
                if compiled.edge_kind[edge] == EDGE_TO_NODE:
                    assert compiled.depth[node_id] < compiled.depth[compiled.edge_target[edge]]

    def test_levels_partition_the_nodes(self, compiled):
        seen = np.concatenate(
            [compiled.nodes_at_level(level) for level in range(compiled.num_levels)]
        )
        assert sorted(seen.tolist()) == list(range(compiled.num_nodes))


class TestAgreementWithTheRules:
    def test_edges_lead_where_applying_the_action_leads(self, compiled, rules, tree):
        """Follow random action sequences and check the arrays match the walk."""
        rng = random.Random(7)
        for _ in range(40):
            state = _settle(_root(rules))
            for _ in range(30):
                node_id = tree.node_id(state)
                actions = tree.legal_actions(node_id)
                slot = rng.randrange(len(actions))
                start, _ = compiled.edges_of(node_id)
                edge = start + slot

                child = _settle(rules.apply_action(state, actions[slot]))
                if child.is_terminal:
                    assert compiled.edge_kind[edge] == EDGE_TO_TERMINAL
                    break
                assert compiled.edge_kind[edge] == EDGE_TO_NODE
                assert compiled.edge_target[edge] == tree.node_id(child)
                state = child

    def test_fold_terminals_carry_the_payoff_the_rules_compute(self, compiled, rules, tree):
        """A fold's value is public, so it must match ``get_payoff`` exactly.

        ``terminal_value`` is button-relative; the fixture deals button 0, so
        the button's payoff is player 0's.
        """
        checked = 0
        rng = random.Random(11)
        for _ in range(60):
            state = _settle(_root(rules))
            for _ in range(30):
                node_id = tree.node_id(state)
                actions = tree.legal_actions(node_id)
                slot = rng.randrange(len(actions))
                start, _ = compiled.edges_of(node_id)
                child = _settle(rules.apply_action(state, actions[slot]))

                if child.is_terminal:
                    terminal = compiled.edge_target[start + slot]
                    if child.ended_by_fold:
                        assert compiled.terminal_kind[terminal] == TerminalKind.FOLD
                        assert compiled.terminal_value[terminal] == pytest.approx(
                            child.get_payoff(0, rules)
                        )
                        checked += 1
                    break
                state = child
        assert checked > 10

    def test_showdown_terminals_carry_half_the_pot(self, compiled, rules, tree):
        """The winner is unknown without cards, so only the stake is stored."""
        checked = 0
        rng = random.Random(13)
        for _ in range(60):
            state = _settle(_root(rules))
            for _ in range(30):
                node_id = tree.node_id(state)
                actions = tree.legal_actions(node_id)
                slot = rng.randrange(len(actions))
                start, _ = compiled.edges_of(node_id)
                child = _settle(rules.apply_action(state, actions[slot]))

                if child.is_terminal:
                    terminal = compiled.edge_target[start + slot]
                    if not child.ended_by_fold:
                        assert compiled.terminal_kind[terminal] == TerminalKind.SHOWDOWN
                        assert compiled.terminal_value[terminal] == pytest.approx(child.pot / 2)
                        checked += 1
                    break
                state = child
        assert checked > 10
