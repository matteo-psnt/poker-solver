"""Do the derived walk arrays agree with the independent rules walk?

`WalkArrays` reads `BettingTree.edges`; `CompiledTree` re-applies every action
against `GameRules` to rediscover the same structure. That is two walks of the
same rules, and two walks can drift. Until they are collapsed into one, this is
the guard that makes the duplication safe rather than merely tolerated: where
the two overlap they must agree edge for edge.

The payoff checks go the other way — against `GameRules.get_payoff` on a
concrete terminal state — because `WalkArrays` carries per-seat payoffs that
`CompiledTree` reduces to one button-relative number, so there is nothing to
compare against for those.
"""

from __future__ import annotations

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.vector.compiled_tree import (
    EDGE_TO_NODE,
    EDGE_TO_TERMINAL,
    TerminalKind,
    compile_tree,
)
from src.engine.solver.vector.walk_arrays import WalkArrays
from tests.test_helpers import make_test_config

COUNTS = {Street.FLOP: 2, Street.TURN: 2, Street.RIVER: 2}


class Buckets:
    def get_bucket(self, hole_cards, board, street):
        return 0

    def num_buckets(self, street):
        return COUNTS.get(street, 169)


@pytest.fixture(scope="module")
def built():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(rules, ActionModel(config), Buckets(), starting_stack=20)
    return rules, tree, WalkArrays(tree), compile_tree(tree, rules)


class TestAgreesWithTheIndependentWalk:
    def test_the_edge_layout_is_the_same(self, built):
        _, _, walk, compiled = built
        assert walk.num_edges == compiled.num_edges
        assert (walk.edge_offset == compiled.edge_offset).all()

    def test_every_edge_agrees_on_terminal_versus_node(self, built):
        _, _, walk, compiled = built
        walk_is_terminal = walk.edge_terminal >= 0
        compiled_is_terminal = compiled.edge_kind == EDGE_TO_TERMINAL
        assert (walk_is_terminal == compiled_is_terminal).all()

    def test_every_decision_edge_names_the_same_child(self, built):
        _, _, walk, compiled = built
        nodes = compiled.edge_kind == EDGE_TO_NODE
        assert (walk.edge_child[nodes] == compiled.edge_target[nodes]).all()
        # And the terminal slots carry no child, so a reader cannot follow one.
        assert (walk.edge_child[~nodes] == -1).all()

    def test_the_two_terminal_tables_classify_edges_the_same_way(self, built):
        """Row ids differ (different dedup keys); the KIND per edge must not."""
        _, _, walk, compiled = built
        for edge in range(walk.num_edges):
            if compiled.edge_kind[edge] != EDGE_TO_TERMINAL:
                continue
            theirs = compiled.terminal_kind[compiled.edge_target[edge]]
            ours = walk.terminal_is_fold[walk.edge_terminal[edge]]
            assert ours == (1 if theirs == TerminalKind.FOLD else 0)


class TestTheThingsCompiledTreeDoesNotCarry:
    def test_deal_counts_match_the_street_transition(self, built):
        _, tree, walk, _ = built
        seen = set()
        for node_id, edges in enumerate(tree.edges):
            base = int(walk.edge_offset[node_id])
            for slot, edge in enumerate(edges):
                if edge.terminal is not None:
                    continue
                child_street = tree.nodes[edge.child_id].street
                owed = child_street.board_card_count - tree.nodes[node_id].street.board_card_count
                assert walk.edge_deal[base + slot] == owed
                seen.add(owed)
        assert seen == {0, 3, 1}, f"expected within-street, flop and turn/river deals; saw {seen}"

    def test_payoffs_are_what_the_rules_compute(self, built):
        """Per-seat, against `get_payoff` on the terminal state itself."""
        _, tree, walk, _ = built
        checked = 0
        for node_id, edges in enumerate(tree.edges):
            base = int(walk.edge_offset[node_id])
            for slot, edge in enumerate(edges):
                terminal = edge.terminal
                if terminal is None or not terminal.is_fold:
                    continue
                row = walk.edge_terminal[base + slot]
                assert tuple(walk.terminal_fold[row]) == terminal.fold
                assert walk.terminal_is_fold[row] == 1
                checked += 1
        assert checked > 10

    def test_a_showdown_row_carries_all_three_outcomes(self, built):
        _, _, walk, _ = built
        showdowns = walk.terminal_is_fold == 0
        assert showdowns.any()
        # Win beats tie beats lose, for both seats, on every showdown row.
        assert (walk.terminal_win[showdowns] > walk.terminal_tie[showdowns]).all()
        assert (walk.terminal_tie[showdowns] > walk.terminal_lose[showdowns]).all()

    def test_showdowns_owing_a_runout_are_recorded(self, built):
        _, _, walk, _ = built
        owed = walk.terminal_cards_to_deal[walk.terminal_is_fold == 0]
        assert (owed > 0).any(), "no all-in showdown before the river"
        assert (walk.terminal_cards_to_deal[walk.terminal_is_fold == 1] == 0).all()
