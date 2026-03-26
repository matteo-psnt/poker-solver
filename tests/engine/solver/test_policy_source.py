"""The bridge that lets evaluation read a tree-addressed blueprint.

The exact-BR engine used to build infoset keys inline, which hard-wired it to
one storage layout. These tests cover the seam that replaced that, and in
particular the ordering hazard it exposed: the BR engine enumerates preflop
buckets by INDEX and gathers policy rows through a combo->index map, so if that
map and the solver's own preflop ordering ever disagree, every preflop policy
row is attached to the wrong hand — silently, with no error anywhere.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.infoset_index import (
    NUM_PREFLOP_HANDS,
    bucket_of,
    preflop_hand_index,
    preflop_hand_string_at,
)
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.policy_source import TreePolicySource
from src.engine.solver.storage.static_array import StaticArrayStorage
from tests.test_helpers import make_test_config

BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}


class Buckets:
    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        return hash((repr(hole_cards[0]), repr(board[0]))) % BUCKETS[street]

    def num_buckets(self, street: Street) -> int:
        return BUCKETS[street]


def _solver(stack: int = 20, iterations: int = 300) -> StaticTreeSolver:
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=stack)
    action_model = ActionModel(config)
    abstraction = Buckets()
    rules = GameRules(small_blind=1, big_blind=2)
    tree = build_betting_tree(rules, action_model, abstraction, starting_stack=stack)
    solver = StaticTreeSolver(
        action_model, abstraction, StaticArrayStorage(tree), config, tree=tree
    )
    random.seed(1)
    np.random.seed(1)
    for _ in range(iterations):
        solver.train_iteration()
    return solver


class TestPreflopOrderingIsShared:
    """The hazard: two independently-derived orderings of the 169 classes."""

    def test_br_combo_map_matches_the_solver_ordering(self):
        from src.pipeline.evaluation.public_tree_br import (
            _PREFLOP_CLASS_OF_COMBO,
            ALL_COMBOS,
        )

        expected = np.array([preflop_hand_index(combo) for combo in ALL_COMBOS])
        np.testing.assert_array_equal(_PREFLOP_CLASS_OF_COMBO, expected)

    def test_index_and_string_are_inverses(self):
        seen = set()
        for index in range(NUM_PREFLOP_HANDS):
            seen.add(preflop_hand_string_at(index))
        assert len(seen) == NUM_PREFLOP_HANDS

    def test_round_trip_through_a_concrete_combo(self):
        for cards, expected in (
            (("As", "Ah"), "AA"),
            (("As", "Ks"), "AKs"),
            (("As", "Kd"), "AKo"),
            (("2s", "2d"), "22"),
        ):
            combo = (Card.new(cards[0]), Card.new(cards[1]))
            assert preflop_hand_string_at(preflop_hand_index(combo)) == expected


class TestTreePolicySource:
    def test_resolves_a_trained_infoset(self):
        solver = _solver()
        try:
            source = TreePolicySource(solver.tree, solver.storage, solver.card_abstraction)
            state = solver.deal_initial_state()
            infoset = source.infoset_at(state, preflop_hand_index(state.hole_cards[0]))
            assert infoset is not None
            assert len(infoset.regrets) == infoset.num_actions
        finally:
            solver.storage.close()

    def test_reports_the_trees_bucket_counts(self):
        solver = _solver(iterations=1)
        try:
            source = TreePolicySource(solver.tree, solver.storage, solver.card_abstraction)
            assert source.num_buckets(Street.PREFLOP) == NUM_PREFLOP_HANDS
            assert source.num_buckets(Street.RIVER) == BUCKETS[Street.RIVER]
        finally:
            solver.storage.close()

    def test_out_of_range_bucket_returns_none_not_a_wrong_row(self):
        """An oversized bucket must not silently alias another node's infoset."""
        solver = _solver(iterations=1)
        try:
            source = TreePolicySource(solver.tree, solver.storage, solver.card_abstraction)
            state = solver.deal_initial_state()
            assert source.infoset_at(state, NUM_PREFLOP_HANDS) is None
            assert source.infoset_at(state, -1) is None
        finally:
            solver.storage.close()

    def test_lookup_does_not_mark_coverage(self):
        """Scoring must not make an untrained tree look explored."""
        solver = _solver(iterations=1)
        try:
            source = TreePolicySource(solver.tree, solver.storage, solver.card_abstraction)
            before = solver.storage.num_touched_infosets()
            state = solver.deal_initial_state()
            for bucket in range(NUM_PREFLOP_HANDS):
                source.infoset_at(state, bucket)
            assert solver.storage.num_touched_infosets() == before
        finally:
            solver.storage.close()

    def test_off_tree_state_raises_rather_than_scoring_a_different_game(self):
        solver = _solver(iterations=1)
        try:
            source = TreePolicySource(solver.tree, solver.storage, solver.card_abstraction)
            deep = solver.rules.create_initial_state(
                200, ((Card.new("As"), Card.new("Kd")), (Card.new("Qh"), Card.new("Jc"))), button=0
            )
            action_model = solver.action_model
            for _ in range(3):
                actions = solver.rules.get_legal_actions(deep, action_model=action_model)
                raises = [a for a in actions if a.amount and a.amount > 0]
                if not raises:
                    break
                deep = solver.rules.apply_action(deep, raises[-1])
            with pytest.raises(KeyError, match="off-tree"):
                source.infoset_at(deep, 0)
        finally:
            solver.storage.close()


class TestSourceSelection:
    def test_blueprint_exposes_a_tree_source(self):
        solver = _solver(iterations=1)
        try:
            assert isinstance(solver.policy_source, TreePolicySource)
        finally:
            solver.storage.close()

    def test_the_source_is_built_once(self):
        """Cached: it is reached once per decision on the play and search paths."""
        solver = _solver(iterations=1)
        try:
            assert solver.policy_source is solver.policy_source
        finally:
            solver.storage.close()


class TestExactBROverStaticStorage:
    """End to end: the artifact this bridge exists to unblock."""

    @pytest.mark.timeout(120)
    def test_exact_br_runs_and_resolves_every_lookup(self):
        from src.pipeline.evaluation.public_tree_br import (
            PublicBRConfig,
            compute_public_tree_br,
        )

        solver = _solver(iterations=400)
        try:
            result = compute_public_tree_br(
                solver,
                PublicBRConfig(num_flops=2, num_turns=1, num_rivers=1, board_seed=7),
                starting_stack=20,
            )
            assert np.isfinite(result.exploitability_mbb)
            assert result.exploitability_mbb > 0
            # A BROKEN bridge resolves nothing and falls back everywhere, so it
            # reads 1.0. What it must NOT be asserted to be is 0.0: the static
            # table allocates every row up front, so at 400 iterations a large
            # share of rows is genuinely unvisited and honestly reported as
            # fallback. Demanding 0.0 here would only pass by treating an
            # allocated-but-untrained row as a trained one, which is the very
            # conflation that made this diagnostic meaningless.
            assert result.missing_policy_mass < 0.9
        finally:
            solver.storage.close()


class TestStaticBlueprintCanPlay:
    """The runtime paths, not just the evaluators.

    resolver, heads_up_session and range_inference all used to build an
    InfoSetKey and call storage.get_infoset directly, which hard-wired them to
    the key-addressed backend: a tree-addressed blueprint could be trained and
    scored but not PLAYED. These are the tests that say it can.
    """

    def test_bucket_for_matches_what_the_solver_stores(self):
        """The bridge must resolve the same bucket the traversal wrote to."""
        solver = _solver(iterations=200)
        try:
            source = TreePolicySource(solver.tree, solver.storage, solver.card_abstraction)
            state = solver.deal_initial_state()
            player = state.current_player
            assert source.bucket_for(state, player) == preflop_hand_index(state.hole_cards[player])
        finally:
            solver.storage.close()

    def test_range_inference_runs_against_a_static_blueprint(self):
        from src.engine.search.range_inference import infer_ranges, update_ranges

        solver = _solver(iterations=400)
        try:
            state = solver.deal_initial_state()
            ranges = infer_ranges(state, solver)
            actions = solver.rules.get_legal_actions(state, action_model=solver.action_model)
            updated = update_ranges(
                state=state,
                ranges=ranges,
                observed_action=actions[0],
                blueprint=solver,
            )
            assert updated is not None
        finally:
            solver.storage.close()

    def test_source_buckets_hands_the_same_way_the_solver_does(self):
        """The source and the training kernel must partition hands identically.

        They reach it by different routes -- the source through the policy
        bridge, the kernel through its own lookup -- and a divergence would
        attach every scored policy row to a different hand than the one trained,
        silently and with no error anywhere.
        """
        solver = _solver(iterations=1)
        try:
            source = TreePolicySource(solver.tree, solver.storage, solver.card_abstraction)
            for _ in range(20):
                state = solver.deal_initial_state()
                player = state.current_player
                assert source.bucket_for(state, player) == bucket_of(
                    state, player, solver.card_abstraction
                )
        finally:
            solver.storage.close()

    def test_sample_action_from_strategy_works_on_a_static_blueprint(self):
        """The Blueprint protocol's own sampling method.

        StaticTreeSolver inherits it from MCCFRSolver, so a key-based lookup in
        there would leave a tree-addressed blueprint unplayable while every other
        runtime path worked — the gap that this test exists to keep closed.
        """
        solver = _solver(iterations=400)
        try:
            state = solver.deal_initial_state()
            legal = solver.rules.get_legal_actions(state, action_model=solver.action_model)
            for _ in range(10):
                assert solver.sample_action_from_strategy(state) in legal
        finally:
            solver.storage.close()
