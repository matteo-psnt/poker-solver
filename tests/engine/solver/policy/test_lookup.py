"""Tests for the canonical blueprint-strategy lookup.

These pin the unified filtering predicate (candidate membership AND rules
validity) and the duplicate-action aggregation that every consumer — training
traversal, blueprint sampling, resolver, range inference, LBR opponent model —
now shares.
"""

from typing import cast

import numpy as np

from src.core.game.actions import bet, call, fold
from src.core.game.rules import GameRules
from src.core.game.state import GameState, Street
from src.engine.solver.infoset.model import InfoSet, InfoSetKey
from src.engine.solver.policy.lookup import blueprint_action_distribution, filter_stored_actions


class _StubRules:
    """Validity predicate under test control; real GameRules needs a full state."""

    def __init__(self, invalid=()):
        self.invalid = set(invalid)

    def is_action_valid(self, state, action):
        del state
        return action not in self.invalid


def _rules(invalid=()) -> GameRules:
    return cast("GameRules", _StubRules(invalid))


def _infoset(actions):
    return InfoSet(InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1), actions)


STATE = cast("GameState", object())  # opaque to _StubRules


class TestFilterStoredActions:
    def test_membership_and_validity_both_required(self):
        stored = [fold(), call(), bet(50), bet(100)]
        infoset = _infoset(stored)
        rules = _rules(invalid=[bet(100)])

        # bet(200) fails membership (not stored), bet(100) fails validity.
        indices, actions = filter_stored_actions(
            infoset, STATE, rules, [fold(), call(), bet(100), bet(200)]
        )

        assert actions == [fold(), call()]
        assert indices == [0, 1]

    def test_empty_when_nothing_survives(self):
        infoset = _infoset([bet(50)])
        indices, actions = filter_stored_actions(infoset, STATE, _rules(), [fold(), call()])
        assert indices == []
        assert actions == []


class TestBlueprintActionDistribution:
    def test_missing_infoset_is_none(self):
        assert (
            blueprint_action_distribution(None, STATE, _rules(), [fold(), call()], use_average=True)
            is None
        )

    def test_no_survivors_is_none(self):
        infoset = _infoset([bet(50)])
        assert (
            blueprint_action_distribution(
                infoset, STATE, _rules(), [fold(), call()], use_average=True
            )
            is None
        )

    def test_distribution_normalized_over_survivors(self):
        stored = [fold(), call(), bet(50)]
        infoset = _infoset(stored)
        infoset.strategy_sum[:] = [1.0, 3.0, 6.0]  # avg strategy 0.1 / 0.3 / 0.6

        dist = blueprint_action_distribution(
            infoset, STATE, _rules(invalid=[bet(50)]), stored, use_average=True
        )

        # bet(50) filtered out; remaining mass renormalized over fold/call.
        assert dist is not None
        assert set(dist) == {fold(), call()}
        assert np.isclose(dist[fold()], 1.0 / 4.0)
        assert np.isclose(dist[call()], 3.0 / 4.0)
        assert np.isclose(sum(dist.values()), 1.0)

    def test_duplicate_stored_actions_aggregate_mass(self):
        # Placeholder rows (e.g. cache-miss reconstruction) can repeat an action;
        # its probability mass must sum, not overwrite.
        stored = [fold(), fold(), call()]
        infoset = _infoset(stored)
        infoset.strategy_sum[:] = [1.0, 2.0, 1.0]

        dist = blueprint_action_distribution(
            infoset, STATE, _rules(), [fold(), call()], use_average=True
        )

        assert dist is not None
        assert np.isclose(dist[fold()], 0.75)
        assert np.isclose(dist[call()], 0.25)

    def test_current_strategy_uses_regrets(self):
        stored = [fold(), call()]
        infoset = _infoset(stored)
        infoset.regrets[:] = [1.0, 3.0]
        infoset.strategy_sum[:] = [1.0, 0.0]  # average would say all-fold

        dist = blueprint_action_distribution(infoset, STATE, _rules(), stored, use_average=False)

        assert dist is not None
        assert np.isclose(dist[fold()], 0.25)
        assert np.isclose(dist[call()], 0.75)


class TestBlueprintPolicyTable:
    """The vectorised table is the per-bucket answer, row for row, byte for byte.

    Exact-BR fills one of these per betting context and its scores are a
    zero-variance gate, so "close" is not the bar -- the same floats are.
    """

    @staticmethod
    def _decision_states(solver, playouts: int = 40):
        rng = np.random.default_rng(11)
        rules = solver.rules
        states = []
        for _ in range(playouts):
            state = solver.deal_initial_state()
            while not state.is_terminal:
                if solver.is_chance_node(state):
                    state = solver.sample_chance_outcome(state)
                    continue
                states.append(state)
                legal = rules.get_legal_actions(state, solver.action_model)
                state = rules.apply_action(state, legal[int(rng.integers(len(legal)))])
        return states

    def _assert_table_matches(self, solver):
        from src.engine.solver.policy.lookup import blueprint_policy_table

        source = solver.policy_source
        rules = solver.rules
        compared = 0
        for state in self._decision_states(solver):
            legal = rules.get_legal_actions(state, solver.action_model)
            table, missing = blueprint_policy_table(source.rows_at(state), state, rules, legal)
            for bucket in range(source.num_buckets(state.street)):
                distribution = blueprint_action_distribution(
                    source.infoset_at(state, bucket), state, rules, legal, use_average=True
                )
                if distribution is None:
                    expected = np.full(len(legal), 1.0 / len(legal))
                else:
                    expected = np.array([distribution.get(a, 0.0) for a in legal])
                assert missing[bucket] == (distribution is None)
                assert np.array_equal(table[bucket], expected), (state, bucket)
                compared += 1
        return compared

    def test_trained_rows_match_the_per_bucket_lookup(self):
        from tests.test_helpers import build_trained_test_solver

        solver = build_trained_test_solver(6)
        assert self._assert_table_matches(solver) > 1000
        # The comparison must have crossed both kinds of row.
        assert 0 < int(solver.storage.visited.sum()) < solver.storage.visited.size

    def test_a_visited_row_summing_to_zero_is_uniform_over_survivors(self):
        from tests.test_helpers import build_trained_test_solver

        solver = build_trained_test_solver(0)
        # Every row is an untouched allocation; mark them all visited so the
        # zero-sum branch of average_strategy -- not the missing fallback -- is
        # what the table has to reproduce.
        solver.storage.visited[:] = 1
        assert self._assert_table_matches(solver) > 1000
