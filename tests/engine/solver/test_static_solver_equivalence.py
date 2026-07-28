"""Does the flat ragged layout store exactly what a naive layout would?

The risky part of ``StaticArrayStorage`` is arithmetic: ``slot_offset[n] +
bucket * num_actions[n]``. An off-by-one there does not crash — it silently
leaks regret between unrelated infosets, and training still "works". So the
check here is against an oracle with no layout at all: a dict of independently
allocated arrays, one per ``(node_id, bucket)``. Both back the same solver
through the same identity seam, so any divergence is the layout arithmetic and
nothing else.

This is the acceptance test for the storage increment. It is deliberately
stronger than "kernel conformance still passes": the conformance suite swaps out
the HUNL state machine entirely and therefore never exercises the betting tree.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.state import Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.infoset import InfoSet
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import (
    REGRET_DTYPE,
    STRATEGY_DTYPE,
    StaticArrayStorage,
)
from tests.test_helpers import make_test_config


class BucketsByStreet:
    """Deterministic, cheap stand-in for the equity abstraction."""

    def __init__(self, counts):
        self.counts = counts

    def get_bucket(self, hole_cards, board, street):
        h = hash((repr(hole_cards[0]), repr(hole_cards[1]), repr(board[0]), street.name))
        return h % self.counts[street]

    def num_buckets(self, street):
        return self.counts[street]


class DictOracleStorage:
    """Reference storage with no layout arithmetic whatsoever.

    Each infoset gets its own freshly allocated arrays, keyed by a tuple. Slower
    and larger than the flat backend, and trivially correct — which is the point.
    Same dtypes as the real backend so float32 rounding cannot masquerade as a
    layout bug.
    """

    def __init__(self, tree):
        self.tree = tree
        self._infosets: dict[tuple[int, int], InfoSet] = {}
        self.reach_counts: dict[tuple[int, int], int] = {}

    def infoset_at(self, node_id: int, bucket: int) -> InfoSet:
        cached = self._infosets.get((node_id, bucket))
        if cached is not None:
            return cached
        node = self.tree.nodes[node_id]
        infoset = InfoSet(None, node.legal_actions, allocate_arrays=False)
        infoset.regrets = np.zeros(node.num_actions, dtype=REGRET_DTYPE)
        infoset.strategy_sum = np.zeros(node.num_actions, dtype=STRATEGY_DTYPE)
        infoset.node_id = node_id
        infoset.bucket = bucket
        self._infosets[(node_id, bucket)] = infoset
        return infoset

    def num_touched_infosets(self) -> int:
        return len(self._infosets)


COUNTS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}
ITERATIONS = 120


def _build_solver(storage_factory, *, seed=42, weighting="dcfr"):
    config = make_test_config(
        seed=seed,
        small_blind=1,
        big_blind=2,
        starting_stack=20,
        iteration_weighting=weighting,
    )
    action_model = ActionModel(config)
    abstraction = BucketsByStreet(COUNTS)
    from src.core.game.rules import GameRules

    rules = GameRules(small_blind=config.game.small_blind, big_blind=config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, abstraction, starting_stack=config.game.starting_stack
    )
    storage = storage_factory(tree)
    solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)
    return solver, tree, storage


def _run(solver, iterations=ITERATIONS, seed=42):
    """Train under a fully pinned RNG so two backends see identical sampling."""
    random.seed(seed)
    np.random.seed(seed)
    for _ in range(iterations):
        solver.train_iteration()


@pytest.fixture(scope="module")
def trained_pair():
    flat_solver, tree, flat = _build_solver(lambda t: StaticArrayStorage(t))
    _run(flat_solver)

    oracle_solver, _, oracle = _build_solver(lambda t: DictOracleStorage(t))
    _run(oracle_solver)

    yield tree, flat, oracle, flat_solver, oracle_solver
    flat.close()


class TestLayoutEquivalence:
    def test_training_actually_happened(self, trained_pair):
        _, flat, oracle, _, _ = trained_pair
        assert oracle.num_touched_infosets() > 20
        assert flat.num_touched_infosets() == oracle.num_touched_infosets()
        assert np.count_nonzero(flat.regrets) > 0

    def test_regrets_match_the_oracle_exactly(self, trained_pair):
        _, flat, oracle, _, _ = trained_pair
        for (node_id, bucket), ref in oracle._infosets.items():
            got = flat.infoset_at(node_id, bucket)
            np.testing.assert_array_equal(
                got.regrets,
                ref.regrets,
                err_msg=f"regret mismatch at node={node_id} bucket={bucket}",
            )

    def test_strategy_sums_match_the_oracle_exactly(self, trained_pair):
        _, flat, oracle, _, _ = trained_pair
        for (node_id, bucket), ref in oracle._infosets.items():
            got = flat.infoset_at(node_id, bucket)
            np.testing.assert_array_equal(
                got.strategy_sum,
                ref.strategy_sum,
                err_msg=f"strategy mismatch at node={node_id} bucket={bucket}",
            )

    def test_untouched_rows_stayed_zero(self, trained_pair):
        """No stray writes outside the infosets the traversal actually visited."""
        tree, flat, oracle, _, _ = trained_pair
        touched = set(oracle._infosets)
        for node in tree.nodes:
            for bucket in range(tree.num_buckets(node.street)):
                if (node.node_id, bucket) in touched:
                    continue
                start, end = tree.slots(node.node_id, bucket)
                assert not flat.regrets[start:end].any(), (
                    f"untouched infoset node={node.node_id} bucket={bucket} was written"
                )
                assert not flat.strategy_sum[start:end].any()


class TestDeterminism:
    def test_same_seed_gives_identical_arrays(self):
        a_solver, _, a = _build_solver(lambda t: StaticArrayStorage(t))
        b_solver, _, b = _build_solver(lambda t: StaticArrayStorage(t))
        try:
            _run(a_solver, iterations=60)
            _run(b_solver, iterations=60)
            np.testing.assert_array_equal(a.regrets, b.regrets)
            np.testing.assert_array_equal(a.strategy_sum, b.strategy_sum)
        finally:
            a.close()
            b.close()

    def test_different_seed_diverges(self):
        """Guard against the equality tests passing because nothing is written."""
        a_solver, _, a = _build_solver(lambda t: StaticArrayStorage(t))
        b_solver, _, b = _build_solver(lambda t: StaticArrayStorage(t))
        try:
            _run(a_solver, iterations=60, seed=1)
            _run(b_solver, iterations=60, seed=2)
            assert not np.array_equal(a.regrets, b.regrets)
        finally:
            a.close()
            b.close()


class TestNoDroppedUpdates:
    def test_drop_counter_stays_at_zero(self, trained_pair):
        """The measured 39-74% drop rate has no code path in this design."""
        _, _, _, flat_solver, _ = trained_pair
        assert flat_solver.dropped_unknown_id_updates == 0
        assert flat_solver.applied_updates > 0

    @pytest.mark.parametrize("weighting", ["none", "linear", "dcfr"])
    def test_layout_holds_across_weighting_schemes(self, weighting):
        """Each scheme drives a different write pattern through the kernel."""
        flat_solver, _, flat = _build_solver(lambda t: StaticArrayStorage(t), weighting=weighting)
        oracle_solver, _, oracle = _build_solver(
            lambda t: DictOracleStorage(t), weighting=weighting
        )
        try:
            _run(flat_solver, iterations=60)
            _run(oracle_solver, iterations=60)
            for (node_id, bucket), ref in oracle._infosets.items():
                np.testing.assert_array_equal(flat.infoset_at(node_id, bucket).regrets, ref.regrets)
        finally:
            flat.close()


class TestConvenienceConstructor:
    def test_build_wires_tree_storage_and_solver(self):
        """``StaticTreeSolver.build`` is the natural entry point; keep it exercised."""
        config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
        abstraction = BucketsByStreet(COUNTS)
        solver = StaticTreeSolver.build(ActionModel(config), abstraction, config)
        try:
            assert solver.tree is solver.storage.tree
            assert solver.storage.regrets.shape == (solver.tree.num_slots,)
            for _ in range(20):
                solver.train_iteration()
            assert solver.num_infosets() > 0
            assert solver.dropped_unknown_id_updates == 0
        finally:
            solver.storage.close()
