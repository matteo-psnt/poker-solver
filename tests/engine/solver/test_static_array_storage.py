"""StaticArrayStorage: the flat-array backend indexed by the betting tree.

The defect class most likely to hide here is an off-by-one in the ragged slot
layout — two infosets aliasing the same slots, or an infoset's view straddling a
neighbour's. Those are silent: training still runs, regrets just leak between
unrelated infosets. Several tests below exist only to make that loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.storage.static_array import StaticArrayStorage
from tests.test_helpers import make_test_config

BUCKETS = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


@pytest.fixture(scope="module")
def tree():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    rules = GameRules(small_blind=config.game.small_blind, big_blind=config.game.big_blind)
    return BettingTree(
        rules,
        ActionModel(config),
        starting_stack=config.game.starting_stack,
        buckets_per_street=BUCKETS,
    )


@pytest.fixture
def storage(tree):
    store = StaticArrayStorage(tree)
    yield store
    store.close()


class TestAllocation:
    def test_arrays_are_sized_exactly_to_the_tree(self, storage, tree):
        assert storage.regrets.shape == (tree.num_slots,)
        assert storage.strategy_sum.shape == (tree.num_slots,)
        assert storage.reach_counts.shape == (tree.num_rows,)
        assert storage.cumulative_utility.shape == (tree.num_rows,)

    def test_starts_zeroed(self, storage):
        assert not storage.regrets.any()
        assert not storage.strategy_sum.any()
        assert not storage.reach_counts.any()

    def test_num_infosets_is_the_full_tree_not_progress(self, storage, tree):
        """Capacity is static; visits are the progress signal, and they differ."""
        assert storage.num_infosets() == tree.num_rows
        assert storage.num_touched_infosets() == 0


class TestViewAliasing:
    def test_writes_land_in_backing_array(self, storage, tree):
        infoset = storage.infoset_at(0, 0)
        infoset.regrets[0] = 3.5
        start, _ = tree.slots(0, 0)
        assert storage.regrets[start] == pytest.approx(3.5)

    def test_view_width_matches_node_action_count(self, storage, tree):
        for node in tree.nodes[:50]:
            infoset = storage.infoset_at(node.node_id, 0)
            assert len(infoset.regrets) == node.num_actions
            assert len(infoset.strategy_sum) == node.num_actions
            assert infoset.num_actions == node.num_actions

    def test_distinct_infosets_never_share_slots(self, storage, tree):
        """Write a unique value per infoset; every slot must end up unique.

        This is the aliasing check: if two infosets overlapped, the later write
        would overwrite the earlier and the count would come up short.
        """
        expected = 0
        marker = 1
        for node in tree.nodes:
            for bucket in range(tree.num_buckets(node.street)):
                infoset = storage.infoset_at(node.node_id, bucket)
                infoset.regrets[:] = marker
                expected += node.num_actions
                marker += 1

        num_infosets = marker - 1
        assert expected == tree.num_slots
        # Every slot written exactly once => no zeros left, and exactly one
        # distinct marker survives per infoset (an alias would lose one).
        assert np.count_nonzero(storage.regrets) == tree.num_slots
        assert len(np.unique(storage.regrets)) == num_infosets

    def test_neighbouring_infosets_are_independent(self, storage, tree):
        """Writing one infoset must not perturb the ones on either side."""
        node = tree.nodes[len(tree) // 2]
        buckets = tree.num_buckets(node.street)
        if buckets < 3:
            pytest.skip("needs at least three buckets to have a middle")

        storage.infoset_at(node.node_id, 1).regrets[:] = 7.0
        assert not storage.infoset_at(node.node_id, 0).regrets.any()
        assert not storage.infoset_at(node.node_id, 2).regrets.any()

    def test_row_roundtrip(self, storage, tree):
        for node in (tree.nodes[0], tree.nodes[len(tree) // 3], tree.nodes[-1]):
            for bucket in (0, tree.num_buckets(node.street) - 1):
                row = tree.row(node.node_id, bucket)
                recovered = storage.view_at_row(row)
                assert recovered.node_id == node.node_id
                assert recovered.bucket == bucket


class TestStats:
    def test_reach_counts_track_per_infoset(self, storage, tree):
        a = storage.infoset_at(0, 0)
        b = storage.infoset_at(0, 1)
        a.increment_reach_count()
        a.increment_reach_count()
        b.increment_reach_count()

        assert storage.reach_counts[tree.row(0, 0)] == 2
        assert storage.reach_counts[tree.row(0, 1)] == 1
        assert storage.num_touched_infosets() == 2

    def test_every_infoset_is_writable(self, storage, tree):
        """The property that makes dropped updates structurally impossible."""
        for node in tree.nodes[:100]:
            assert storage.infoset_at(node.node_id, 0).writable

    def test_touched_iteration_is_the_visited_subset(self, storage, tree):
        storage.infoset_at(3, 0).increment_reach_count()
        storage.infoset_at(5, 1).increment_reach_count()
        touched = list(storage.iter_touched_infosets())
        assert len(touched) == 2
        assert {(i.node_id, i.bucket) for i in touched} == {(3, 0), (5, 1)}

    def test_full_iteration_covers_every_row(self, storage, tree):
        assert sum(1 for _ in storage.iter_infosets()) == tree.num_rows

    def test_iteration_does_not_mark_coverage(self, storage):
        """A metrics sweep must not make an untrained tree look fully covered."""
        list(storage.iter_infosets())
        assert storage.num_touched_infosets() == 0


class TestSharedMemory:
    def test_worker_attach_sees_writes(self, tree):
        session = "static-storage-test"
        owner = StaticArrayStorage(tree, session_id=session)
        try:
            owner.infoset_at(2, 0).regrets[:] = 9.0
            worker = StaticArrayStorage(tree, session_id=session, attach=True)
            try:
                start, _ = tree.slots(2, 0)
                assert worker.regrets[start] == pytest.approx(9.0)
                # And the reverse direction: the worker writes, owner observes.
                worker.infoset_at(4, 0).regrets[:] = 4.0
                assert owner.regrets[tree.slots(4, 0)[0]] == pytest.approx(4.0)
            finally:
                worker.close()
        finally:
            owner.close()


class TestBucketBounds:
    """An out-of-range bucket must fail loudly, not alias another node's rows.

    Rows are contiguous per node, so `row_offset[n] + oversized_bucket` lands on
    a real row owned by a later node. Without a check, two unrelated infosets
    silently share storage and nothing anywhere reports an error.
    """

    def test_bucket_past_the_end_raises(self, storage, tree):
        node = tree.nodes[0]
        n = tree.num_buckets(node.street)
        with pytest.raises(IndexError, match="out of range"):
            storage.infoset_at(node.node_id, n)

    def test_negative_bucket_raises(self, storage, tree):
        with pytest.raises(IndexError, match="out of range"):
            storage.infoset_at(tree.nodes[0].node_id, -1)

    def test_oversized_bucket_would_have_hit_a_real_row(self, storage, tree):
        """Shows the check is load-bearing: the bad index is otherwise valid."""
        node = tree.nodes[0]
        n = tree.num_buckets(node.street)
        aliased = int(tree.row_offset[node.node_id]) + n
        assert aliased < tree.num_rows, "expected the bad index to be in-array"

    def test_last_valid_bucket_still_works(self, storage, tree):
        node = tree.nodes[0]
        infoset = storage.infoset_at(node.node_id, tree.num_buckets(node.street) - 1)
        assert len(infoset.regrets) == node.num_actions
