"""StaticArrayStorage: the flat-array backend indexed by the betting tree.

The defect class most likely to hide here is an off-by-one in the ragged slot
layout — two infosets aliasing the same slots, or an infoset's view straddling a
neighbour's. Those are silent: training still runs, regrets just leak between
unrelated infosets. Several tests below exist only to make that loud.
"""

from __future__ import annotations

import subprocess
import sys
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.shared import repo
from tests.test_helpers import make_test_config

BUCKETS = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
PROJECT_ROOT = repo.ROOT


def build_tree() -> BettingTree:
    """The tree the whole module shares — a function, not just a fixture.

    ``test_a_killed_creator_still_has_its_segments_reclaimed`` runs out of
    process and its child has to compute the SAME segment names, which are keyed
    by the tree fingerprint. Both ends calling this is what makes that true by
    construction instead of by two definitions agreeing.
    """
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    rules = GameRules(small_blind=config.game.small_blind, big_blind=config.game.big_blind)
    return BettingTree(
        rules,
        ActionModel(config),
        starting_stack=config.game.starting_stack,
        buckets_per_street=BUCKETS,
    )


KILLED_CREATOR_SESSION = "killed-creator-probe"


def attach_and_exit() -> None:
    """Child entry point for the out-of-process test: attach, then go away."""
    StaticArrayStorage(build_tree(), session_id=KILLED_CREATOR_SESSION, attach=True).close()


# The import path is `__name__`, not a literal: this program imports the module
# that generates it, and a literal is only correct until the file moves -- which
# it has. Spelled out, the breakage is a subprocess exiting 1 with an ImportError
# buried in captured stderr, which reads as the shared-memory case failing.
KILLED_CREATOR_PROGRAM = f"""\
import os, sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from multiprocessing import get_context
from src.engine.solver.storage.static_array import StaticArrayStorage
from {__name__} import KILLED_CREATOR_SESSION, attach_and_exit, build_tree

if __name__ == "__main__":
    owner = StaticArrayStorage(build_tree(), session_id=KILLED_CREATOR_SESSION)
    child = get_context("spawn").Process(target=attach_and_exit)
    child.start()
    child.join()
    print("\\n".join(owner._shm_name(a) for a in owner._spec()))
    sys.stdout.flush()
    # No close(), no atexit, no finally: the resource tracker is now the only
    # thing that can reclaim these. That IS the SIGKILL case.
    os._exit(0)
"""


@pytest.fixture(scope="module")
def tree():
    return build_tree()


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

    An oversized bucket does not fall off the array — `base[n] + bucket *
    stride[n]` lands on a real row owned by another node. Without a check, two
    unrelated infosets silently share storage and nothing anywhere reports it.
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
        aliased = int(tree.row_base[node.node_id]) + n * int(tree.row_stride[node.node_id])
        assert aliased < tree.num_rows, "expected the bad index to be in-array"

    def test_last_valid_bucket_still_works(self, storage, tree):
        node = tree.nodes[0]
        infoset = storage.infoset_at(node.node_id, tree.num_buckets(node.street) - 1)
        assert len(infoset.regrets) == node.num_actions


class TestSharedNamesAreTreeKeyed:
    """A worker that built a different tree must not attach at all.

    Workers rebuild the tree locally and then index SHARED arrays with it. Two
    processes disagreeing about the tree would each write to different rows of
    the same memory — silent, total corruption. Tree enumeration is in fact
    deterministic across processes (verified separately), but "deterministic
    today" is not a guarantee, so the segment names are keyed by the fingerprint
    to make a mismatch unattachable rather than merely unlikely.
    """

    def test_mismatched_tree_cannot_attach(self, tree):
        config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
        rules = GameRules(small_blind=1, big_blind=2)
        other = BettingTree(
            rules,
            ActionModel(config),
            starting_stack=20,
            buckets_per_street={Street.FLOP: 7, Street.TURN: 9, Street.RIVER: 11},
        )
        assert other.fingerprint() != tree.fingerprint()

        owner = StaticArrayStorage(tree, session_id="tree-keyed")
        try:
            with pytest.raises(FileNotFoundError, match="DIFFERENT tree"):
                StaticArrayStorage(other, session_id="tree-keyed", attach=True)
        finally:
            owner.close()

    def test_matching_tree_attaches(self, tree):
        owner = StaticArrayStorage(tree, session_id="tree-keyed-ok")
        try:
            worker = StaticArrayStorage(tree, session_id="tree-keyed-ok", attach=True)
            worker.close()
        finally:
            owner.close()

    def test_names_fit_the_platform_limit(self, tree):
        """macOS caps shared-memory names at 31 chars including the leading slash."""
        storage = StaticArrayStorage(tree)
        try:
            for array in ("regrets", "strategy_sum", "cumulative_utility"):
                assert len(storage._shm_name(array)) <= 30
        finally:
            storage.close()


class TestAttachingCannotDestroy:
    """A process that only ATTACHES must never be able to unlink the arrays.

    POSIX SharedMemory registers with the attaching process's resource tracker by
    default, and a tracker unlinks what it holds when its process dies. Measured:
    a spawned child shares the coordinator's tracker and is harmless, but a
    SEPARATE interpreter that attaches and then dies — a second task on the node, a
    probe, a worker re-parented onto a fresh tracker — destroys the segments the
    coordinator is still training on. A 50M task died at 38,000,000 exactly this
    way, mid-run, with no error until the next chunk failed to attach.

    The tracker holds ONE entry per name, not one per process, so "attach and
    then unregister" is not a private correction — it deletes the creator's
    entry. Both halves below therefore have to hold at once.
    """

    def test_attaching_never_touches_the_resource_tracker(self, tree, monkeypatch):
        """Not registered, and so not unregistered either.

        White-box on purpose. The BEHAVIOUR — a dying tracker-holding attacher
        unlinks the segment — is a property of CPython, reproduced separately;
        what this repo has to keep true is that attaching claims no ownership.
        Asserting on the tracker calls is deterministic, where racing a real
        tracker is not.

        Both verbs are spied, because the failure this replaced made exactly the
        pair: register on attach, unregister immediately after.
        """
        owner = StaticArrayStorage(tree, session_id="attach-untracked")
        calls: list[tuple[str, str, str]] = []
        for verb in ("register", "unregister"):
            real = getattr(resource_tracker, verb)

            def spy(name, rtype, _verb=verb, _real=real):
                calls.append((_verb, name, rtype))
                return _real(name, rtype)

            monkeypatch.setattr(resource_tracker, verb, spy)
        try:
            reader = StaticArrayStorage(tree, session_id="attach-untracked", attach=True)
            reader.close()
            # SNAPSHOT HERE. The owner's close() unlinks, and unlink() legitimately
            # unregisters — letting it into the list would drown the thing measured.
            snapshot = list(calls)
        finally:
            owner.close()

        assert snapshot == [], (
            f"attaching touched the resource tracker ({snapshot}); registering claims "
            "ownership the attacher must not have, and unregistering deletes the "
            "CREATOR's entry, since the tracker holds one entry per name"
        )

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_a_killed_creator_still_has_its_segments_reclaimed(self, tmp_path):
        """The half a spy cannot see: the tracker's own cache, in another process.

        A creator that never reaches close() — SIGKILL, an OOM, the wall-clock
        guard — is reclaimed by its resource tracker and by nothing else. When an
        attaching worker unregistered, that entry was gone and the segments
        survived the run that made them; the evidence arrived one task later, as
        `Reclaiming shared segment left behind by an earlier task`.

        Run out of process because the state under test IS a separate process's
        cache. `os._exit` skips every interpreter shutdown hook, so the tracker
        losing the pipe is the only thing left to do the cleanup — exactly the
        SIGKILL case. Capturing stderr makes this deterministic rather than racy:
        the tracker inherits that pipe, so the read ends only when IT exits.
        """
        path = tmp_path / "killed_creator.py"
        path.write_text(KILLED_CREATOR_PROGRAM)
        done = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, timeout=50, check=True
        )
        names = done.stdout.split()

        try:
            assert "KeyError" not in done.stderr, (
                "the resource tracker raised on a name it no longer held — an attacher "
                f"unregistered the creator's entry:\n{done.stderr}"
            )
            assert names, f"the probe printed no segment names:\n{done.stderr}"
            for name in names:
                with pytest.raises(FileNotFoundError):
                    shared_memory.SharedMemory(name=name, create=False, track=False).close()
        finally:
            # On failure the segments are exactly what the test says they are —
            # still there — and leaving them would break the NEXT run of this test.
            for name in names:
                try:
                    stale = shared_memory.SharedMemory(name=name, create=False, track=False)
                    stale.close()
                    stale.unlink()
                except FileNotFoundError:
                    pass


class TestAbandonedSegmentsAreReclaimed:
    """A task killed before close() must not lock the next task out of its own run.

    POSIX shared memory outlives its creator, and the session id IS the run id,
    so a coordinator lost to SIGKILL/OOM/the wall-clock guard leaves segments
    that the retry then collides with. That killed a real task on the pool: the
    next task died on FileExistsError before doing any work, which is precisely
    the retry the absolute-iteration design promises is safe.
    """

    def test_create_reclaims_a_segment_left_by_a_dead_coordinator(self, tree):
        abandoned = StaticArrayStorage(tree, session_id="orphaned-task")
        abandoned.regrets[0] = 123.0
        # Drop the mappings WITHOUT unlinking, which is what a killed process
        # leaves behind. close() would unlink and defeat the point.
        for shm in abandoned._shm:
            shm.close()
        abandoned._shm = []

        successor = StaticArrayStorage(tree, session_id="orphaned-task")
        try:
            assert successor.regrets[0] == 0.0, "reclaimed segment must start zeroed"
        finally:
            successor.close()

    def test_reclaim_leaves_other_sessions_alone(self, tree):
        neighbour = StaticArrayStorage(tree, session_id="other-run")
        neighbour.regrets[0] = 7.0
        orphan = StaticArrayStorage(tree, session_id="reclaimed-run")
        for shm in orphan._shm:
            shm.close()
        orphan._shm = []

        successor = StaticArrayStorage(tree, session_id="reclaimed-run")
        try:
            assert neighbour.regrets[0] == 7.0
        finally:
            successor.close()
            neighbour.close()
