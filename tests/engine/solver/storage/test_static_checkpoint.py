"""Static checkpoints: round-trip fidelity, and refusing the wrong tree.

A static checkpoint is a bare array of numbers. Nothing in it says which infoset
each row belongs to — the tree does. So the failure that matters is not a
corrupt file (loaders catch that); it is loading a *valid* checkpoint against a
*different* tree, which reinterprets every row as some other infoset and lets
training continue on scrambled regrets with no error anywhere. The fingerprint
tests below are the only thing standing between that and a silently wrong run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.storage import snapshot_format
from src.engine.solver.storage.static_array import _ARRAYS, StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import (
    AbstractionMismatchError,
    FingerprintMismatchError,
    StaticCheckpointManifest,
    _legacy_index_maps,
    load_checkpoint,
    read_strategy_sum,
    retained_iterations,
    save_checkpoint,
)
from tests.test_helpers import make_test_config

BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}


def _tree(stack: int = 20, buckets: dict | None = None) -> BettingTree:
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=stack)
    rules = GameRules(small_blind=1, big_blind=2)
    return BettingTree(
        rules,
        ActionModel(config),
        starting_stack=stack,
        buckets_per_street=buckets or BUCKETS,
    )


@pytest.fixture(scope="module")
def tree():
    return _tree()


def _populate(storage: StaticArrayStorage, seed: int = 0) -> None:
    """Fill every array with distinctive values so a shuffle would show up."""
    rng = np.random.default_rng(seed)
    storage.regrets[:] = rng.normal(size=storage.regrets.shape).astype(storage.regrets.dtype)
    storage.strategy_sum[:] = rng.random(storage.strategy_sum.shape).astype(
        storage.strategy_sum.dtype
    )
    storage.reach_counts[:] = rng.integers(0, 1000, size=storage.reach_counts.shape)
    storage.cumulative_utility[:] = rng.normal(size=storage.cumulative_utility.shape)
    storage.visited[:] = rng.integers(0, 2, size=storage.visited.shape)


class TestRoundTrip:
    def test_every_array_survives_exactly(self, tree, tmp_path):
        source = StaticArrayStorage(tree)
        try:
            _populate(source)
            expected = {name: np.array(getattr(source, name)) for name in _ARRAYS}
            save_checkpoint(source, tmp_path, 1000)
        finally:
            source.close()

        target = StaticArrayStorage(tree)
        try:
            assert load_checkpoint(target, tmp_path) == 1000
            for name in _ARRAYS:
                np.testing.assert_array_equal(getattr(target, name), expected[name], err_msg=name)
        finally:
            target.close()

    def test_load_reports_the_iteration(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 7_500_000)
            assert load_checkpoint(storage, tmp_path) == 7_500_000
        finally:
            storage.close()

    def test_missing_manifest_raises(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            with pytest.raises(FileNotFoundError):
                load_checkpoint(storage, tmp_path)
        finally:
            storage.close()


class TestFingerprintGuard:
    """The load-bearing tests: a mismatched tree must be refused, not reinterpreted."""

    def test_different_bucket_counts_refused(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            _populate(storage)
            save_checkpoint(storage, tmp_path, 100)
        finally:
            storage.close()

        other = _tree(buckets={Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 5})
        target = StaticArrayStorage(other)
        try:
            with pytest.raises(FingerprintMismatchError, match="reinterpret every row"):
                load_checkpoint(target, tmp_path)
        finally:
            target.close()

    def test_different_stack_depth_refused(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 100)
        finally:
            storage.close()

        target = StaticArrayStorage(_tree(stack=40))
        try:
            with pytest.raises(FingerprintMismatchError):
                load_checkpoint(target, tmp_path)
        finally:
            target.close()

    def test_manifest_and_arrays_must_agree(self, tree, tmp_path):
        """A doctored manifest must not smuggle in a mismatched snapshot."""
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 100)
        finally:
            storage.close()

        manifest_path = tmp_path / "STATIC_CHECKPOINT.json"
        import json

        raw = json.loads(manifest_path.read_text())
        raw["fingerprint"] = _tree(stack=40).fingerprint()
        manifest_path.write_text(json.dumps(raw))

        target = StaticArrayStorage(_tree(stack=40))
        try:
            with pytest.raises(FingerprintMismatchError, match=r"corrupt|reinterpret"):
                load_checkpoint(target, tmp_path)
        finally:
            target.close()


class TestRetentionLadder:
    """Retention is what makes a within-run convergence curve possible at all."""

    def test_ladder_keeps_one_per_band(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            for iteration in (1000, 1500, 2000, 2500, 3000):
                save_checkpoint(storage, tmp_path, iteration, retain_every=1000)
        finally:
            storage.close()

        # Bands 1,2,3 take their FIRST entrant; 1500 and 2500 are superseded.
        assert retained_iterations(tmp_path) == [1000, 2000, 3000]

    def test_retained_rungs_are_loadable(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            storage.regrets[:] = 1.0
            save_checkpoint(storage, tmp_path, 1000, retain_every=1000)
            storage.regrets[:] = 2.0
            save_checkpoint(storage, tmp_path, 2000, retain_every=1000)
        finally:
            storage.close()

        target = StaticArrayStorage(tree)
        try:
            assert load_checkpoint(target, tmp_path, at_iteration=1000) == 1000
            assert target.regrets[0] == pytest.approx(1.0)
            assert load_checkpoint(target, tmp_path, at_iteration=2000) == 2000
            assert target.regrets[0] == pytest.approx(2.0)
        finally:
            target.close()

    def test_pruning_removes_superseded_snapshots(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            for iteration in (1000, 1500, 2000):
                save_checkpoint(storage, tmp_path, iteration, retain_every=1000)
        finally:
            storage.close()
        on_disk = {p.name for p in tmp_path.glob("static-*")}
        assert on_disk == {
            f"static-1000{snapshot_format.SUFFIX}",
            f"static-2000{snapshot_format.SUFFIX}",
        }

    def test_ladder_survives_a_task_that_forgets_retain_every(self, tree, tmp_path):
        """A resume that drops the knob must not delete earlier measurement points."""
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 1000, retain_every=1000)
            save_checkpoint(storage, tmp_path, 2000, retain_every=0)
        finally:
            storage.close()
        assert 1000 in retained_iterations(tmp_path)

    def test_unknown_rung_lists_what_is_available(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 1000, retain_every=1000)
        finally:
            storage.close()
        target = StaticArrayStorage(tree)
        try:
            with pytest.raises(KeyError, match="have"):
                load_checkpoint(target, tmp_path, at_iteration=999)
        finally:
            target.close()


class TestManifestAtomicity:
    def test_manifest_records_current_and_ladder(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 1000, retain_every=1000)
            save_checkpoint(storage, tmp_path, 2000, retain_every=1000)
        finally:
            storage.close()

        manifest = StaticCheckpointManifest.read(tmp_path)
        assert manifest is not None
        assert manifest.iteration == 2000
        assert manifest.fingerprint == tree.fingerprint()
        assert [int(e["iteration"]) for e in manifest.retained] == [1000, 2000]

    def test_no_temp_file_left_behind(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 1000)
        finally:
            storage.close()
        assert not list(tmp_path.glob("*.tmp"))


class TestSolverIntegration:
    """The solver-level path: train, ladder, restore, resume."""

    def _solver(self, tmp_path, retain_every=0):
        from src.core.actions.action_model import ActionModel
        from src.engine.solver.mccfr.static_solver import StaticTreeSolver

        config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)

        class Buckets:
            def get_bucket(
                self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
            ) -> int:
                return (hole_cards[0].rank_eval7() + board[0].rank_eval7()) % BUCKETS[street]

            def num_buckets(self, street: Street) -> int:
                return BUCKETS[street]

        built = _tree()
        return StaticTreeSolver(
            ActionModel(config),
            Buckets(),
            StaticArrayStorage(built),
            config,
            tree=built,
            checkpoint_dir=tmp_path if tmp_path is not None else None,
            checkpoint_retain_every=retain_every,
        )

    def test_train_checkpoint_restore_round_trip(self, tmp_path):
        import random

        solver = self._solver(tmp_path)
        try:
            random.seed(1)
            for _ in range(200):
                solver.train_iteration()
            solver.checkpoint()
            trained = np.array(solver.storage.regrets)
            touched = solver.num_infosets()
            assert touched > 0
            assert trained.any()
        finally:
            solver.storage.close()

        fresh = self._solver(tmp_path)
        try:
            assert fresh.restore() == 200
            np.testing.assert_array_equal(fresh.storage.regrets, trained)
            assert fresh.num_infosets() == touched
        finally:
            fresh.storage.close()

    def test_ladder_accumulates_across_training(self, tmp_path):
        import random

        solver = self._solver(tmp_path, retain_every=100)
        try:
            random.seed(1)
            for _ in range(300):
                solver.train_iteration()
                if solver.iteration % 100 == 0:
                    solver.checkpoint()
        finally:
            solver.storage.close()
        assert retained_iterations(tmp_path) == [100, 200, 300]

    def test_checkpoint_without_a_dir_is_refused(self):
        solver = self._solver(None)
        try:
            with pytest.raises(ValueError, match="no checkpoint_dir"):
                solver.checkpoint()
        finally:
            solver.storage.close()


class TestAbstractionIdentity:
    """The tree fingerprint pins layout; it cannot pin bucket ASSIGNMENT.

    Two abstractions with identical per-street counts produce an identical
    fingerprint while mapping hands to different buckets. Nothing about the
    arrays reveals that, so resuming or scoring across such a change would
    silently train on rebucketed hands — the exact hazard the justfile warns
    about for recomputed abstractions.
    """

    def test_load_under_a_different_abstraction_is_refused(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 100, abstraction_id="abstraction-A")
        finally:
            storage.close()

        target = StaticArrayStorage(tree)
        try:
            with pytest.raises(AbstractionMismatchError, match="different hand"):
                load_checkpoint(target, tmp_path, abstraction_id="abstraction-B")
        finally:
            target.close()

    def test_matching_abstraction_loads(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 100, abstraction_id="abstraction-A")
            assert load_checkpoint(storage, tmp_path, abstraction_id="abstraction-A") == 100
        finally:
            storage.close()

    def test_appending_a_differently_bucketed_rung_is_refused(self, tree, tmp_path):
        """A ladder whose rungs are bucketed differently is not comparable."""
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 100, retain_every=100, abstraction_id="A")
            with pytest.raises(AbstractionMismatchError, match="not comparable"):
                save_checkpoint(storage, tmp_path, 200, retain_every=100, abstraction_id="B")
        finally:
            storage.close()

    def test_absent_ids_stay_permissive(self, tree, tmp_path):
        """Checkpoints written before this existed must remain loadable."""
        storage = StaticArrayStorage(tree)
        try:
            save_checkpoint(storage, tmp_path, 100)
            assert load_checkpoint(storage, tmp_path, abstraction_id="anything") == 100
        finally:
            storage.close()


class TestLegacyLayoutTranslation:
    """A v1 node-major checkpoint of the SAME tree loads by permutation.

    The bucket-major layout moved every row's address without changing its
    values, so a legacy snapshot is a permutation of a current one — and the
    share holds nothing but legacy snapshots the day the layout lands.
    """

    def test_a_node_major_checkpoint_round_trips(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        rng = np.random.default_rng(7)
        storage.regrets[:] = rng.standard_normal(tree.num_slots).astype(np.float32)
        storage.strategy_sum[:] = rng.standard_normal(tree.num_slots).astype(np.float32)
        storage.reach_counts[:] = rng.integers(0, 9, tree.num_rows)
        storage.cumulative_utility[:] = rng.standard_normal(tree.num_rows)
        storage.visited[:] = rng.integers(0, 2, tree.num_rows).astype(np.uint8)
        expected = {name: getattr(storage, name).copy() for name in _ARRAYS}
        save_checkpoint(storage, tmp_path, 5)
        storage.close()

        # Write the snapshot as v1 wrote it -- a ZARR DIRECTORY, every value
        # scattered back to its node-major address, both fingerprints stamped
        # with the legacy id. Built directly rather than by mutating what
        # `save_checkpoint` produced: it no longer produces zarr, and the point
        # of this test is reading the artifact the share is actually full of.
        row_source, slot_source = _legacy_index_maps(tree)
        _write_legacy_zarr(
            tmp_path / "static-5.zarr",
            {
                name: _scatter(expected[name], slot_source if name in _SLOTTED else row_source)
                for name in _ARRAYS
            },
            {"iteration": 5, "fingerprint": tree.legacy_fingerprint()},
        )
        manifest_path = tmp_path / "STATIC_CHECKPOINT.json"
        raw = json.loads(manifest_path.read_text())
        raw["fingerprint"] = tree.legacy_fingerprint()
        raw["zarr"] = "static-5.zarr"
        raw["retained"] = [{"iteration": 5, "zarr": "static-5.zarr"}]
        manifest_path.write_text(json.dumps(raw))

        fresh = StaticArrayStorage(tree)
        try:
            assert load_checkpoint(fresh, tmp_path) == 5
            for name in _ARRAYS:
                assert np.array_equal(getattr(fresh, name), expected[name]), name
        finally:
            fresh.close()

    def test_the_maps_are_bijections(self, tree):
        row_source, slot_source = _legacy_index_maps(tree)
        assert len(np.unique(row_source)) == tree.num_rows
        assert len(np.unique(slot_source)) == tree.num_slots


class TestTheAbstractionGuardIsArmed:
    """`static_parallel` omitted `abstraction_id`, so EVERY run it wrote left the
    manifest field null -- and `load_checkpoint`'s AbstractionMismatchError needs
    the id on BOTH sides to fire. The guard its own docstring calls "the only way
    that failure is ever visible" had never been able to fire."""

    def test_a_fresh_manifest_records_the_bucket_assignment(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        save_checkpoint(storage, tmp_path, 10, abstraction_id="abs-1")
        manifest = StaticCheckpointManifest.read(tmp_path)
        assert manifest is not None
        assert manifest.abstraction_id == "abs-1"

    def test_a_genuine_mismatch_is_refused(self, tree, tmp_path):
        storage = StaticArrayStorage(tree)
        save_checkpoint(storage, tmp_path, 10, abstraction_id="abs-1")
        fresh = StaticArrayStorage(tree)
        with pytest.raises(AbstractionMismatchError):
            load_checkpoint(fresh, tmp_path, abstraction_id="abs-2")

    def test_a_historical_manifest_without_the_field_still_loads(self, tree, tmp_path):
        """The case that keeps other sessions' mid-training runs safe: every
        checkpoint on the share today predates this field. Arming the guard must
        be additive, never a gate."""
        storage = StaticArrayStorage(tree)
        storage.strategy_sum[:] = 1.0
        save_checkpoint(storage, tmp_path, 10)
        raw = json.loads((tmp_path / "STATIC_CHECKPOINT.json").read_text())
        assert raw["abstraction_id"] is None

        fresh = StaticArrayStorage(tree)
        assert load_checkpoint(fresh, tmp_path, abstraction_id="abs-anything") == 10
        assert float(fresh.strategy_sum[0]) == 1.0


#: The two arrays addressed per SLOT rather than per row.
_SLOTTED = ("regrets", "strategy_sum")


class TestALadderStraddlingTheLayoutChange:
    """MEASURED 09-09. A run that was training when the bucket-major layout
    landed holds v1 rungs and v2 rungs under ONE manifest, whose fingerprint can
    only describe the rung that was current when it was last written.

    Deciding the vintage from that manifest refused every older rung of five
    published ladders -- 18 rungs, and always the EARLY ones, so what it cost
    was the left half of a within-run convergence curve for the four 100M
    abstraction arms. The rung says what layout it is; the manifest does not
    speak for it.
    """

    def _straddling(self, tree, tmp_path):
        """Rung 5 node-major, rung 10 bucket-major, manifest naming rung 10."""
        storage = StaticArrayStorage(tree)
        rng = np.random.default_rng(11)
        storage.regrets[:] = rng.standard_normal(tree.num_slots).astype(np.float32)
        storage.strategy_sum[:] = rng.standard_normal(tree.num_slots).astype(np.float32)
        storage.reach_counts[:] = rng.integers(0, 9, tree.num_rows)
        storage.cumulative_utility[:] = rng.standard_normal(tree.num_rows)
        storage.visited[:] = rng.integers(0, 2, tree.num_rows).astype(np.uint8)
        early = {name: getattr(storage, name).copy() for name in _ARRAYS}

        # Rung 10 is written the way the trainer writes today, and its manifest
        # carries the CURRENT fingerprint.
        storage.regrets[:] = rng.standard_normal(tree.num_slots).astype(np.float32)
        late = {name: getattr(storage, name).copy() for name in _ARRAYS}
        save_checkpoint(storage, tmp_path, 10)
        storage.close()

        row_source, slot_source = _legacy_index_maps(tree)
        _write_legacy_zarr(
            tmp_path / "static-5.zarr",
            {
                name: _scatter(early[name], slot_source if name in _SLOTTED else row_source)
                for name in _ARRAYS
            },
            {"iteration": 5, "fingerprint": tree.legacy_fingerprint()},
        )
        manifest_path = tmp_path / "STATIC_CHECKPOINT.json"
        raw = json.loads(manifest_path.read_text())
        raw["retained"] = [
            {"iteration": 5, "zarr": "static-5.zarr"},
            {"iteration": 10, "zarr": raw["zarr"]},
        ]
        manifest_path.write_text(json.dumps(raw))
        return early, late

    def test_the_older_node_major_rung_still_loads(self, tree, tmp_path):
        early, _late = self._straddling(tree, tmp_path)
        fresh = StaticArrayStorage(tree)
        try:
            assert load_checkpoint(fresh, tmp_path, at_iteration=5) == 5
            for name in _ARRAYS:
                assert np.array_equal(getattr(fresh, name), early[name]), name
        finally:
            fresh.close()

    def test_the_current_bucket_major_rung_still_loads(self, tree, tmp_path):
        """The other half: fixing the old rung must not permute the new one."""
        _early, late = self._straddling(tree, tmp_path)
        fresh = StaticArrayStorage(tree)
        try:
            assert load_checkpoint(fresh, tmp_path, at_iteration=10) == 10
            for name in _ARRAYS:
                assert np.array_equal(getattr(fresh, name), late[name]), name
        finally:
            fresh.close()

    def test_read_strategy_sum_reads_the_old_rung_in_this_order(self, tree, tmp_path):
        """A windowed average combines rungs, so this reader must translate per
        rung too -- combining a v1 and a v2 rung in one ORDER would silently
        average two different addressings."""
        early, _late = self._straddling(tree, tmp_path)
        fresh = StaticArrayStorage(tree)
        try:
            values = read_strategy_sum(fresh, tmp_path, 5)
            assert np.array_equal(values, early["strategy_sum"])
        finally:
            fresh.close()

    def test_a_rung_of_a_third_tree_is_still_refused(self, tree, tmp_path):
        """The permission is exactly two layouts of THIS tree, not any snapshot
        that happens to be beside a manifest that passed."""
        self._straddling(tree, tmp_path)
        _write_legacy_zarr(
            tmp_path / "static-5.zarr",
            {name: np.zeros_like(getattr(StaticArrayStorage(tree), name)) for name in _ARRAYS},
            {"iteration": 5, "fingerprint": "deadbeefdeadbeef"},
        )
        fresh = StaticArrayStorage(tree)
        try:
            with pytest.raises(FingerprintMismatchError, match="neither this tree"):
                load_checkpoint(fresh, tmp_path, at_iteration=5)
        finally:
            fresh.close()


def _scatter(values, gather):
    """`values` put back at their node-major addresses."""
    legacy = np.empty_like(values)
    legacy[gather] = values
    return legacy


def _write_legacy_zarr(path, arrays, attrs):
    """A rung in the pre-migration ARRAY ORDER, in the format that replaced it.

    The zarr directory is gone from every store, but node-major order is not:
    the migration re-encoded those rungs byte-for-byte, so the container is full
    of `.ckpt.zst` objects whose arrays are still scattered the old way. The
    translation this exercises is what makes them loadable.

    Named for the manifest's spelling, written under the object's: a legacy
    manifest still says `static-5.zarr` and `records.object_name` is what turns
    that into the file beside it.
    """
    from src.shared import records

    snapshot_format.write_snapshot(
        path.with_name(records.object_name(path.name)),
        {name: np.asarray(array) for name, array in arrays.items()},
        dict(attrs),
    )
