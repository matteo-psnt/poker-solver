"""Static checkpoints: round-trip fidelity, and refusing the wrong tree.

A static checkpoint is a bare array of numbers. Nothing in it says which infoset
each row belongs to — the tree does. So the failure that matters is not a
corrupt file (loaders catch that); it is loading a *valid* checkpoint against a
*different* tree, which reinterprets every row as some other infoset and lets
training continue on scrambled regrets with no error anywhere. The fingerprint
tests below are the only thing standing between that and a silently wrong run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.storage.static_array import _ARRAYS, StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import (
    AbstractionMismatchError,
    FingerprintMismatchError,
    StaticCheckpointManifest,
    load_checkpoint,
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
        on_disk = {p.name for p in tmp_path.glob("static-*.zarr")}
        assert on_disk == {"static-1000.zarr", "static-2000.zarr"}

    def test_ladder_survives_a_leg_that_forgets_retain_every(self, tree, tmp_path):
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
            assert touched > 0 and trained.any()
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
