"""Checkpoint retention: keep a measurable ladder, not just the last snapshot.

Every checkpoint commit prunes the snapshots the new manifest supersedes, so by
default a run ends holding exactly one checkpoint. That makes any within-run
measurement -- exploitability against training iterations -- impossible to compute
after the fact, since there is nothing left to score but the endpoint.
``checkpoint_retain_every`` spares one committed snapshot per band so the ladder
survives the run.
"""

import json

import pytest

from src.engine.solver.storage.helpers import (
    CHECKPOINT_MANIFEST_FILE,
    CheckpointPaths,
    commit_checkpoint_manifest,
    retained_checkpoint_iterations,
)


def _commit(checkpoint_dir, iteration: int, *, retain_every: int) -> CheckpointPaths:
    """Write a snapshot's artifacts as the writer would, then publish it."""
    paths = CheckpointPaths.for_iteration(checkpoint_dir, iteration)
    paths.checkpoint_zarr.mkdir(parents=True)
    paths.key_table.mkdir(parents=True)
    commit_checkpoint_manifest(checkpoint_dir, iteration, paths, retain_every=retain_every)
    return paths


def _snapshots_on_disk(checkpoint_dir) -> list[str]:
    return sorted(p.name for p in checkpoint_dir.glob("checkpoint-*.zarr"))


class TestRetentionDisabled:
    def test_only_the_current_snapshot_survives(self, tmp_path):
        for iteration in (100, 200, 300):
            _commit(tmp_path, iteration, retain_every=0)
        assert _snapshots_on_disk(tmp_path) == ["checkpoint-300.zarr"]
        assert retained_checkpoint_iterations(tmp_path) == [300]


class TestRetentionLadder:
    def test_one_snapshot_per_band_survives(self, tmp_path):
        """Bands of 1000 over checkpoints every 400: keepers at 400, 1200, 2000."""
        for iteration in (400, 800, 1200, 1600, 2000, 2400):
            _commit(tmp_path, iteration, retain_every=1000)

        assert retained_checkpoint_iterations(tmp_path) == [400, 1200, 2000, 2400]
        assert _snapshots_on_disk(tmp_path) == [
            "checkpoint-1200.zarr",
            "checkpoint-2000.zarr",
            "checkpoint-2400.zarr",
            "checkpoint-400.zarr",
        ]

    def test_band_keeper_is_the_first_commit_into_it(self, tmp_path):
        """Keepers must not drift as training continues, or the ladder is unstable."""
        _commit(tmp_path, 1000, retain_every=1000)
        _commit(tmp_path, 1500, retain_every=1000)
        _commit(tmp_path, 1900, retain_every=1000)
        assert retained_checkpoint_iterations(tmp_path) == [1000, 1900]

    def test_key_tables_are_retained_alongside_their_arrays(self, tmp_path):
        for iteration in (400, 1200, 1600):
            _commit(tmp_path, iteration, retain_every=1000)
        assert sorted(p.name for p in tmp_path.glob("keys-*")) == [
            "keys-1200",
            "keys-1600",
            "keys-400",
        ]

    def test_uncommitted_snapshots_are_never_retained(self, tmp_path):
        """A crash mid-write leaves artifacts that were never published.

        They are indistinguishable from good ones by filename, so retention keyed
        on the disk listing would pin a half-written snapshot into the ladder and
        later score it as a measurement point.
        """
        orphan = CheckpointPaths.for_iteration(tmp_path, 500)
        orphan.checkpoint_zarr.mkdir(parents=True)
        _commit(tmp_path, 900, retain_every=1000)

        assert retained_checkpoint_iterations(tmp_path) == [900]
        assert _snapshots_on_disk(tmp_path) == ["checkpoint-900.zarr"]


class TestLadderSurvivesResume:
    def test_a_leg_without_the_knob_does_not_delete_earlier_keepers(self, tmp_path):
        """A resume whose caller forgot to forward the knob must not wipe the ladder."""
        _commit(tmp_path, 1000, retain_every=1000)
        _commit(tmp_path, 2000, retain_every=1000)
        _commit(tmp_path, 3000, retain_every=0)

        assert retained_checkpoint_iterations(tmp_path) == [1000, 2000, 3000]

    def test_a_resume_into_an_occupied_band_contributes_no_point(self, tmp_path):
        """Bands are absolute, so a resume mid-band adds nothing until it crosses.

        Leg 1 ends at 4200 having already claimed band 4 at 4000; leg 2's first
        checkpoint at 4600 is in that same band and is not retained. The ladder
        therefore shows a gap exactly where a resume happened -- by design, but it
        is the reading that surprises, so it is pinned here.
        """
        for iteration in (4000, 4200):
            _commit(tmp_path, iteration, retain_every=1000)
        for iteration in (4600, 5100):
            _commit(tmp_path, iteration, retain_every=1000)

        assert retained_checkpoint_iterations(tmp_path) == [4000, 5100]

    def test_pre_retention_manifests_are_readable(self, tmp_path):
        """Runs already on the volume have manifests with no ``retained`` field."""
        (tmp_path / CHECKPOINT_MANIFEST_FILE).write_text(
            json.dumps(
                {
                    "iteration": 8_000_000,
                    "zarr": "checkpoint-8000000.zarr",
                    "key_table": "keys-8000000",
                }
            )
        )
        assert retained_checkpoint_iterations(tmp_path) == [8_000_000]


class TestRetainedLookup:
    def test_resolves_a_ladder_snapshot_by_iteration(self, tmp_path):
        _commit(tmp_path, 1000, retain_every=1000)
        _commit(tmp_path, 2000, retain_every=1000)

        paths = CheckpointPaths.for_retained(tmp_path, 1000)
        assert paths.checkpoint_zarr.name == "checkpoint-1000.zarr"
        assert paths.key_table.name == "keys-1000"

    def test_current_snapshot_is_addressable_too(self, tmp_path):
        _commit(tmp_path, 1000, retain_every=1000)
        _commit(tmp_path, 1500, retain_every=1000)

        assert CheckpointPaths.for_retained(tmp_path, 1500).checkpoint_zarr.exists()

    def test_rejects_an_iteration_that_was_pruned(self, tmp_path):
        """1500 shares band 1 with keeper 1000, so committing 1800 prunes it."""
        for iteration in (1000, 1500, 1800):
            _commit(tmp_path, iteration, retain_every=1000)

        with pytest.raises(ValueError, match="No retained checkpoint at iteration 1500"):
            CheckpointPaths.for_retained(tmp_path, 1500)
