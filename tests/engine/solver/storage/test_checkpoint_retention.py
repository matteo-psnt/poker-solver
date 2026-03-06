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


class TestProductionLadderShape:
    """The rung list production actually gets, at the shipped knob values.

    ``checkpoint_frequency: 500000`` with ``checkpoint_retain_every: 1000000``. The
    x-axis of every convergence curve is this list, so it is worth pinning rather
    than deriving: a half-band offset would misplace every point on the plot, and
    the cost of finding out is a whole 8M-iteration training run.
    """

    def test_an_8m_run_lands_rungs_on_round_million_boundaries(self, tmp_path):
        for iteration in range(500_000, 8_000_001, 500_000):
            _commit(tmp_path, iteration, retain_every=1_000_000)

        assert retained_checkpoint_iterations(tmp_path) == [
            500_000,
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            6_000_000,
            7_000_000,
            8_000_000,
        ]

    def test_a_resume_leg_that_forgets_the_knob_keeps_the_earlier_rungs(self, tmp_path):
        """Production 8M runs are stitched from resume legs (the guillotine workaround).

        A later leg spawned without the knob must not silently delete the points an
        earlier leg was told to keep -- that would yield a curve with no early half.
        """
        for iteration in (500_000, 1_000_000, 1_500_000, 2_000_000):
            _commit(tmp_path, iteration, retain_every=1_000_000)
        for iteration in (2_500_000, 3_000_000):
            _commit(tmp_path, iteration, retain_every=0)

        assert retained_checkpoint_iterations(tmp_path) == [
            500_000,
            1_000_000,
            2_000_000,
            3_000_000,
        ]


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

    def test_a_run_without_a_ladder_names_what_it_has(self, tmp_path):
        """The common case: --at against a run trained before retention was enabled.

        The message must list the alternatives, or it reads like a corrupt checkpoint.
        """
        _commit(tmp_path, 1000, retain_every=0)

        with pytest.raises(ValueError, match=r"available: \[1000\]"):
            CheckpointPaths.for_retained(tmp_path, 500)


class TestResolve:
    """One entry point for every reader, so published and rung loads cannot diverge."""

    def test_none_resolves_the_published_snapshot(self, tmp_path):
        _commit(tmp_path, 1000, retain_every=1000)
        _commit(tmp_path, 2000, retain_every=1000)

        assert CheckpointPaths.resolve(tmp_path) == CheckpointPaths.from_dir(tmp_path)
        assert CheckpointPaths.resolve(tmp_path).checkpoint_zarr.name == "checkpoint-2000.zarr"

    def test_an_iteration_resolves_that_rung(self, tmp_path):
        _commit(tmp_path, 1000, retain_every=1000)
        _commit(tmp_path, 2000, retain_every=1000)

        resolved = CheckpointPaths.resolve(tmp_path, 1000)
        assert resolved == CheckpointPaths.for_retained(tmp_path, 1000)
        assert resolved.checkpoint_zarr.name == "checkpoint-1000.zarr"
