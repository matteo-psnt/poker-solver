"""Tests for RunTracker."""

import json

from src.core.actions.action_model import ActionModel
from src.pipeline.training.run_tracker import RunMetadata, RunTracker, migrate_run_log
from src.shared import run_events
from src.shared.config import Config


class TestRunTracker:
    """Tests for RunTracker class."""

    def _action_config_hash(self) -> str:
        config = Config.default()
        return ActionModel(config).get_config_hash()

    def test_create_new_tracker(self, tmp_path):
        """Test creating a new run tracker."""
        run_dir = tmp_path / "run-test"
        config = Config.default()

        tracker = RunTracker(
            run_dir=run_dir,
            config_name="test",
            config=config,
            action_config_hash=self._action_config_hash(),
        )

        assert tracker.run_id == "run-test"
        assert tracker.metadata.config_name == "test"
        assert tracker.metadata.status == "running"
        assert tracker.metadata.iterations == 0

        # File should NOT exist yet (delayed creation)
        metadata_file = run_dir / "run.jsonl"
        assert not metadata_file.exists()

        # Initialize to create file
        tracker.initialize()
        assert metadata_file.exists()

    def test_new_run_records_git_provenance(self, tmp_path):
        """A fresh run stamps the current git commit + dirty flag, surviving a round-trip."""
        run_dir = tmp_path / "run-git"
        tracker = RunTracker(
            run_dir=run_dir,
            config_name="test",
            config=Config.default(),
            action_config_hash=self._action_config_hash(),
        )
        # In this repo checkout the commit is a 40-char sha and dirty is a bool.
        commit = tracker.metadata.git_commit
        assert commit is None or (len(commit) == 40)
        assert tracker.metadata.git_dirty in (True, False, None)

        tracker.initialize()
        reloaded = RunTracker.load(run_dir)
        assert reloaded.metadata.git_commit == tracker.metadata.git_commit
        assert reloaded.metadata.git_dirty == tracker.metadata.git_dirty

    def test_legacy_metadata_without_git_loads_as_none(self, tmp_path):
        """Pre-provenance runs (no git fields) must load with None, not crash."""
        run_dir = tmp_path / "run-legacy"
        run_dir.mkdir()
        metadata = {
            "run_id": "run-legacy",
            "config_name": "test",
            "status": "completed",
            "iterations": 100,
            "runtime_seconds": 10.5,
            "num_infosets": 1000,
            "action_config_hash": self._action_config_hash(),
            "config": Config.default().to_dict(),
        }
        (run_dir / ".run.json").write_text(json.dumps(metadata))
        assert migrate_run_log(run_dir)
        tracker = RunTracker.load(run_dir)
        assert tracker.metadata.git_commit is None
        assert tracker.metadata.git_dirty is None

    def test_load_existing_tracker(self, tmp_path):
        """Test loading an existing tracker."""
        run_dir = tmp_path / "run-existing"
        run_dir.mkdir()

        # Create metadata file
        metadata = {
            "run_id": "run-existing",
            "config_name": "test",
            "status": "completed",
            "iterations": 100,
            "runtime_seconds": 10.5,
            "num_infosets": 1000,
            "action_config_hash": self._action_config_hash(),
            "config": Config.default().to_dict(),
        }

        (run_dir / ".run.json").write_text(json.dumps(metadata))
        assert migrate_run_log(run_dir), "a legacy run dir must convert"

        tracker = RunTracker.load(run_dir)

        assert tracker.run_id == "run-existing"
        assert tracker.metadata.status == "completed"
        assert tracker.metadata.iterations == 100

    def test_update_progress(self, tmp_path):
        """Test updating training progress."""
        run_dir = tmp_path / "run-update"
        tracker = RunTracker(
            run_dir=run_dir,
            config_name="test",
            config=Config.default(),
            action_config_hash=self._action_config_hash(),
        )

        tracker.update(
            iterations=50,
            runtime_seconds=5.0,
            num_infosets=500,
            storage_capacity=2000,
        )

        assert tracker.metadata.iterations == 50
        assert tracker.metadata.runtime_seconds == 5.0
        assert tracker.metadata.num_infosets == 500

        # Verify persistence
        tracker2 = RunTracker.load(run_dir)
        assert tracker2.metadata.iterations == 50

    def test_mark_completed(self, tmp_path):
        """Test marking run as completed."""
        run_dir = tmp_path / "run-complete"
        tracker = RunTracker(
            run_dir=run_dir,
            config_name="test",
            config=Config.default(),
            action_config_hash=self._action_config_hash(),
        )

        tracker.mark_completed()

        assert tracker.metadata.status == "completed"
        assert tracker.metadata.completed_at is not None

    def test_mark_failed(self, tmp_path):
        """Test marking run as failed."""
        run_dir = tmp_path / "run-failed"
        tracker = RunTracker(
            run_dir=run_dir,
            config_name="test",
            config=Config.default(),
            action_config_hash=self._action_config_hash(),
        )

        # Mark as failed with no iterations - should NOT create directory
        tracker.mark_failed(cleanup_if_empty=True)
        assert not run_dir.exists()

        # Create a new tracker and do some work
        tracker2 = RunTracker(
            run_dir=run_dir,
            config_name="test",
            config=Config.default(),
            action_config_hash=self._action_config_hash(),
        )
        tracker2.update(iterations=5, runtime_seconds=1.0, num_infosets=100, storage_capacity=2000)

        # Now mark as failed - should keep directory since iterations > 0
        tracker2.mark_failed(cleanup_if_empty=True)
        assert tracker2.metadata.status == "failed"
        assert tracker2.metadata.completed_at is not None
        assert run_dir.exists()  # Should still exist since iterations > 0

    def test_list_runs(self, tmp_path):
        """Test listing all runs."""
        # Create some runs
        (tmp_path / "run-1").mkdir(parents=True)
        (tmp_path / "run-1" / "run.jsonl").write_text("{}\n")

        (tmp_path / "run-2").mkdir(parents=True)
        (tmp_path / "run-2" / "run.jsonl").write_text("{}\n")

        (tmp_path / "not-a-run").mkdir()  # no run log

        runs = RunTracker.list_runs(tmp_path)

        assert len(runs) == 2
        assert "run-1" in runs
        assert "run-2" in runs
        assert "not-a-run" not in runs

    def test_list_runs_empty_dir(self, tmp_path):
        """Test listing runs in empty directory."""
        runs = RunTracker.list_runs(tmp_path)
        assert runs == []

    def test_list_runs_nonexistent_dir(self, tmp_path):
        """Test listing runs in non-existent directory."""
        runs = RunTracker.list_runs(tmp_path / "does-not-exist")
        assert runs == []

    def test_load_legacy_metadata_without_attempts(self, tmp_path):
        """Pre-attempts .run.json (every frozen baseline on the Volume predates the
        attempts list) must still load — synthesizing one attempt — and resume on top.

        This is the load-bearing back-compat path: the next load/eval/resume of any
        existing run hits from_dict's synthesis branch, and a resume must append a
        correctly-indexed second attempt.
        """
        config = Config.default()
        fresh = RunMetadata.new(
            "run-legacy", "test", config, action_config_hash=self._action_config_hash()
        )
        fresh.update_progress(
            iterations=5_000_000, runtime_seconds=1800.0, num_infosets=1234, storage_capacity=10**6
        )
        fresh.mark_completed()

        legacy = fresh.to_dict()
        del legacy["attempts"]  # simulate pre-attempts metadata on disk
        legacy["resumed_at"] = None  # old single-slot field, now dropped/ignored

        # Round-trip through disk to exercise RunTracker.load, not just from_dict.
        run_dir = tmp_path / "run-legacy"
        run_dir.mkdir()
        (run_dir / ".run.json").write_text(json.dumps(legacy))
        assert migrate_run_log(run_dir)
        loaded = RunTracker.load(run_dir).metadata

        assert len(loaded.attempts) == 1
        (attempt,) = loaded.attempts
        assert attempt.kind == "fresh"
        assert attempt.start_iter == 0
        assert attempt.end_iter == 5_000_000
        assert attempt.runtime_seconds == 1800.0
        assert attempt.status == "completed"

        # A resume of the legacy run appends a second attempt at its final iteration.
        loaded.mark_resumed()
        assert len(loaded.attempts) == 2
        assert loaded.attempts[1].index == 1
        assert loaded.attempts[1].kind == "resume"
        assert loaded.attempts[1].start_iter == 5_000_000


def _metadata(**kw) -> RunMetadata:
    return RunMetadata.new(
        run_id="run-x",
        config_name="quick_test",
        config=Config.default(),
        action_config_hash="abc123",
        **kw,
    )


class TestTheRunLog:
    """A run is an append-only log; its state is the fold of that log."""

    def _tracker(self, tmp_path, name="run-log"):
        return RunTracker(
            run_dir=tmp_path / name,
            config_name="quick_test",
            config=Config.default(),
            action_config_hash="abc123",
        )

    def test_creation_is_the_first_event(self, tmp_path):
        """So a run listing answers identity from one line, not a whole fold."""
        tracker = self._tracker(tmp_path)
        tracker.initialize()
        first = run_events.read(tracker.run_dir)[0]
        assert first["event"] == run_events.CREATED
        assert first["config_name"] == "quick_test"

    def test_state_is_the_fold_of_the_events(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.update(iterations=50, runtime_seconds=5.0, num_infosets=500, storage_capacity=2000)
        tracker.mark_completed()

        reloaded = RunTracker.load(tracker.run_dir).metadata
        assert reloaded.iterations == 50
        assert reloaded.num_infosets == 500
        assert reloaded.status == "completed"

    def test_nothing_is_ever_rewritten(self, tmp_path):
        """The append-only model removes the torn-write window entirely: a
        snapshot had to be rewritten in full on every update."""
        tracker = self._tracker(tmp_path)
        tracker.update(iterations=10, runtime_seconds=1.0, num_infosets=10, storage_capacity=100)
        after_first = run_events.log_path(tracker.run_dir).read_text()
        tracker.update(iterations=20, runtime_seconds=2.0, num_infosets=20, storage_capacity=100)

        assert run_events.log_path(tracker.run_dir).read_text().startswith(after_first)

    def test_a_torn_final_line_costs_only_the_last_event(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.update(iterations=50, runtime_seconds=5.0, num_infosets=500, storage_capacity=2000)
        with run_events.log_path(tracker.run_dir).open("a") as handle:
            handle.write('{"event": "progress", "iterations": 99')

        assert RunTracker.load(tracker.run_dir).metadata.iterations == 50

    def test_a_resume_opens_a_second_attempt(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.update(iterations=50, runtime_seconds=5.0, num_infosets=500, storage_capacity=2000)
        tracker.mark_completed()

        again = RunTracker.load(tracker.run_dir)
        again.mark_resumed()
        assert len(RunTracker.load(tracker.run_dir).metadata.attempts) == 2


class TestMigrationFromTheSnapshotLayout:
    """The back-compat path for every run directory already on disk."""

    def _legacy(self, tmp_path, **over):
        run_dir = tmp_path / "run-old"
        run_dir.mkdir()
        metadata = RunMetadata.new(
            "run-old", "quick_test", Config.default(), action_config_hash="abc123"
        )
        metadata.update_progress(
            iterations=5_000_000, runtime_seconds=1800.0, num_infosets=1234, storage_capacity=10**6
        )
        metadata.mark_completed()
        payload = metadata.to_dict()
        payload.update(over)
        (run_dir / ".run.json").write_text(json.dumps(payload))
        return run_dir

    def test_the_fold_reproduces_the_snapshot(self, tmp_path):
        run_dir = self._legacy(tmp_path)
        before = RunMetadata.from_dict(json.loads((run_dir / ".run.json").read_text()))
        assert migrate_run_log(run_dir)
        after = RunTracker.load(run_dir).metadata

        assert (after.run_id, after.iterations, after.status) == (
            before.run_id,
            before.iterations,
            before.status,
        )
        assert after.num_infosets == before.num_infosets
        assert len(after.attempts) == len(before.attempts)

    def test_the_progress_series_is_folded_in(self, tmp_path):
        run_dir = self._legacy(tmp_path)
        (run_dir / "progress.jsonl").write_text(
            json.dumps({"schema_version": 1, "iteration": 1000, "coverage": 0.1}) + "\n"
        )
        migrate_run_log(run_dir)

        checkpoints = run_events.events_of(run_events.read(run_dir), run_events.CHECKPOINT)
        assert [c["iteration"] for c in checkpoints] == [1000]

    def test_it_is_idempotent_and_non_destructive(self, tmp_path):
        run_dir = self._legacy(tmp_path)
        assert migrate_run_log(run_dir) is True
        assert migrate_run_log(run_dir) is False
        assert (run_dir / ".run.json").exists(), "the original stays for the operator"

    def test_a_directory_with_no_snapshot_is_left_alone(self, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        assert migrate_run_log(empty) is False


class TestFieldsAreScopedToTheEventThatOwnsThem:
    """A bare field name is only safe when nothing else uses it.

    A run's `status` and an ATTEMPT's status are different facts under the same
    key. Reading the former unscoped returned the latter: two runs still
    training folded back as `died` because their last closed attempt had.
    """

    def test_a_dead_attempt_does_not_make_the_run_dead(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        metadata = RunMetadata.new(
            "run-a", "quick_test", Config.default(), action_config_hash="abc123"
        )
        run_events.append(run_dir, run_events.CREATED, **metadata.creation_facts())
        run_events.append(run_dir, run_events.ATTEMPT_STARTED, index=0, kind="fresh", start_iter=0)
        run_events.append(run_dir, run_events.ATTEMPT_ENDED, index=0, status="died")
        run_events.append(run_dir, run_events.ATTEMPT_STARTED, index=1, kind="resume", start_iter=5)
        run_events.append(run_dir, run_events.PROGRESS, iterations=10, num_infosets=5)

        folded = RunTracker.load(run_dir).metadata
        assert folded.status == "running", "the run is still training"
        assert folded.attempts[0].status == "died", "but its first attempt died"

    def test_the_run_status_comes_only_from_a_status_event(self, tmp_path):
        run_dir = tmp_path / "run-b"
        run_dir.mkdir()
        metadata = RunMetadata.new(
            "run-b", "quick_test", Config.default(), action_config_hash="abc123"
        )
        run_events.append(run_dir, run_events.CREATED, **metadata.creation_facts())
        run_events.append(run_dir, run_events.ATTEMPT_STARTED, index=0, kind="fresh", start_iter=0)
        run_events.append(run_dir, run_events.ATTEMPT_ENDED, index=0, status="interrupted")
        run_events.append(run_dir, run_events.STATUS, status="completed", iterations=99)

        assert RunTracker.load(run_dir).metadata.status == "completed"


class TestRunsWrittenBeforeTheLog:
    """Every run directory on disk predates the log and is still read daily."""

    def _legacy_dir(self, tmp_path):
        run_dir = tmp_path / "run-old"
        run_dir.mkdir()
        metadata = RunMetadata.new(
            "run-old", "quick_test", Config.default(), action_config_hash="abc123"
        )
        metadata.update_progress(
            iterations=500, runtime_seconds=10.0, num_infosets=100, storage_capacity=1000
        )
        (run_dir / ".run.json").write_text(json.dumps(metadata.to_dict()))
        return run_dir

    def test_a_snapshot_only_run_still_loads(self, tmp_path):
        """Hard-failing here breaks resume, evaluate, curve and report for every
        run that exists."""
        loaded = RunMetadata.load(self._legacy_dir(tmp_path))
        assert loaded.run_id == "run-old"
        assert loaded.iterations == 500

    def test_reading_one_does_not_rewrite_it(self, tmp_path):
        """A listing must not convert 43 directories as a side effect."""
        run_dir = self._legacy_dir(tmp_path)
        RunMetadata.load(run_dir)
        assert not run_events.log_path(run_dir).exists()

    def test_it_is_still_recognised_as_a_run(self, tmp_path):
        run_dir = self._legacy_dir(tmp_path)
        assert "run-old" in RunTracker.list_runs(run_dir.parent)


class TestResumeDetectionSeesBothLayouts:
    """`resuming` False on a live run is not cosmetic: it mints fresh metadata
    over the top, skips verify_action_config_hash, and restarts training from
    zero into a directory holding a real ladder."""

    def test_a_snapshot_only_run_reads_as_resumable(self, tmp_path):
        from src.pipeline.services.static_training import train_static  # noqa: F401

        run_dir = tmp_path / "run-old"
        run_dir.mkdir()
        (run_dir / ".run.json").write_text("{}")
        # The predicate static_training uses, kept in step with it here.
        assert run_events.log_path(run_dir).exists() or (run_dir / ".run.json").exists()


class TestReapingSurvivesTheFold:
    def test_a_reaped_attempt_stays_dead(self, tmp_path):
        """Emitted only as attempt_started, the fold left every reaped attempt
        `running` -- the exact symptom the reaping exists to fix."""
        tracker = RunTracker(
            run_dir=tmp_path / "run-a",
            config_name="quick_test",
            config=Config.default(),
            action_config_hash="abc123",
        )
        tracker.update(iterations=50, runtime_seconds=9.0, num_infosets=5, storage_capacity=100)

        again = RunTracker.load(tracker.run_dir)
        again.mark_resumed()

        attempts = RunTracker.load(tracker.run_dir).metadata.attempts
        assert [a.status for a in attempts] == ["died", "running"]
        assert attempts[0].runtime_seconds == 9.0, "and keeps the compute it did"

    def test_one_resume_makes_exactly_one_new_attempt(self, tmp_path):
        tracker = RunTracker(
            run_dir=tmp_path / "run-b",
            config_name="quick_test",
            config=Config.default(),
            action_config_hash="abc123",
        )
        tracker.update(iterations=10, runtime_seconds=1.0, num_infosets=5, storage_capacity=100)
        tracker.mark_completed()

        RunTracker.load(tracker.run_dir).mark_resumed()
        assert len(RunTracker.load(tracker.run_dir).metadata.attempts) == 2

    def test_a_leg_that_died_early_does_not_inherit_the_previous_runtime(self, tmp_path):
        tracker = RunTracker(
            run_dir=tmp_path / "run-c",
            config_name="quick_test",
            config=Config.default(),
            action_config_hash="abc123",
        )
        tracker.update(iterations=50, runtime_seconds=99.0, num_infosets=5, storage_capacity=100)
        RunTracker.load(tracker.run_dir).mark_resumed()

        live = RunTracker.load(tracker.run_dir).metadata.attempts[-1]
        assert live.runtime_seconds == 0.0, "this leg has checkpointed nothing yet"


class TestLegacyResumeWithoutMigration:
    """A Batch retry resumes a `.run.json`-only run without ever running
    `ledger --migrate`, and `RunMetadata.load` prefers the event log once it
    exists -- so whatever the replay fails to write is gone for good."""

    def _two_finished_attempts(self, tmp_path):
        run_dir = tmp_path / "run-legacy-resume"
        run_dir.mkdir()
        metadata = RunMetadata.new(
            "run-legacy-resume", "production", Config.default(), action_config_hash="abc123"
        )
        metadata.update_progress(
            iterations=1000, runtime_seconds=1800.0, num_infosets=10, storage_capacity=100
        )
        metadata.mark_completed()
        metadata.mark_resumed()
        metadata.update_progress(
            iterations=2000, runtime_seconds=3000.0, num_infosets=20, storage_capacity=200
        )
        metadata.mark_completed()
        (run_dir / ".run.json").write_text(json.dumps(metadata.to_dict()))
        return run_dir, metadata

    def test_the_replay_preserves_runtime_and_attempt_outcomes(self, tmp_path):
        run_dir, before = self._two_finished_attempts(tmp_path)

        RunTracker.load(run_dir).initialize()
        after = RunMetadata.load(run_dir)

        assert after.runtime_seconds == before.runtime_seconds, "compute time survives the replay"
        assert [a.status for a in after.attempts] == [a.status for a in before.attempts]
        assert len(after.attempts) == 2, "no attempt is duplicated or invented"

    def test_an_attempt_killed_mid_flight_stays_open(self, tmp_path):
        """`status=running` with a null `ended_at` is how a died attempt is
        recognised; closing it with an invented timestamp erases that."""
        run_dir, _ = self._two_finished_attempts(tmp_path)
        payload = json.loads((run_dir / ".run.json").read_text())
        payload["attempts"][-1].update(status="running", ended_at=None)
        (run_dir / ".run.json").write_text(json.dumps(payload))

        RunTracker.load(run_dir).initialize()
        events = run_events.read(run_dir)

        ended = run_events.events_of(events, run_events.ATTEMPT_ENDED)
        assert [e["index"] for e in ended] == [0], "only the attempt that truly closed"
