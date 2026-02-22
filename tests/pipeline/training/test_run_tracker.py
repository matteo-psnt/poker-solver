"""Tests for RunTracker."""

import json

from src.core.actions.action_model import ActionModel
from src.pipeline.training.run_tracker import RunTracker
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
        metadata_file = run_dir / ".run.json"
        assert not metadata_file.exists()

        # Initialize to create file
        tracker.initialize()
        assert metadata_file.exists()

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

        metadata_file = run_dir / ".run.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Load tracker
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
        (tmp_path / "run-1" / ".run.json").parent.mkdir(parents=True)
        (tmp_path / "run-1" / ".run.json").write_text("{}")

        (tmp_path / "run-2" / ".run.json").parent.mkdir(parents=True)
        (tmp_path / "run-2" / ".run.json").write_text("{}")

        (tmp_path / "not-a-run").mkdir()  # No .run.json

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
