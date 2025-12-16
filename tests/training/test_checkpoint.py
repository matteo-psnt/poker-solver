"""Tests for checkpoint management."""

import tempfile
from pathlib import Path

from src.abstraction.action_abstraction import ActionAbstraction
from src.abstraction.card_abstraction import RankBasedBucketing
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training.checkpoint import CheckpointInfo, CheckpointManager


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_create_checkpoint_info(self):
        info = CheckpointInfo(
            iteration=100,
            timestamp="2025-12-16T12:00:00",
            num_infosets=500,
            config_name="test_config",
            checkpoint_dir=Path("/tmp/test"),
        )

        assert info.iteration == 100
        assert info.timestamp == "2025-12-16T12:00:00"
        assert info.num_infosets == 500
        assert info.config_name == "test_config"

    def test_to_dict(self):
        info = CheckpointInfo(
            iteration=100,
            timestamp="2025-12-16T12:00:00",
            num_infosets=500,
            config_name="test_config",
            checkpoint_dir=Path("/tmp/test"),
        )

        d = info.to_dict()

        assert d["iteration"] == 100
        assert d["timestamp"] == "2025-12-16T12:00:00"
        assert d["num_infosets"] == 500
        assert d["config_name"] == "test_config"
        assert d["checkpoint_dir"] == "/tmp/test"

    def test_from_dict(self):
        d = {
            "iteration": 100,
            "timestamp": "2025-12-16T12:00:00",
            "num_infosets": 500,
            "config_name": "test_config",
            "checkpoint_dir": "/tmp/test",
        }

        info = CheckpointInfo.from_dict(d)

        assert info.iteration == 100
        assert info.timestamp == "2025-12-16T12:00:00"
        assert info.num_infosets == 500
        assert info.config_name == "test_config"
        assert info.checkpoint_dir == Path("/tmp/test")

    def test_from_dict_without_optional_fields(self):
        d = {
            "iteration": 100,
            "timestamp": "2025-12-16T12:00:00",
            "num_infosets": 500,
        }

        info = CheckpointInfo.from_dict(d)

        assert info.iteration == 100
        assert info.config_name is None
        assert info.checkpoint_dir is None


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_create_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            assert manager.base_checkpoint_dir == Path(tmpdir)
            assert manager.config_name == "default"
            assert manager.run_id.startswith("run_")
            assert manager.checkpoint_dir.exists()

    def test_create_manager_with_config_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir), config_name="test_config")

            assert manager.config_name == "test_config"

    def test_create_manager_with_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir), config_name="test", run_id="run_test123")

            assert manager.run_id == "run_test123"
            assert "run_test123" in str(manager.checkpoint_dir)

    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create solver
            action_abs = ActionAbstraction()
            card_abs = RankBasedBucketing()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            # Train briefly to create infosets
            solver.train(num_iterations=5, verbose=False)

            # Save checkpoint
            manager = CheckpointManager(Path(tmpdir))
            checkpoint_path = manager.save(solver, iteration=5)

            assert checkpoint_path.exists()

            # Check metadata file exists
            metadata_path = manager.checkpoint_dir / "checkpoint_5_metadata.json"
            assert metadata_path.exists()

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = RankBasedBucketing()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = CheckpointManager(Path(tmpdir))

            # Initially no checkpoints
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 0

            # Save multiple checkpoints
            solver.train(num_iterations=5, verbose=False)
            manager.save(solver, iteration=5)

            solver.train(num_iterations=5, verbose=False)
            manager.save(solver, iteration=10)

            # Check list
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2
            assert checkpoints[0].iteration == 5
            assert checkpoints[1].iteration == 10

    def test_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = RankBasedBucketing()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            solver.train(num_iterations=5, verbose=False)

            manager = CheckpointManager(Path(tmpdir))
            manager.save(solver, iteration=5)

            # Load checkpoint
            info = manager.load_checkpoint(iteration=5)

            assert info is not None
            assert info.iteration == 5
            assert info.num_infosets > 0

    def test_load_nonexistent_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            info = manager.load_checkpoint(iteration=999)

            assert info is None

    def test_load_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = RankBasedBucketing()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = CheckpointManager(Path(tmpdir))

            # No checkpoints initially
            latest = manager.load_latest()
            assert latest is None

            # Save multiple checkpoints
            solver.train(num_iterations=5, verbose=False)
            manager.save(solver, iteration=5)

            solver.train(num_iterations=5, verbose=False)
            manager.save(solver, iteration=10)

            solver.train(num_iterations=5, verbose=False)
            manager.save(solver, iteration=15)

            # Load latest
            latest = manager.load_latest()

            assert latest is not None
            assert latest.iteration == 15

    def test_delete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = RankBasedBucketing()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            solver.train(num_iterations=5, verbose=False)

            manager = CheckpointManager(Path(tmpdir))
            manager.save(solver, iteration=5)

            # Check exists
            assert manager.load_checkpoint(5) is not None

            # Delete
            manager.delete_checkpoint(5)

            # Check deleted
            assert manager.load_checkpoint(5) is None

    def test_clean_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = RankBasedBucketing()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = CheckpointManager(Path(tmpdir))

            # Save 5 checkpoints
            for i in range(1, 6):
                solver.train(num_iterations=5, verbose=False)
                manager.save(solver, iteration=i * 5)

            # Check all exist
            assert len(manager.list_checkpoints()) == 5

            # Keep last 2
            manager.clean_old_checkpoints(keep_last_n=2)

            # Check only 2 remain
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2
            assert checkpoints[0].iteration == 20
            assert checkpoints[1].iteration == 25

    def test_clean_old_checkpoints_keeps_all_if_under_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = RankBasedBucketing()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = CheckpointManager(Path(tmpdir))

            # Save 3 checkpoints
            for i in range(1, 4):
                solver.train(num_iterations=5, verbose=False)
                manager.save(solver, iteration=i * 5)

            # Try to keep last 5 (more than we have)
            manager.clean_old_checkpoints(keep_last_n=5)

            # All should remain
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 3

    def test_list_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple runs (creating managers creates the directories)
            _manager1 = CheckpointManager(Path(tmpdir), run_id="run_001")
            _manager2 = CheckpointManager(Path(tmpdir), run_id="run_002")
            _manager3 = CheckpointManager(Path(tmpdir), run_id="run_003")

            # List runs
            runs = CheckpointManager.list_runs(Path(tmpdir))

            assert len(runs) == 3
            assert "run_001" in runs
            assert "run_002" in runs
            assert "run_003" in runs

    def test_list_runs_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = CheckpointManager.list_runs(Path(tmpdir))
            assert len(runs) == 0

    def test_list_runs_nonexistent_directory(self):
        runs = CheckpointManager.list_runs(Path("/nonexistent/directory"))
        assert len(runs) == 0

    def test_from_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a run
            manager1 = CheckpointManager(Path(tmpdir), run_id="run_test")

            # Load from run ID
            manager2 = CheckpointManager.from_run_id(
                Path(tmpdir), run_id="run_test", config_name="test_config"
            )

            assert manager2.run_id == "run_test"
            assert manager2.config_name == "test_config"
            assert manager2.checkpoint_dir == manager1.checkpoint_dir

    def test_str_representation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir), run_id="run_test")

            s = str(manager)

            assert "CheckpointManager" in s
            assert "run_test" in s
            assert "checkpoints=0" in s
