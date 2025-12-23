"""Tests for checkpoint management."""

import tempfile
from pathlib import Path

from src.abstraction.action_abstraction import ActionAbstraction
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training.checkpoint import RunManager
from tests.test_helpers import DummyCardAbstraction


class TestRunManager:
    """Tests for RunManager."""

    def test_create_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(Path(tmpdir))

            assert manager.base_checkpoint_dir == Path(tmpdir)
            assert manager.config_name == "default"
            assert manager.run_id.startswith("run_")
            # Directory not created until first save
            assert not manager.initialized

    def test_create_manager_with_config_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(Path(tmpdir), config_name="test_config")

            assert manager.config_name == "test_config"

    def test_create_manager_with_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(Path(tmpdir), config_name="test", run_id="run_test123")

            assert manager.run_id == "run_test123"
            assert "run_test123" in str(manager.checkpoint_dir)

    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create solver
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            # Train briefly to create infosets
            solver.train(num_iterations=2, verbose=False)

            # Save checkpoint
            manager = RunManager(Path(tmpdir))
            checkpoint_path = manager.save(solver, iteration=2)

            assert checkpoint_path.exists()

            # Check metadata files exist
            assert (manager.checkpoint_dir / "run_metadata.json").exists()
            assert (manager.checkpoint_dir / "checkpoint_manifest.json").exists()

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = RunManager(Path(tmpdir))

            # Initially no checkpoints
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 0

            # Save multiple checkpoints
            solver.train(num_iterations=2, verbose=False)
            manager.save(solver, iteration=2)

            solver.train(num_iterations=2, verbose=False)
            manager.save(solver, iteration=4)

            # Check list
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2
            assert checkpoints[0]["iteration"] == 2
            assert checkpoints[1]["iteration"] == 4

    def test_get_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            solver.train(num_iterations=2, verbose=False)

            manager = RunManager(Path(tmpdir))
            manager.save(solver, iteration=2)

            # Get checkpoint
            info = manager.get_checkpoint(iteration=2)

            assert info is not None
            assert info["iteration"] == 2
            assert info["num_infosets"] > 0

    def test_get_nonexistent_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(Path(tmpdir))

            info = manager.get_checkpoint(iteration=999)

            assert info is None

    def test_get_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = RunManager(Path(tmpdir))

            # No checkpoints initially
            latest = manager.get_latest_checkpoint()
            assert latest is None

            # Save multiple checkpoints
            solver.train(num_iterations=2, verbose=False)
            manager.save(solver, iteration=2)

            solver.train(num_iterations=2, verbose=False)
            manager.save(solver, iteration=4)

            solver.train(num_iterations=2, verbose=False)
            manager.save(solver, iteration=6)

            # Get latest
            latest = manager.get_latest_checkpoint()

            assert latest is not None
            assert latest["iteration"] == 6

    def test_get_latest_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = RunManager(Path(tmpdir))

            # No checkpoints initially
            assert manager.get_latest_iteration() == 0

            # Save checkpoints
            solver.train(num_iterations=3, verbose=False)
            manager.save(solver, iteration=3)

            assert manager.get_latest_iteration() == 3

    def test_list_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            # Create multiple runs with actual checkpoints (so metadata exists)
            manager1 = RunManager(Path(tmpdir), run_id="run_001")
            solver.train(num_iterations=2, verbose=False)
            manager1.save(solver, iteration=2)

            manager2 = RunManager(Path(tmpdir), run_id="run_002")
            solver.train(num_iterations=2, verbose=False)
            manager2.save(solver, iteration=4)

            manager3 = RunManager(Path(tmpdir), run_id="run_003")
            solver.train(num_iterations=2, verbose=False)
            manager3.save(solver, iteration=6)

            # List runs
            runs = RunManager.list_runs(Path(tmpdir))

            assert len(runs) == 3
            assert "run_001" in runs
            assert "run_002" in runs
            assert "run_003" in runs

    def test_list_runs_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = RunManager.list_runs(Path(tmpdir))
            assert len(runs) == 0

    def test_list_runs_nonexistent_directory(self):
        runs = RunManager.list_runs(Path("/nonexistent/directory"))
        assert len(runs) == 0

    def test_from_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            # Create a run with actual checkpoint
            manager1 = RunManager(Path(tmpdir), run_id="run_test")
            solver.train(num_iterations=2, verbose=False)
            manager1.save(solver, iteration=2)

            # Load from run ID
            manager2 = RunManager.from_run_id(
                Path(tmpdir), run_id="run_test", config_name="test_config"
            )

            assert manager2.run_id == "run_test"
            assert manager2.checkpoint_dir == manager1.checkpoint_dir

    def test_str_representation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(Path(tmpdir), run_id="run_test")

            s = str(manager)

            assert "RunManager" in s
            assert "run_test" in s

    def test_update_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = RunManager(Path(tmpdir))
            solver.train(num_iterations=2, verbose=False)  # Reduced from 10
            manager.save(solver, iteration=2)  # Match iteration count

            # Update stats
            manager.update_stats(
                total_iterations=10,
                total_runtime_seconds=60.0,
                num_infosets=100,
                cache_hit_rate=0.8,
                avg_traversal_depth=15.0,
            )

            assert manager.run_metadata is not None
            assert manager.run_metadata.statistics.total_iterations == 10

    def test_mark_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action_abs = ActionAbstraction()
            card_abs = DummyCardAbstraction()
            storage = InMemoryStorage()
            solver = MCCFRSolver(action_abs, card_abs, storage, config={"seed": 42})

            manager = RunManager(Path(tmpdir))
            solver.train(num_iterations=2, verbose=False)
            manager.save(solver, iteration=2)

            manager.mark_completed()

            assert manager.run_metadata.status == "completed"
