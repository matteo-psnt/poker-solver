"""Tests for Trainer class."""

import tempfile
from pathlib import Path

from src.abstraction.action_abstraction import ActionAbstraction
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training.trainer import Trainer
from src.utils.config import Config


class TestTrainer:
    """Tests for Trainer class."""

    def test_create_trainer(self):
        config = Config.default()
        trainer = Trainer(config)

        assert trainer.config is not None
        assert trainer.solver is not None
        assert trainer.action_abstraction is not None
        assert trainer.card_abstraction is not None

    def test_build_action_abstraction(self):
        config = Config.default()
        trainer = Trainer(config)

        action_abs = trainer.action_abstraction
        assert isinstance(action_abs, ActionAbstraction)

    # test_build_card_abstraction removed - now requires equity bucketing file

    def test_build_storage_memory(self):
        config = Config.default()
        config.set("storage.backend", "memory")

        trainer = Trainer(config)
        assert isinstance(trainer.storage, InMemoryStorage)

    def test_build_solver(self):
        config = Config.default()
        trainer = Trainer(config)

        solver = trainer.solver
        assert isinstance(solver, MCCFRSolver)
        assert solver.iteration == 0

    def test_train_executes(self):
        """Test that training runs without errors."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.checkpoint_frequency", 1)
        config.set("training.log_frequency", 1)
        config.set("training.verbose", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            trainer = Trainer(config)
            results = trainer.train(num_iterations=1)

            assert results["total_iterations"] == 1
            assert results["final_infosets"] > 0
            assert "avg_utility" in results
            assert "elapsed_time" in results

    def test_train_with_iterations_override(self):
        """Test overriding num_iterations."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            trainer = Trainer(config)
            results = trainer.train(num_iterations=1)  # Override

            assert results["total_iterations"] == 1

    def test_train_creates_checkpoint(self):
        """Test that checkpoints are created."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)
        config.set("training.checkpoint_frequency", 1)  # Checkpoint every iteration

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            trainer = Trainer(config)
            trainer.train(num_iterations=1)

            # Check checkpoint exists in run subdirectory
            checkpoint_dir = Path(tmpdir)
            run_dirs = list(checkpoint_dir.glob("run_*"))
            assert len(run_dirs) > 0, "No run subdirectory created"

            # Check for checkpoint manifest file
            manifest_file = run_dirs[0] / "checkpoint_manifest.json"
            assert manifest_file.exists(), "No checkpoint manifest found"

    def test_evaluate(self):
        """Test evaluation method."""
        config = Config.default()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            trainer = Trainer(config)
            results = trainer.evaluate()

            assert "num_infosets" in results

    def test_str_representation(self):
        config = Config.default()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            trainer = Trainer(config)
            s = str(trainer)

            assert "Trainer" in s

    def test_train_with_default_iterations(self):
        """Test training with num_iterations=None (uses config default)."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            trainer = Trainer(config)
            results = trainer.train(num_iterations=None)

            assert results["total_iterations"] == 1

    def test_train_with_resume(self):
        """Test training with resume=True."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            # First training session
            trainer1 = Trainer(config)
            trainer1.train(num_iterations=1)

            # Second session with resume (should find no checkpoint and start from 0)
            trainer2 = Trainer(config)
            results = trainer2.train(num_iterations=1, resume=True)

            # Since we used different trainer instances, resume won't find anything
            # This tests the resume=True code path
            assert "total_iterations" in results

    def test_train_non_verbose(self):
        """Test training with verbose=False."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("training.checkpoint_dir", tmpdir)

            trainer = Trainer(config)
            results = trainer.train(num_iterations=1)

            assert results["total_iterations"] == 1
