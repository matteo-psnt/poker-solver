"""Tests for Trainer class."""

import pytest

from src.abstraction.action_abstraction import ActionAbstraction
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training.trainer import Trainer
from src.utils.config import Config
from tests.test_helpers import create_minimal_bucketing


@pytest.fixture(scope="session")
def shared_bucketing_file(tmp_path_factory):
    """Create a single bucketing file for all tests in this session."""
    bucketing_path = tmp_path_factory.mktemp("data") / "test_bucketing.pkl"
    create_minimal_bucketing(save_path=bucketing_path)
    return bucketing_path


@pytest.fixture
def config_with_bucketing(tmp_path, shared_bucketing_file):
    """Create a config with a shared bucketing file."""
    config = Config.default()
    # Use tmp_path for runs_dir to prevent creating runs in data/runs
    config.set("training.runs_dir", str(tmp_path / "runs"))
    config.set("card_abstraction.bucketing_path", str(shared_bucketing_file))

    return config


class TestTrainer:
    """Tests for Trainer class."""

    def test_create_trainer(self, config_with_bucketing):
        trainer = Trainer(config_with_bucketing)

        assert trainer.config is not None
        assert trainer.solver is not None
        assert trainer.action_abstraction is not None
        assert trainer.card_abstraction is not None

    def test_build_action_abstraction(self, config_with_bucketing):
        trainer = Trainer(config_with_bucketing)

        action_abs = trainer.action_abstraction
        assert isinstance(action_abs, ActionAbstraction)

    # test_build_card_abstraction removed - now requires equity bucketing file

    def test_build_storage_memory(self, config_with_bucketing):
        config_with_bucketing.set("storage.backend", "memory")

        trainer = Trainer(config_with_bucketing)
        assert isinstance(trainer.storage, InMemoryStorage)

    def test_build_solver(self, config_with_bucketing):
        trainer = Trainer(config_with_bucketing)

        solver = trainer.solver
        assert isinstance(solver, MCCFRSolver)
        assert solver.iteration == 0

    @pytest.mark.timeout(15)
    def test_train_executes(self, tmp_path, shared_bucketing_file):
        """Test that training runs without errors."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.checkpoint_frequency", 1)
        config.set("training.log_frequency", 1)
        config.set("training.verbose", False)
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("card_abstraction.bucketing_path", str(shared_bucketing_file))

        trainer = Trainer(config)
        results = trainer.train(num_iterations=1)

        assert results["total_iterations"] == 1
        assert results["final_infosets"] > 0
        assert "avg_utility" in results
        assert "elapsed_time" in results

    @pytest.mark.timeout(10)
    def test_train_with_iterations_override(self, tmp_path, shared_bucketing_file):
        """Test overriding num_iterations."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("card_abstraction.bucketing_path", str(shared_bucketing_file))

        trainer = Trainer(config)
        results = trainer.train(num_iterations=1)  # Override

        assert results["total_iterations"] == 1

    @pytest.mark.timeout(15)
    def test_train_creates_checkpoint(self, tmp_path, shared_bucketing_file):
        """Test that checkpoints are created."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)
        config.set("training.checkpoint_frequency", 1)  # Checkpoint every iteration
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("card_abstraction.bucketing_path", str(shared_bucketing_file))

        trainer = Trainer(config)
        trainer.train(num_iterations=1)

        # Check checkpoint exists in run subdirectory
        runs_dir = tmp_path / "runs"
        run_dirs = list(runs_dir.glob("run*"))
        assert len(run_dirs) > 0, (
            f"No run subdirectory created in {runs_dir}. Contents: {list(runs_dir.iterdir())}"
        )

        # Check for manifest file
        manifest_file = run_dirs[0] / "snapshots.json"
        if not manifest_file.exists():
            manifest_file = run_dirs[0] / "checkpoint_manifest.json"
        assert manifest_file.exists(), (
            f"No manifest found in {run_dirs[0]}. Contents: {list(run_dirs[0].iterdir())}"
        )

    def test_str_representation(self, tmp_path, shared_bucketing_file):
        config = Config.default()
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("card_abstraction.bucketing_path", str(shared_bucketing_file))

        trainer = Trainer(config)
        s = str(trainer)

        assert "Trainer" in s

    @pytest.mark.timeout(15)
    def test_train_with_resume(self, tmp_path, shared_bucketing_file):
        """Test training with resume=True."""
        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("card_abstraction.bucketing_path", str(shared_bucketing_file))

        # First training session
        trainer1 = Trainer(config)
        trainer1.train(num_iterations=1)

        # Second session with resume (should find no checkpoint and start from 0)
        trainer2 = Trainer(config)
        results = trainer2.train(num_iterations=1, resume=True)

        # Since we used different trainer instances, resume won't find anything
        # This tests the resume=True code path
        assert "total_iterations" in results
