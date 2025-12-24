"""Tests for Trainer class."""

import pytest

from src.actions.betting_actions import BettingActions
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training.trainer import Trainer
from src.utils.config import Config
from tests.test_helpers import DummyCardAbstraction


@pytest.fixture
def config_with_dummy_abstraction(tmp_path, monkeypatch):
    """Create a config with dummy card abstraction."""
    config = Config.default()
    # Use tmp_path for runs_dir to prevent creating runs in data/runs
    config.set("training.runs_dir", str(tmp_path / "runs"))

    # Mock the builder to return DummyCardAbstraction
    from src.training import builders

    def mock_build_card_abstraction(config, prompt_user=False, auto_compute=False):
        return DummyCardAbstraction()

    monkeypatch.setattr(builders, "build_card_abstraction", mock_build_card_abstraction)

    return config


class TestTrainer:
    """Tests for Trainer class."""

    def test_create_trainer(self, config_with_dummy_abstraction):
        trainer = Trainer(config_with_dummy_abstraction)

        assert trainer.config is not None
        assert trainer.solver is not None
        assert trainer.action_abstraction is not None
        assert trainer.card_abstraction is not None

    def test_build_action_abstraction(self, config_with_dummy_abstraction):
        trainer = Trainer(config_with_dummy_abstraction)

        action_abs = trainer.action_abstraction
        assert isinstance(action_abs, BettingActions)

    def test_build_storage_memory(self, config_with_dummy_abstraction):
        config_with_dummy_abstraction.set("storage.backend", "memory")

        trainer = Trainer(config_with_dummy_abstraction)
        assert isinstance(trainer.storage, InMemoryStorage)

    def test_build_solver(self, config_with_dummy_abstraction):
        trainer = Trainer(config_with_dummy_abstraction)

        solver = trainer.solver
        assert isinstance(solver, MCCFRSolver)
        assert solver.iteration == 0

    @pytest.mark.timeout(5)
    def test_train_executes(self, tmp_path, monkeypatch):
        """Test that training runs without errors."""
        # Mock the builder
        from src.training import builders

        monkeypatch.setattr(
            builders, "build_card_abstraction", lambda *args, **kwargs: DummyCardAbstraction()
        )

        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.checkpoint_frequency", 1000)
        config.set("training.log_frequency", 1)
        config.set("training.verbose", False)
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("storage.backend", "memory")

        trainer = Trainer(config)
        results = trainer.train(num_iterations=1)

        assert results["total_iterations"] == 1
        assert results["final_infosets"] > 0
        assert "avg_utility" in results
        assert "elapsed_time" in results

    @pytest.mark.timeout(5)
    def test_train_with_iterations_override(self, tmp_path, monkeypatch):
        """Test overriding num_iterations."""
        # Mock the builder
        from src.training import builders

        monkeypatch.setattr(
            builders, "build_card_abstraction", lambda *args, **kwargs: DummyCardAbstraction()
        )

        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("storage.backend", "memory")
        config.set("training.checkpoint_frequency", 1000)

        trainer = Trainer(config)
        results = trainer.train(num_iterations=1)  # Override

        assert results["total_iterations"] == 1

    @pytest.mark.slow
    @pytest.mark.timeout(20)
    def test_train_creates_checkpoint(self, tmp_path, monkeypatch):
        """Test that checkpoints are created."""
        # Mock the builder
        from src.training import builders

        monkeypatch.setattr(
            builders, "build_card_abstraction", lambda *args, **kwargs: DummyCardAbstraction()
        )

        config = Config.default()
        config.set("training.num_iterations", 1)
        config.set("training.verbose", False)
        config.set("training.checkpoint_frequency", 1)  # Checkpoint every iteration
        config.set("training.runs_dir", str(tmp_path / "runs"))

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

    def test_str_representation(self, config_with_dummy_abstraction):
        trainer = Trainer(config_with_dummy_abstraction)
        s = str(trainer)

        assert "Trainer" in s
