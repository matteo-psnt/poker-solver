"""Tests for Trainer class."""

import pytest

from src.actions.betting_actions import BettingActions
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training import components
from src.training.trainer import TrainingSession
from src.utils.config import Config
from tests.test_helpers import DummyCardAbstraction


@pytest.fixture
def config_with_dummy_abstraction(tmp_path, monkeypatch):
    """Create a config with dummy card abstraction."""
    config = Config.default()
    # Use tmp_path for runs_dir to prevent creating runs in data/runs
    config.set("training.runs_dir", str(tmp_path / "runs"))

    # Mock the builder to return DummyCardAbstraction
    def mock_build_card_abstraction(config, prompt_user=False, auto_compute=False):
        return DummyCardAbstraction()

    monkeypatch.setattr(components, "build_card_abstraction", mock_build_card_abstraction)

    return config


class TestTrainer:
    """Tests for TrainingSession class."""

    def test_create_trainer(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)

        assert trainer.config is not None
        assert trainer.solver is not None
        assert trainer.action_abstraction is not None
        assert trainer.card_abstraction is not None

    def test_build_action_abstraction(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)

        action_abs = trainer.action_abstraction
        assert isinstance(action_abs, BettingActions)

    def test_build_storage_memory(self, config_with_dummy_abstraction):
        config_with_dummy_abstraction.set("storage.backend", "memory")

        trainer = TrainingSession(config_with_dummy_abstraction)
        assert isinstance(trainer.storage, InMemoryStorage)

    def test_build_solver(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)

        solver = trainer.solver
        assert isinstance(solver, MCCFRSolver)
        assert solver.iteration == 0

    def test_str_representation(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)
        s = str(trainer)

        assert "TrainingSession" in s

    def test_initialization_failure_no_directory(self, tmp_path, monkeypatch):
        """Test that failed initialization doesn't create run directory."""

        # Mock card abstraction to fail
        def mock_build_card_abstraction_fail(*args, **kwargs):
            raise ValueError("Abstraction not found")

        monkeypatch.setattr(components, "build_card_abstraction", mock_build_card_abstraction_fail)

        config = Config.default()
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("storage.backend", "memory")

        # Attempt to create trainer (should fail)
        with pytest.raises(ValueError, match="Abstraction not found"):
            TrainingSession(config)

        # Check that no run directory was created
        runs_dir = tmp_path / "runs"
        if runs_dir.exists():
            run_dirs = list(runs_dir.glob("run*"))
            assert len(run_dirs) == 0, (
                f"Run directory should not exist after init failure, found: {run_dirs}"
            )
