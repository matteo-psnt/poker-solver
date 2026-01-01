"""Tests for training component builders."""

import json

import pytest

from src.actions.action_model import ActionModel
from src.bucketing.config import PrecomputeConfig
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.shared_array import SharedArrayStorage
from src.training import components
from src.utils.config import Config
from tests.test_helpers import DummyCardAbstraction


class TestBuildCardAbstraction:
    """Tests for build_card_abstraction."""

    def test_build_fails_with_invalid_config_name(self):
        """Test that building fails when config has no matching abstraction."""
        config = Config.default().merge({"card_abstraction": {"config": "nonexistent_config_xyz"}})

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            components.build_card_abstraction(config)

    def test_build_loads_unique_hash_match(self, tmp_path, monkeypatch):
        """Build uses the unique abstraction path matching the expected config hash."""
        expected_hash = PrecomputeConfig.from_yaml("default").get_config_hash()
        base_path = tmp_path / "data" / "combo_abstraction"
        candidate = base_path / "default-a"
        candidate.mkdir(parents=True)
        with open(candidate / "metadata.json", "w") as f:
            json.dump(
                {
                    "config_hash": expected_hash,
                    "config": {
                        "config_name": "default",
                    },
                },
                f,
            )

        loaded_path = None

        def _mock_load(path):
            nonlocal loaded_path
            loaded_path = path
            return DummyCardAbstraction()

        monkeypatch.setattr(components.PostflopPrecomputer, "load", _mock_load)
        config = Config.default().merge({"card_abstraction": {"config": "default"}})

        abstraction = components.build_card_abstraction(
            config,
            abstractions_dir=base_path,
        )

        assert isinstance(abstraction, DummyCardAbstraction)
        assert loaded_path == candidate

    def test_build_fails_when_multiple_hash_matches(self, tmp_path):
        """Multiple matching abstractions should fail and ask for explicit path."""
        expected_hash = PrecomputeConfig.from_yaml("default").get_config_hash()
        base_path = tmp_path / "data" / "combo_abstraction"
        for name in ["default-a", "default-b"]:
            candidate = base_path / name
            candidate.mkdir(parents=True)
            with open(candidate / "metadata.json", "w") as f:
                json.dump(
                    {
                        "config_hash": expected_hash,
                        "config": {
                            "config_name": "default",
                        },
                    },
                    f,
                )

        config = Config.default().merge({"card_abstraction": {"config": "default"}})
        with pytest.raises(ValueError, match="Multiple combo abstractions found"):
            components.build_card_abstraction(
                config,
                abstractions_dir=base_path,
            )


class TestBuildStorage:
    """Tests for build_storage."""

    def test_build_storage_with_checkpointing(self, tmp_path):
        """Test building storage with checkpointing enabled."""
        config = Config.default().merge({"storage": {"checkpoint_enabled": True}})

        storage = components.build_storage(config, run_dir=tmp_path)

        assert isinstance(storage, SharedArrayStorage)
        assert storage.num_infosets() == 0
        assert storage.checkpoint_dir == tmp_path

    def test_build_storage_without_checkpointing(self):
        """Test building storage without checkpointing."""
        config = Config.default().merge({"storage": {"checkpoint_enabled": False}})

        storage = components.build_storage(config, run_dir=None)

        assert isinstance(storage, SharedArrayStorage)
        assert storage.num_infosets() == 0
        assert storage.checkpoint_dir is None

    def test_build_storage_checkpointing_requires_run_dir(self):
        """Test that checkpointing enabled without run_dir raises error."""
        config = Config.default().merge({"storage": {"checkpoint_enabled": True}})

        with pytest.raises(ValueError, match="run_dir is required"):
            components.build_storage(config, run_dir=None)


class TestBuildSolver:
    """Tests for build_solver."""

    def test_build_solver_basic(self):
        """Test building solver with basic components."""
        config = Config.default()

        action_model = ActionModel(config)
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = components.build_solver(config, action_model, card_abs, storage)

        assert isinstance(solver, MCCFRSolver)
        assert solver.action_model == action_model
        assert solver.card_abstraction == card_abs
        assert solver.storage == storage
        assert solver.iteration == 0

    def test_build_solver_with_seed(self):
        """Test that solver uses configured seed."""
        config = Config.default().merge({"system": {"seed": 42}})

        action_model = ActionModel(config)
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = components.build_solver(config, action_model, card_abs, storage)

        assert isinstance(solver, MCCFRSolver)
        # Solver should be initialized successfully

    def test_build_solver_respects_game_config(self):
        """Test that game configuration is passed to solver."""
        config = Config.default().merge(
            {"game": {"starting_stack": 500, "small_blind": 5, "big_blind": 10}}
        )

        action_model = ActionModel(config)
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = components.build_solver(config, action_model, card_abs, storage)

        assert isinstance(solver, MCCFRSolver)
        # Solver should be created with the custom game config


class TestComponentIntegration:
    """Integration tests for component builders."""

    def test_build_all_components_together(self, tmp_path):
        """Test building all components together for a training session."""
        config = Config.default().merge({"storage": {"checkpoint_enabled": True}})

        # Build all components
        action_model = ActionModel(config)
        storage = components.build_storage(config, run_dir=tmp_path)

        # Use dummy abstraction since we don't have real data
        card_abs = DummyCardAbstraction()

        solver = components.build_solver(config, action_model, card_abs, storage)

        # Verify everything is connected
        assert isinstance(action_model, ActionModel)
        assert isinstance(storage, SharedArrayStorage)
        assert isinstance(solver, MCCFRSolver)
        assert solver.storage == storage
        assert solver.action_model == action_model

    def test_build_with_checkpointing_creates_directory(self, tmp_path):
        """Test that storage with checkpointing properly initializes."""
        config = Config.default().merge({"storage": {"checkpoint_enabled": True}})

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()  # Create run directory
        storage = components.build_storage(config, run_dir=run_dir)

        assert isinstance(storage, SharedArrayStorage)
        assert storage.checkpoint_dir == run_dir
