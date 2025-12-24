"""Tests for training component builders."""

import pytest

from src.actions.betting_actions import BettingActions
from src.solver.base import BaseSolver
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import DiskBackedStorage, InMemoryStorage
from src.training import components
from src.utils.config import Config
from tests.test_helpers import DummyCardAbstraction


class TestBuildActionAbstraction:
    """Tests for build_action_abstraction."""

    def test_build_default(self):
        """Test building action abstraction with default config."""
        config = Config.default()
        action_abs = components.build_action_abstraction(config)

        assert isinstance(action_abs, BettingActions)
        assert len(action_abs.preflop_raises) > 0
        assert action_abs.big_blind == 2

    def test_build_with_custom_big_blind(self):
        """Test building with custom big blind."""
        config = Config.default()
        config.set("game.big_blind", 10)

        action_abs = components.build_action_abstraction(config)

        assert isinstance(action_abs, BettingActions)
        assert action_abs.big_blind == 10


class TestBuildCardAbstraction:
    """Tests for build_card_abstraction."""

    def test_build_fails_without_abstraction(self):
        """Test that building fails when no abstraction is configured."""
        config = Config.default()
        config.set("card_abstraction.abstraction_path", None)
        config.set("card_abstraction.config", None)

        with pytest.raises(ValueError, match="card_abstraction requires either"):
            components.build_card_abstraction(config, prompt_user=False, auto_compute=False)

    def test_build_fails_with_missing_path(self):
        """Test that building fails when path doesn't exist."""
        config = Config.default()
        config.set("card_abstraction.abstraction_path", "/nonexistent/path.pkl")

        with pytest.raises(FileNotFoundError):
            components.build_card_abstraction(config, prompt_user=False, auto_compute=False)

    def test_build_fails_with_invalid_config_name(self):
        """Test that building fails when config has no matching abstraction."""
        config = Config.default()
        config.set("card_abstraction.abstraction_path", None)
        config.set("card_abstraction.config", "nonexistent_config_xyz")

        with pytest.raises(FileNotFoundError, match="No combo abstraction found"):
            components.build_card_abstraction(config, prompt_user=False, auto_compute=False)


class TestBuildStorage:
    """Tests for build_storage."""

    def test_build_memory_storage(self, tmp_path):
        """Test building in-memory storage."""
        config = Config.default()
        config.set("storage.backend", "memory")

        storage = components.build_storage(config, run_dir=tmp_path)

        assert isinstance(storage, InMemoryStorage)
        assert storage.num_infosets() == 0

    def test_build_disk_storage(self, tmp_path):
        """Test building disk-backed storage."""
        config = Config.default()
        config.set("storage.backend", "disk")

        storage = components.build_storage(config, run_dir=tmp_path)

        assert isinstance(storage, DiskBackedStorage)
        assert storage.num_infosets() == 0

    def test_build_invalid_backend(self, tmp_path):
        """Test that invalid backend raises error."""
        config = Config.default()
        config.set("storage.backend", "invalid_backend")

        with pytest.raises(ValueError, match="Unknown storage backend"):
            components.build_storage(config, run_dir=tmp_path)

    def test_build_disk_storage_with_cache_size(self, tmp_path):
        """Test that cache size configuration is applied."""
        config = Config.default()
        config.set("storage.backend", "disk")
        config.set("storage.cache_size", 5000)

        storage = components.build_storage(config, run_dir=tmp_path)

        assert isinstance(storage, DiskBackedStorage)
        assert storage.cache_size == 5000

    def test_build_disk_storage_with_flush_frequency(self, tmp_path):
        """Test that flush frequency configuration is applied."""
        config = Config.default()
        config.set("storage.backend", "disk")
        config.set("storage.flush_frequency", 500)

        storage = components.build_storage(config, run_dir=tmp_path)

        assert isinstance(storage, DiskBackedStorage)
        assert storage.flush_frequency == 500


class TestBuildSolver:
    """Tests for build_solver."""

    def test_build_solver_basic(self):
        """Test building solver with basic components."""
        config = Config.default()

        action_abs = components.build_action_abstraction(config)
        card_abs = DummyCardAbstraction()
        storage = InMemoryStorage()

        solver = components.build_solver(config, action_abs, card_abs, storage)

        assert isinstance(solver, MCCFRSolver)
        assert solver.action_abstraction == action_abs
        assert solver.card_abstraction == card_abs
        assert solver.storage == storage
        assert solver.iteration == 0

    def test_build_solver_with_seed(self):
        """Test that solver uses configured seed."""
        config = Config.default()
        config.set("system.seed", 42)

        action_abs = components.build_action_abstraction(config)
        card_abs = DummyCardAbstraction()
        storage = InMemoryStorage()

        solver = components.build_solver(config, action_abs, card_abs, storage)

        assert isinstance(solver, MCCFRSolver)
        # Solver should be initialized successfully

    def test_build_solver_respects_game_config(self):
        """Test that game configuration is passed to solver."""
        config = Config.default()
        config.set("game.starting_stack", 500)
        config.set("game.small_blind", 5)
        config.set("game.big_blind", 10)

        action_abs = components.build_action_abstraction(config)
        card_abs = DummyCardAbstraction()
        storage = InMemoryStorage()

        solver = components.build_solver(config, action_abs, card_abs, storage)

        assert isinstance(solver, BaseSolver)
        # Solver should be created with the custom game config


class TestComponentIntegration:
    """Integration tests for component builders."""

    def test_build_all_components_together(self, tmp_path):
        """Test building all components together for a training session."""
        config = Config.default()
        config.set("storage.backend", "memory")

        # Build all components
        action_abs = components.build_action_abstraction(config)
        storage = components.build_storage(config, run_dir=tmp_path)

        # Use dummy abstraction since we don't have real data
        card_abs = DummyCardAbstraction()

        solver = components.build_solver(config, action_abs, card_abs, storage)

        # Verify everything is connected
        assert isinstance(action_abs, BettingActions)
        assert isinstance(storage, InMemoryStorage)
        assert isinstance(solver, MCCFRSolver)
        assert solver.storage == storage
        assert solver.action_abstraction == action_abs

    def test_build_with_disk_storage_creates_directory(self, tmp_path):
        """Test that disk storage properly initializes directory structure."""
        config = Config.default()
        config.set("storage.backend", "disk")

        run_dir = tmp_path / "test_run"
        storage = components.build_storage(config, run_dir=run_dir)

        assert isinstance(storage, DiskBackedStorage)
        # Directory should be created
        assert run_dir.exists()
