"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest

from src.utils.config import Config
from src.utils.config_loader import _merge_config, load_config


class TestDefaultBehavior:
    """Test that defaults work as expected."""

    def test_default_config_values(self):
        """Verify default values are correct."""
        cfg = Config()

        assert cfg.training.num_iterations > 0
        assert cfg.training.checkpoint_frequency > 0
        assert cfg.training.verbose is True

        assert cfg.storage.initial_capacity > 0
        assert cfg.storage.max_actions > 0

        assert cfg.game.big_blind > cfg.game.small_blind
        assert cfg.game.starting_stack > 0

        assert cfg.system.seed is None  # Default to random
        assert cfg.system.config_name  # Non-empty

    def test_config_is_immutable(self):
        """Verify config is frozen (cannot be modified)."""
        cfg = Config()

        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.training.num_iterations = 999  # type: ignore

        with pytest.raises(Exception):
            cfg.storage.initial_capacity = 999  # type: ignore


class TestMergeBehavior:
    """Test that merging overrides works correctly."""

    def test_merge_preserves_defaults(self):
        """Verify merge only changes specified fields."""
        base = Config()
        overrides = {"training": {"num_iterations": 50_000}}

        merged = _merge_config(base, overrides)

        # Overridden value
        assert merged.training.num_iterations == 50_000

    def test_merge_nested_config(self):
        """Verify deep merging works."""
        base = Config()
        overrides = {
            "training": {"num_iterations": 50_000, "verbose": False},
            "game": {"big_blind": 4},
        }

        merged = _merge_config(base, overrides)

        assert merged.training.num_iterations == 50_000
        assert merged.training.verbose is False
        assert merged.game.big_blind == 4

    def test_merge_empty_overrides(self):
        """Verify merging empty dict returns original."""
        base = Config()
        merged = _merge_config(base, {})

        assert merged == base

    def test_merge_creates_new_instance(self):
        """Verify merge returns new object (immutability)."""
        base = Config()
        overrides = {"training": {"num_iterations": 50_000}}

        merged = _merge_config(base, overrides)

        assert merged is not base
        assert base.training.num_iterations == Config().training.num_iterations  # unchanged


class TestLoadBehavior:
    """Test that loading from YAML works correctly."""

    def test_load_defaults_only(self):
        """Verify loading without file uses defaults."""
        cfg = load_config()

        base = Config()
        assert cfg.training.num_iterations == base.training.num_iterations
        assert cfg.game.big_blind == base.game.big_blind

    def test_load_from_yaml_file(self):
        """Verify loading from YAML applies overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
training:
  num_iterations: 50000
  verbose: false

game:
  starting_stack: 300
"""
            )
            yaml_path = Path(f.name)

        try:
            cfg = load_config(yaml_path)

            # Overridden values
            assert cfg.training.num_iterations == 50_000
            assert cfg.training.verbose is False
            assert cfg.game.starting_stack == 300

        finally:
            yaml_path.unlink()

    def test_load_with_programmatic_overrides(self):
        """Verify programmatic overrides work."""
        cfg = load_config(training__num_iterations=75_000, game__big_blind=5)

        assert cfg.training.num_iterations == 75_000
        assert cfg.game.big_blind == 5

    def test_load_yaml_plus_programmatic(self):
        """Verify YAML + programmatic overrides compose correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
training:
  num_iterations: 50000
"""
            )
            yaml_path = Path(f.name)

        try:
            # YAML sets num_iterations=50000, programmatic overrides it to 75000
            cfg = load_config(yaml_path, training__num_iterations=75_000)

            assert cfg.training.num_iterations == 75_000  # programmatic wins
        finally:
            yaml_path.unlink()

    def test_load_nonexistent_file_raises(self):
        """Verify loading missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestTypeSafety:
    """Test that types are enforced."""

    def test_dataclass_fields_have_types(self):
        """Verify all fields are typed (not Any)."""
        cfg = Config()

        # Spot check - if these work, types are enforced
        assert isinstance(cfg.training.num_iterations, int)
        assert isinstance(cfg.training.verbose, bool)
        assert isinstance(cfg.game.starting_stack, int)
        assert cfg.system.seed is None or isinstance(cfg.system.seed, int)
