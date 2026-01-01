"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config import Config, GameConfig, SolverConfig, SystemConfig
from src.utils.config_loader import load_config


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

        with pytest.raises(ValidationError):
            cfg.training.num_iterations = 999

        with pytest.raises(ValidationError):
            cfg.storage.initial_capacity = 999


class TestMergeBehavior:
    """Test that merging overrides works correctly."""

    def test_merge_preserves_defaults(self):
        """Verify merge only changes specified fields."""
        base = Config()
        overrides = {"training": {"num_iterations": 50_000}}

        merged = base.merge(overrides)

        # Overridden value
        assert merged.training.num_iterations == 50_000

    def test_merge_nested_config(self):
        """Verify deep merging works."""
        base = Config()
        overrides = {
            "training": {"num_iterations": 50_000, "verbose": False},
            "game": {"big_blind": 4},
        }

        merged = base.merge(overrides)

        assert merged.training.num_iterations == 50_000
        assert merged.training.verbose is False
        assert merged.game.big_blind == 4

    def test_merge_empty_overrides(self):
        """Verify merging empty dict returns equivalent config."""
        base = Config()
        merged = base.merge({})

        assert merged == base

    def test_merge_creates_new_instance(self):
        """Verify merge returns new object (immutability)."""
        base = Config()
        overrides = {"training": {"num_iterations": 50_000}}

        merged = base.merge(overrides)

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

    def test_load_legacy_action_abstraction_raises(self):
        """Strict schema should reject removed action_abstraction key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
action_abstraction:
  preflop_raises: [2.5, 3.5]
"""
            )
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValidationError):
                load_config(yaml_path)
        finally:
            yaml_path.unlink()

    def test_load_extends_chain(self):
        """Verify YAML extends inheritance: child values override base, rest inherited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.yaml"
            child_path = Path(tmpdir) / "child.yaml"

            base_path.write_text(
                """
training:
  num_iterations: 1000
  verbose: false
solver:
  cfr_plus: true
"""
            )
            child_path.write_text(
                """
extends: base.yaml
training:
  num_iterations: 9999
"""
            )

            cfg = load_config(child_path)

            assert cfg.training.num_iterations == 9999  # child wins
            assert cfg.training.verbose is False  # inherited from base
            assert cfg.solver.cfr_plus is True  # inherited from base


class TestTypeSafety:
    """Test that types are enforced."""

    def test_dataclass_fields_have_types(self):
        """Verify all fields are typed correctly."""
        cfg = Config()

        assert isinstance(cfg.training.num_iterations, int)
        assert isinstance(cfg.training.verbose, bool)
        assert isinstance(cfg.game.starting_stack, int)
        assert cfg.system.seed is None or isinstance(cfg.system.seed, int)


class TestValidation:
    """Test that field-level and cross-field validation fires correctly."""

    def test_dcfr_alpha_must_be_positive(self):
        """dcfr_alpha <= 0 should raise ValidationError."""
        with pytest.raises(ValidationError, match="greater than 0"):
            SolverConfig(dcfr_alpha=-1.0)

    def test_dcfr_alpha_zero_invalid(self):
        """dcfr_alpha = 0 should raise ValidationError (must be > 0)."""
        with pytest.raises(ValidationError):
            SolverConfig(dcfr_alpha=0.0)

    def test_sampling_method_literal(self):
        """Invalid sampling_method value should raise ValidationError."""
        with pytest.raises(ValidationError):
            SolverConfig(sampling_method="bad_method")  # type: ignore

    def test_sampling_method_valid_values(self):
        """Both valid sampling methods should be accepted."""
        SolverConfig(sampling_method="external")
        SolverConfig(sampling_method="outcome")

    def test_big_blind_must_exceed_small_blind(self):
        """big_blind <= small_blind should raise ValidationError."""
        with pytest.raises(ValidationError, match="big_blind"):
            GameConfig(small_blind=5, big_blind=5)

        with pytest.raises(ValidationError, match="big_blind"):
            GameConfig(small_blind=5, big_blind=3)

    def test_log_level_literal(self):
        """Invalid log_level should raise ValidationError."""
        with pytest.raises(ValidationError):
            SystemConfig(log_level="VERBOSE")  # type: ignore

    def test_log_level_valid_values(self):
        """All valid log levels should be accepted."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            SystemConfig(log_level=level)

    def test_extra_keys_rejected(self):
        """Unknown keys in dict should raise ValidationError."""
        with pytest.raises(ValidationError):
            Config.model_validate({"unknown_section": {"foo": 1}})

    def test_unknown_yaml_key_raises_on_load(self):
        """Unknown top-level key in YAML raises ValidationError at load time."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("completely_unknown_key: 42\n")
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValidationError):
                load_config(yaml_path)
        finally:
            yaml_path.unlink()

    def test_zarr_compression_level_bounds(self):
        """zarr_compression_level must be between 1 and 9 inclusive."""
        from src.utils.config import StorageConfig

        with pytest.raises(ValidationError):
            StorageConfig(zarr_compression_level=0)

        with pytest.raises(ValidationError):
            StorageConfig(zarr_compression_level=10)

        # Boundary values are valid
        StorageConfig(zarr_compression_level=1)
        StorageConfig(zarr_compression_level=9)
