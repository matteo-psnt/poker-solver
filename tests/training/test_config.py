"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import Config, get_default_config, load_config


class TestConfig:
    """Tests for Config class."""

    def test_create_from_dict(self):
        config_dict = {"key": "value", "nested": {"inner": 42}}
        config = Config.from_dict(config_dict)

        assert config.get("key") == "value"
        assert config.get("nested.inner") == 42

    def test_get_with_dot_notation(self):
        config = Config.default()

        assert config.get("game.starting_stack") == 200
        assert config.get("game.small_blind") == 1
        assert config.get("training.num_iterations") == 1000

    def test_get_with_default(self):
        config = Config.default()

        assert config.get("nonexistent", "default") == "default"
        assert config.get("game.nonexistent", 99) == 99

    def test_get_section(self):
        config = Config.default()

        game_section = config.get_section("game")
        assert game_section["starting_stack"] == 200
        assert game_section["big_blind"] == 2

    def test_set_value(self):
        config = Config.default()

        config.set("game.starting_stack", 100)
        assert config.get("game.starting_stack") == 100

    def test_set_nested_value(self):
        config = Config.default()

        config.set("new.nested.value", 42)
        assert config.get("new.nested.value") == 42

    def test_merge_configs(self):
        base = Config.from_dict({"a": 1, "b": {"x": 10}})
        override = Config.from_dict({"b": {"y": 20}, "c": 3})

        base.merge(override)

        assert base.get("a") == 1  # Preserved
        assert base.get("b.x") == 10  # Preserved
        assert base.get("b.y") == 20  # Added
        assert base.get("c") == 3  # Added

    def test_to_dict(self):
        config = Config.default()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "game" in config_dict
        assert "training" in config_dict

    def test_validate_valid_config(self):
        config = Config.default()
        # Should not raise
        config.validate()

    def test_validate_invalid_starting_stack(self):
        config = Config.default()
        config.set("game.starting_stack", 0)

        with pytest.raises(ValueError, match="starting_stack must be positive"):
            config.validate()

    def test_validate_invalid_storage_backend(self):
        config = Config.default()
        config.set("storage.backend", "invalid")

        with pytest.raises(ValueError, match="storage backend must be"):
            config.validate()

    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"game": {"starting_stack": 150}}, f)
            config_path = Path(f.name)

        try:
            config = Config.from_file(config_path)
            assert config.get("game.starting_stack") == 150
        finally:
            config_path.unlink()

    def test_str_representation(self):
        config = Config.default()
        s = str(config)
        assert "Config" in s


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self):
        config = load_config()

        assert config.get("game.starting_stack") == 200
        assert config.get("storage.backend") == "disk"

    def test_load_with_overrides(self):
        config = load_config(
            training__num_iterations=5000,
            storage__backend="disk",
        )

        assert config.get("training.num_iterations") == 5000
        assert config.get("storage.backend") == "disk"

    def test_load_from_file_with_overrides(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "game": {"starting_stack": 150, "small_blind": 1, "big_blind": 2},
                "training": {"num_iterations": 100, "checkpoint_frequency": 10},
                "storage": {"backend": "memory"}
            }, f)
            config_path = Path(f.name)

        try:
            config = load_config(config_path, game__starting_stack=250)
            assert config.get("game.starting_stack") == 250  # Override applied
        finally:
            config_path.unlink()


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_dict(self):
        config_dict = get_default_config()
        assert isinstance(config_dict, dict)

    def test_has_required_sections(self):
        config_dict = get_default_config()

        assert "game" in config_dict
        assert "action_abstraction" in config_dict
        assert "card_abstraction" in config_dict
        assert "solver" in config_dict
        assert "training" in config_dict
        assert "storage" in config_dict
        assert "system" in config_dict

    def test_game_config_values(self):
        config_dict = get_default_config()
        game = config_dict["game"]

        assert game["starting_stack"] == 200
        assert game["small_blind"] == 1
        assert game["big_blind"] == 2
