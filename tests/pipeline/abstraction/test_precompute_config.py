"""Tests for PrecomputeConfig YAML loading."""

from pathlib import Path

import pytest
import yaml

from src.core.game.state import Street
from src.pipeline.abstraction.config import PrecomputeConfig


def _load_yaml_config(name: str) -> dict:
    config_dir = Path(__file__).resolve().parents[3] / "config" / "abstraction"
    with open(config_dir / f"{name}.yaml") as f:
        return yaml.safe_load(f)


class TestPrecomputeConfig:
    """Test configuration loading from YAML files."""

    def test_load_quick_test(self):
        """Test loading quick_test config."""
        config = PrecomputeConfig.from_yaml("quick_test")
        data = _load_yaml_config("quick_test")

        assert config.num_board_clusters[Street.FLOP] == data["board_clusters"]["flop"]
        assert config.num_board_clusters[Street.TURN] == data["board_clusters"]["turn"]
        assert config.num_board_clusters[Street.RIVER] == data["board_clusters"]["river"]

        assert config.num_buckets[Street.FLOP] == data["buckets"]["flop"]
        assert config.num_buckets[Street.TURN] == data["buckets"]["turn"]
        assert config.num_buckets[Street.RIVER] == data["buckets"]["river"]

        assert config.representatives_per_cluster == data["representatives_per_cluster"]
        assert config.equity_samples == data["equity_samples"]
        assert config.seed == data["seed"]

    def test_load_default(self):
        """Test loading default config."""
        config = PrecomputeConfig.from_yaml("default")
        data = _load_yaml_config("default")

        assert config.num_board_clusters[Street.FLOP] == data["board_clusters"]["flop"]
        assert config.num_board_clusters[Street.TURN] == data["board_clusters"]["turn"]
        assert config.num_board_clusters[Street.RIVER] == data["board_clusters"]["river"]

        assert config.num_buckets[Street.FLOP] == data["buckets"]["flop"]
        assert config.num_buckets[Street.TURN] == data["buckets"]["turn"]
        assert config.num_buckets[Street.RIVER] == data["buckets"]["river"]

    def test_load_production(self):
        """Test loading production config."""
        config = PrecomputeConfig.from_yaml("production")
        data = _load_yaml_config("production")

        assert config.num_board_clusters[Street.FLOP] == data["board_clusters"]["flop"]
        assert config.num_board_clusters[Street.TURN] == data["board_clusters"]["turn"]
        assert config.num_board_clusters[Street.RIVER] == data["board_clusters"]["river"]

        assert config.num_buckets[Street.FLOP] == data["buckets"]["flop"]
        assert config.num_buckets[Street.TURN] == data["buckets"]["turn"]
        assert config.num_buckets[Street.RIVER] == data["buckets"]["river"]

        assert config.representatives_per_cluster == data["representatives_per_cluster"]
        assert config.equity_samples == data["equity_samples"]

    def test_invalid_config_name(self):
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            PrecomputeConfig.from_yaml("nonexistent_config")

    def test_config_name_auto_set(self):
        """Test that config_name is automatically set from filename."""
        # Config name should be auto-set, not read from YAML
        production = PrecomputeConfig.from_yaml("production")
        assert production.config_name == "production"

        quick_test = PrecomputeConfig.from_yaml("quick_test")
        assert quick_test.config_name == "quick_test"

        default = PrecomputeConfig.from_yaml("default")
        assert default.config_name == "default"

    def test_config_files_exist(self):
        """Test that all expected config files exist."""
        config_dir = Path(__file__).resolve().parents[3] / "config" / "abstraction"

        assert (config_dir / "quick_test.yaml").exists()
        assert (config_dir / "default.yaml").exists()
        assert (config_dir / "production.yaml").exists()
