"""Tests for PrecomputeConfig YAML loading."""

from pathlib import Path

import pytest
import yaml

from src.bucketing.postflop import PrecomputeConfig
from src.game.state import Street


def _load_yaml_config(name: str) -> dict:
    config_dir = Path(__file__).parent.parent.parent / "config" / "abstraction"
    with open(config_dir / f"{name}.yaml") as f:
        return yaml.safe_load(f)


class TestPrecomputeConfig:
    """Test configuration loading from YAML files."""

    def test_load_fast_test(self):
        """Test loading fast_test config."""
        config = PrecomputeConfig.from_yaml("fast_test")
        data = _load_yaml_config("fast_test")

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

    def test_default_classmethod(self):
        """Test default() classmethod loads default.yaml."""
        assert PrecomputeConfig.default() == PrecomputeConfig.from_yaml("default")

    def test_fast_test_classmethod(self):
        """Test fast_test() classmethod loads fast_test.yaml."""
        assert PrecomputeConfig.fast_test() == PrecomputeConfig.from_yaml("fast_test")

    def test_invalid_config_name(self):
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            PrecomputeConfig.from_yaml("nonexistent_config")

    def test_config_files_exist(self):
        """Test that all expected config files exist."""
        config_dir = Path(__file__).parent.parent.parent / "config" / "abstraction"

        assert (config_dir / "fast_test.yaml").exists()
        assert (config_dir / "default.yaml").exists()
        assert (config_dir / "production.yaml").exists()
