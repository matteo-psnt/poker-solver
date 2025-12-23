"""Tests for PrecomputeConfig YAML loading."""

from pathlib import Path

import pytest

from src.abstraction.isomorphism import PrecomputeConfig
from src.game.state import Street


class TestPrecomputeConfig:
    """Test configuration loading from YAML files."""

    def test_load_fast_test(self):
        """Test loading fast_test config."""
        config = PrecomputeConfig.from_yaml("fast_test")

        assert config.num_board_clusters[Street.FLOP] == 10
        assert config.num_board_clusters[Street.TURN] == 20
        assert config.num_board_clusters[Street.RIVER] == 30

        assert config.num_buckets[Street.FLOP] == 10
        assert config.num_buckets[Street.TURN] == 20
        assert config.num_buckets[Street.RIVER] == 30

        assert config.representatives_per_cluster == 1
        assert config.equity_samples == 100
        assert config.seed == 42

    def test_load_default(self):
        """Test loading default config."""
        config = PrecomputeConfig.from_yaml("default")

        assert config.num_board_clusters[Street.FLOP] == 50
        assert config.num_board_clusters[Street.TURN] == 100
        assert config.num_board_clusters[Street.RIVER] == 200

        assert config.num_buckets[Street.FLOP] == 50
        assert config.num_buckets[Street.TURN] == 100
        assert config.num_buckets[Street.RIVER] == 200

        assert config.representatives_per_cluster == 1
        assert config.equity_samples == 1000

    def test_load_production(self):
        """Test loading production config."""
        config = PrecomputeConfig.from_yaml("production")

        assert config.num_board_clusters[Street.FLOP] == 100
        assert config.num_board_clusters[Street.TURN] == 200
        assert config.num_board_clusters[Street.RIVER] == 400

        assert config.num_buckets[Street.FLOP] == 100
        assert config.num_buckets[Street.TURN] == 300
        assert config.num_buckets[Street.RIVER] == 600

        assert config.representatives_per_cluster == 2
        assert config.equity_samples == 2000

    def test_default_classmethod(self):
        """Test default() classmethod loads default.yaml."""
        config = PrecomputeConfig.default()

        assert config.num_board_clusters[Street.FLOP] == 50
        assert config.equity_samples == 1000

    def test_fast_test_classmethod(self):
        """Test fast_test() classmethod loads fast_test.yaml."""
        config = PrecomputeConfig.fast_test()

        assert config.num_board_clusters[Street.FLOP] == 10
        assert config.equity_samples == 100

    def test_invalid_config_name(self):
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            PrecomputeConfig.from_yaml("nonexistent_config")

    def test_load_from_path(self):
        """Test loading config from absolute path."""
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "abstraction" / "fast_test.yaml"
        )
        config = PrecomputeConfig.from_yaml(str(config_path))

        assert config.num_board_clusters[Street.FLOP] == 10
        assert config.equity_samples == 100
