"""Tests for config handler."""

from unittest.mock import MagicMock

import pytest

from src.cli.config_handler import select_config


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    test_config = config_dir / "test.yaml"
    test_config.write_text(
        """
training:
  num_iterations: 1000
  checkpoint_frequency: 100

card_abstraction:
  type: rank_based
"""
    )

    return config_dir


@pytest.fixture
def mock_style():
    """Mock questionary style."""
    return MagicMock()


def test_select_config_no_files(tmp_path, mock_style):
    """Test select_config when no config files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = select_config(empty_dir, mock_style)

    assert result is None
