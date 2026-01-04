"""Tests for config handler."""

from unittest.mock import MagicMock

import pytest

from src.cli.flows.config import select_config
from src.cli.ui.context import CliContext


@pytest.fixture
def mock_style():
    """Mock questionary style."""
    return MagicMock()


def test_select_config_no_files(tmp_path, mock_style):
    """Test select_config when no config files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    ctx = CliContext(
        base_dir=empty_dir,
        config_dir=empty_dir,
        runs_dir=empty_dir / "runs",
        equity_buckets_dir=empty_dir / "equity_buckets",
        style=mock_style,
    )

    result = select_config(ctx)

    assert result is None
