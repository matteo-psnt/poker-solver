"""Tests for config handler."""

from unittest.mock import MagicMock

import pytest

from src.interfaces.cli.flows import config as config_flow
from src.interfaces.cli.flows.config import select_config
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


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


def test_edit_card_abstraction_uses_dynamic_yaml_choices(tmp_path, monkeypatch, mock_style):
    """Card abstraction choices should come from config/abstraction/*.yaml files."""
    config_dir = tmp_path / "config"
    abstraction_dir = config_dir / "abstraction"
    abstraction_dir.mkdir(parents=True)
    (abstraction_dir / "zzz.yaml").write_text("seed: 42\n")
    (abstraction_dir / "alpha.yaml").write_text("seed: 42\n")

    ctx = CliContext(
        base_dir=tmp_path,
        config_dir=config_dir,
        runs_dir=tmp_path / "runs",
        equity_buckets_dir=tmp_path / "equity_buckets",
        style=mock_style,
    )
    base_config = Config.default().merge({"card_abstraction": {"config": "default"}})

    captured: dict[str, object] = {}

    def _mock_select(_ctx, _message, choices, default=None):
        captured["choices"] = choices
        captured["default"] = default
        return "alpha"

    monkeypatch.setattr(config_flow.prompts, "select", _mock_select)
    monkeypatch.setattr(config_flow.prompts, "confirm", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(config_flow.ui, "warn", lambda *_args, **_kwargs: None)

    updated = config_flow._edit_card_abstraction(ctx, base_config)

    assert captured["choices"] == ["alpha", "zzz"]
    assert captured["default"] == "alpha"
    assert updated.card_abstraction.config == "alpha"


def test_edit_card_abstraction_no_configs_returns_original(tmp_path, monkeypatch, mock_style):
    """When no abstraction configs exist, editor should keep config unchanged."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    ctx = CliContext(
        base_dir=tmp_path,
        config_dir=config_dir,
        runs_dir=tmp_path / "runs",
        equity_buckets_dir=tmp_path / "equity_buckets",
        style=mock_style,
    )
    base_config = Config.default().merge({"card_abstraction": {"config": "default"}})

    errors: list[str] = []

    monkeypatch.setattr(
        config_flow.prompts,
        "select",
        lambda *_args, **_kwargs: pytest.fail("prompts.select should not be called"),
    )
    monkeypatch.setattr(config_flow.ui, "error", lambda message: errors.append(message))

    updated = config_flow._edit_card_abstraction(ctx, base_config)

    assert updated == base_config
    assert errors
