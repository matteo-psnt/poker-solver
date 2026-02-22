"""Tests for training flow behavior in CLI."""

from unittest.mock import MagicMock

from src.interfaces.cli.flows import training
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def _make_ctx(tmp_path):
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def test_train_solver_syncs_runs_dir_from_relative_config(tmp_path, monkeypatch):
    """train_solver should align ctx.runs_dir with selected config training.runs_dir."""
    ctx = _make_ctx(tmp_path)
    config = Config.default().merge({"training": {"runs_dir": "custom_runs"}})

    monkeypatch.setattr(training, "select_config", lambda _ctx: config)
    monkeypatch.setattr(training, "_ensure_combo_abstraction", lambda _ctx, _config: True)
    monkeypatch.setattr(training, "_prompt_num_workers", lambda _ctx: 2)
    monkeypatch.setattr(training.ui, "header", lambda _title: None)
    monkeypatch.setattr(training.ui, "pause", lambda: None)

    start_calls = []

    def _mock_start(_config, _num_workers):
        start_calls.append((_config, _num_workers))
        return MagicMock()

    monkeypatch.setattr(training, "_start_training", _mock_start)

    training.train_solver(ctx)

    assert ctx.runs_dir == (tmp_path / "custom_runs").resolve()
    assert len(start_calls) == 1


def test_train_solver_syncs_runs_dir_from_absolute_config(tmp_path, monkeypatch):
    """Absolute runs_dir in config should be preserved."""
    ctx = _make_ctx(tmp_path)
    absolute_runs_dir = (tmp_path / "alt" / "runs").resolve()
    config = Config.default().merge({"training": {"runs_dir": str(absolute_runs_dir)}})

    monkeypatch.setattr(training, "select_config", lambda _ctx: config)
    monkeypatch.setattr(training, "_ensure_combo_abstraction", lambda _ctx, _config: True)
    monkeypatch.setattr(training, "_prompt_num_workers", lambda _ctx: 2)
    monkeypatch.setattr(training.ui, "header", lambda _title: None)
    monkeypatch.setattr(training.ui, "pause", lambda: None)
    monkeypatch.setattr(training, "_start_training", lambda _config, _num_workers: MagicMock())

    training.train_solver(ctx)

    assert ctx.runs_dir == absolute_runs_dir
