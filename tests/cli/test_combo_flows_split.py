"""Tests for split combo flow modules and compatibility facade."""

from unittest.mock import MagicMock

from src.cli.flows import combo
from src.cli.flows import combo_menu as combo_menu_module
from src.cli.ui.context import CliContext


def _make_ctx(tmp_path):
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def test_combo_facade_exports_split_handlers():
    assert combo.combo_menu is combo_menu_module.combo_menu
    assert callable(combo.handle_combo_precompute)
    assert callable(combo.handle_combo_info)
    assert callable(combo.handle_combo_test_lookup)
    assert callable(combo.handle_combo_coverage)
    assert callable(combo.handle_combo_analyze_bucketing)


def test_combo_menu_builds_expected_entries(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    captured = {}

    def _fake_run_menu(_ctx, prompt, items, exit_label):
        captured["prompt"] = prompt
        captured["items"] = [item.label for item in items]
        captured["exit_label"] = exit_label

    monkeypatch.setattr(combo_menu_module, "run_menu", _fake_run_menu)

    combo_menu_module.combo_menu(ctx)

    assert captured["prompt"] == "Combo Abstraction Tools:"
    assert captured["exit_label"] == "Back"
    assert captured["items"] == [
        "Precompute Abstraction",
        "View Abstraction Info",
        "Test Bucket Lookup",
        "Analyze Bucketing Patterns",
        "Analyze Coverage (Fallback Rate)",
    ]
