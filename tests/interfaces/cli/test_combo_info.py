"""The abstraction-info screen.

Its job is to summarise whatever is on disk, including abstractions written
before the current metadata schema -- so the load-bearing property is that a
partial or drifted record still renders rather than blanking the panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.game.state import Street
from src.interfaces.cli.flows.combo_precompute import info
from src.interfaces.cli.flows.combo_precompute.common import AbstractionEntry
from src.interfaces.cli.ui.context import CliContext


def _make_ctx(tmp_path: Path) -> CliContext:
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def _write(tmp_path: Path, name: str, metadata: dict) -> Path:
    path = tmp_path / "data" / "combo_abstraction" / name
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


class TestParseMetadataConfig:
    def test_a_non_dict_config_answers_none(self):
        assert info._parse_metadata_config({"config": "quick"}) is None

    def test_a_missing_config_answers_none(self):
        assert info._parse_metadata_config({}) is None

    def test_a_drifted_schema_answers_none_rather_than_raising(self):
        """Drift is expected: the resolver still matches such an abstraction by name."""
        assert info._parse_metadata_config({"config": {"unknown_field": 1}}) is None


class TestHandleComboInfo:
    def test_a_missing_directory_is_reported(self, tmp_path, capsys):
        info.handle_combo_info(_make_ctx(tmp_path))

        assert "No combo abstractions found" in capsys.readouterr().out

    def test_an_empty_directory_is_reported(self, tmp_path, capsys):
        (tmp_path / "data" / "combo_abstraction").mkdir(parents=True)

        info.handle_combo_info(_make_ctx(tmp_path))

        assert "No combo abstractions found" in capsys.readouterr().out

    def test_it_lists_each_abstraction_with_its_config(self, tmp_path, monkeypatch, capsys):
        _write(tmp_path, "buckets-a", {"config": {"config_name": "quick"}})
        _write(tmp_path, "buckets-b", {"config": {"config_name": "deep"}})
        monkeypatch.setattr(info.prompts, "confirm", lambda *_a, **_k: False)

        info.handle_combo_info(_make_ctx(tmp_path))

        out = capsys.readouterr().out
        assert "Found 2 combo abstraction(s)" in out
        assert "buckets-a" in out
        assert "buckets-b" in out

    def test_street_statistics_are_rendered_when_present(self, tmp_path, monkeypatch, capsys):
        _write(
            tmp_path,
            "buckets-a",
            {
                "config": {"config_name": "quick"},
                "streets": {
                    "FLOP": {
                        "num_buckets": 10,
                        "num_boards": 22,
                        "quality": {"combo_count": 1234, "variance_explained": 0.96},
                    }
                },
            },
        )
        monkeypatch.setattr(info.prompts, "confirm", lambda *_a, **_k: False)

        info.handle_combo_info(_make_ctx(tmp_path))

        out = capsys.readouterr().out
        assert "Street statistics" in out
        assert "var expl 0.9600" in out

    def test_an_abstraction_with_no_streets_still_renders(self, tmp_path, monkeypatch, capsys):
        _write(tmp_path, "buckets-a", {"config": {"config_name": "quick"}})
        monkeypatch.setattr(info.prompts, "confirm", lambda *_a, **_k: False)

        info.handle_combo_info(_make_ctx(tmp_path))

        out = capsys.readouterr().out
        assert "buckets-a" in out
        assert "Street statistics" not in out

    def test_declining_the_detail_prompt_selects_nothing(self, tmp_path, monkeypatch):
        _write(tmp_path, "buckets-a", {"config": {"config_name": "quick"}})
        monkeypatch.setattr(info.prompts, "confirm", lambda *_a, **_k: False)
        monkeypatch.setattr(
            info.prompts, "select", lambda *_a, **_k: pytest.fail("must not select")
        )

        info.handle_combo_info(_make_ctx(tmp_path))


class TestShowDetailedInfo:
    def test_choosing_back_returns_without_loading(self, tmp_path, monkeypatch):
        """`Back` arrives as the string, not None -- an `is not None` test lets it through."""
        entry = AbstractionEntry(path=tmp_path / "buckets-a", metadata={})
        monkeypatch.setattr(info.prompts, "select", lambda _c, _m, choices: choices[-1].value)
        monkeypatch.setattr(
            info.PostflopPrecomputer,
            "load",
            staticmethod(lambda _p: pytest.fail("Back must not load")),
        )

        info._show_detailed_info(_make_ctx(tmp_path), [entry])

    def test_a_selected_abstraction_reports_its_distribution(self, tmp_path, monkeypatch, capsys):
        entry = AbstractionEntry(path=tmp_path / "buckets-a", metadata={})
        monkeypatch.setattr(info.prompts, "select", lambda _c, _m, choices: choices[0].value)
        monkeypatch.setattr(
            info.PostflopPrecomputer, "load", staticmethod(lambda _p: _Distribution())
        )

        info._show_detailed_info(_make_ctx(tmp_path), [entry])

        out = capsys.readouterr().out
        assert "DETAILED INFO: buckets-a" in out
        assert "Total combos: 60" in out
        assert "Unique buckets: 3" in out

    def test_a_street_with_no_distribution_is_skipped(self, tmp_path, monkeypatch, capsys):
        entry = AbstractionEntry(path=tmp_path / "buckets-a", metadata={})
        monkeypatch.setattr(info.prompts, "select", lambda _c, _m, choices: choices[0].value)
        monkeypatch.setattr(
            info.PostflopPrecomputer, "load", staticmethod(lambda _p: _Distribution(empty=True))
        )

        info._show_detailed_info(_make_ctx(tmp_path), [entry])

        assert "Total combos" not in capsys.readouterr().out


class _Distribution:
    """A bucketer reporting a fixed per-street bucket distribution."""

    def __init__(self, *, empty: bool = False) -> None:
        self._empty = empty

    def get_bucket_distribution(self, street: Street) -> dict[int, int]:
        assert street in (Street.FLOP, Street.TURN, Street.RIVER)
        return {} if self._empty else {0: 10, 1: 20, 2: 30}
