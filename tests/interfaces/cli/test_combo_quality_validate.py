"""The quality and validation flows.

Both are read-only screens over an abstraction's own metadata, so what matters
is that they degrade rather than crash: a cancelled prompt, an abstraction
predating the quality statistics, a street the metadata never recorded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.interfaces.cli.flows.combo_precompute import quality, validate
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


def _entry(tmp_path: Path, metadata: dict) -> AbstractionEntry:
    path = tmp_path / "buckets-a"
    path.mkdir(exist_ok=True)
    return AbstractionEntry(path=path, metadata=metadata)


def _street_stats(**quality_overrides) -> dict:
    return {
        "num_buckets": 10,
        "num_boards": 22,
        "quality": {
            "combo_count": 1234,
            "equity_std": 0.25,
            "within_bucket_std": 0.05,
            "variance_explained": 0.96,
            "bucket_combos_min": 3,
            "bucket_combos_median": 12.0,
            "bucket_combos_max": 40,
            "occupied_buckets": 9,
            "num_buckets": 10,
            **quality_overrides,
        },
    }


class TestQuality:
    def test_cancelling_prints_no_table(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(quality, "_select_abstraction", lambda _ctx: None)

        quality.handle_combo_quality(_make_ctx(tmp_path))

        assert "street" not in capsys.readouterr().out

    def test_metadata_without_streets_says_so(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(quality, "_select_abstraction", lambda _ctx: _entry(tmp_path, {}))

        quality.handle_combo_quality(_make_ctx(tmp_path))

        assert "No quality statistics in metadata" in capsys.readouterr().out

    def test_it_reports_every_street_present(self, tmp_path, monkeypatch, capsys):
        metadata = {"streets": {name: _street_stats() for name in ("FLOP", "TURN", "RIVER")}}
        monkeypatch.setattr(quality, "_select_abstraction", lambda _ctx: _entry(tmp_path, metadata))

        quality.handle_combo_quality(_make_ctx(tmp_path))

        out = capsys.readouterr().out
        for name in ("FLOP", "TURN", "RIVER"):
            assert name in out
        assert "0.9600" in out
        assert "occupied 9/10" in out

    def test_a_missing_street_is_skipped_not_defaulted(self, tmp_path, monkeypatch, capsys):
        """An abstraction that never recorded RIVER must not print a zeroed row."""
        metadata = {"streets": {"FLOP": _street_stats()}}
        monkeypatch.setattr(quality, "_select_abstraction", lambda _ctx: _entry(tmp_path, metadata))

        quality.handle_combo_quality(_make_ctx(tmp_path))

        out = capsys.readouterr().out
        assert "FLOP" in out
        assert "RIVER  " not in out


class TestValidate:
    def test_cancelling_the_selection_runs_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validate, "_select_abstraction", lambda _ctx: None)
        monkeypatch.setattr(
            validate.prompts, "confirm", lambda *_a, **_k: pytest.fail("must not confirm")
        )

        validate.handle_combo_validate(_make_ctx(tmp_path))

    def test_declining_the_confirm_loads_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validate, "_select_abstraction", lambda _ctx: _entry(tmp_path, {}))
        monkeypatch.setattr(validate.prompts, "confirm", lambda *_a, **_k: False)
        monkeypatch.setattr(
            validate.PostflopPrecomputer,
            "load",
            staticmethod(lambda _p: pytest.fail("must not load")),
        )

        validate.handle_combo_validate(_make_ctx(tmp_path))

    def test_a_failed_load_is_reported(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(validate, "_select_abstraction", lambda _ctx: _entry(tmp_path, {}))
        monkeypatch.setattr(validate.prompts, "confirm", lambda *_a, **_k: True)

        def _boom(_path):
            raise OSError("truncated")

        monkeypatch.setattr(validate.PostflopPrecomputer, "load", staticmethod(_boom))

        validate.handle_combo_validate(_make_ctx(tmp_path))

        assert "Error loading abstraction: truncated" in capsys.readouterr().out

    def test_a_healthy_abstraction_validates(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(validate, "_select_abstraction", lambda _ctx: _entry(tmp_path, {}))
        monkeypatch.setattr(validate.prompts, "confirm", lambda *_a, **_k: True)
        monkeypatch.setattr(
            validate.PostflopPrecomputer, "load", staticmethod(lambda _p: _Bucketer())
        )

        validate.handle_combo_validate(_make_ctx(tmp_path))

        assert "0" in capsys.readouterr().out

    def test_an_out_of_range_bucket_is_counted_as_a_failure(self, tmp_path, monkeypatch, capsys):
        """The check that gives the flow its purpose: a bucket outside 0..n-1."""
        monkeypatch.setattr(validate, "_select_abstraction", lambda _ctx: _entry(tmp_path, {}))
        monkeypatch.setattr(validate.prompts, "confirm", lambda *_a, **_k: True)
        monkeypatch.setattr(
            validate.PostflopPrecomputer,
            "load",
            staticmethod(lambda _p: _Bucketer(bucket=999)),
        )

        validate.handle_combo_validate(_make_ctx(tmp_path))

        assert "DEBUG" in capsys.readouterr().out


class _Bucketer:
    """A bucketer that answers a fixed bucket for every hand."""

    def __init__(self, bucket: int = 3) -> None:
        self._bucket = bucket

    def get_bucket(self, _hole, _board, _street) -> int:
        return self._bucket

    def num_buckets(self, _street) -> int:
        return 10
