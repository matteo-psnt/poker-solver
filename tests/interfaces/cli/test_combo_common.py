"""The helpers every combo flow routes through.

Card parsing is the one that matters: it turns typed text into the cards a
bucket lookup is performed on, so a silent misparse would answer a question
about a different hand than the one asked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.interfaces.cli.flows.combo_precompute import common
from src.interfaces.cli.ui.context import CliContext


def _make_ctx(tmp_path: Path) -> CliContext:
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def _write_abstraction(base: Path, name: str, metadata: dict) -> Path:
    path = base / name
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


class TestParseCards:
    def test_a_plain_pair_of_cards(self):
        cards = common._parse_cards("AsKh", expected=2)
        assert [repr(c) for c in cards] == ["As", "Kh"]

    def test_separators_are_ignored(self):
        assert len(common._parse_cards("As, Kh", expected=2)) == 2

    def test_rank_is_upper_and_suit_is_lower(self):
        """`as` and `AS` must name the same card as `As`."""
        for spelling in ("as", "AS", "aS"):
            assert repr(common._parse_cards(spelling, expected=1)[0]) == "As"

    def test_a_five_card_board(self):
        assert len(common._parse_cards("QsJhTc9d2h", expected=5)) == 5

    def test_a_trailing_half_card_is_refused(self):
        with pytest.raises(ValueError, match="incomplete card"):
            common._parse_cards("AsK", expected=2)

    def test_the_wrong_count_is_refused(self):
        with pytest.raises(ValueError, match="Expected 2 cards, got 3"):
            common._parse_cards("AsKhQd", expected=2)

    def test_an_unknown_rank_is_refused(self):
        with pytest.raises((ValueError, KeyError)):
            common._parse_cards("Xs", expected=1)


class TestConfigNameFromMetadata:
    def test_it_reads_the_nested_name(self):
        assert (
            common._get_config_name_from_metadata({"config": {"config_name": "quick"}}) == "quick"
        )

    def test_a_missing_config_is_unknown(self):
        assert common._get_config_name_from_metadata({}) == "unknown"

    def test_a_non_dict_config_is_unknown(self):
        assert common._get_config_name_from_metadata({"config": "quick"}) == "unknown"

    def test_an_empty_name_is_unknown(self):
        assert common._get_config_name_from_metadata({"config": {"config_name": ""}}) == "unknown"


class TestListExistingAbstractions:
    def test_it_finds_directories_carrying_metadata(self, tmp_path):
        _write_abstraction(tmp_path, "buckets-a", {"config": {"config_name": "quick"}})
        _write_abstraction(tmp_path, "buckets-b", {"config": {"config_name": "deep"}})

        found = common._list_existing_abstractions(tmp_path)

        assert sorted(e.path.name for e in found) == ["buckets-a", "buckets-b"]

    def test_a_directory_without_metadata_is_skipped(self, tmp_path):
        _write_abstraction(tmp_path, "buckets-a", {"config": {"config_name": "quick"}})
        (tmp_path / "buckets-empty").mkdir()

        assert [e.path.name for e in common._list_existing_abstractions(tmp_path)] == ["buckets-a"]

    def test_loose_files_are_skipped(self, tmp_path):
        (tmp_path / "stray.json").write_text("{}")

        assert common._list_existing_abstractions(tmp_path) == []

    def test_the_label_carries_the_config_name(self, tmp_path):
        _write_abstraction(tmp_path, "buckets-a", {"config": {"config_name": "quick"}})

        (entry,) = common._list_existing_abstractions(tmp_path)

        assert entry.label == "buckets-a (quick)"


class TestSelectAbstraction:
    def test_a_missing_directory_is_reported_not_raised(self, tmp_path, capsys):
        assert common._select_abstraction(_make_ctx(tmp_path)) is None
        assert "No combo abstractions found" in capsys.readouterr().out

    def test_an_empty_directory_is_reported(self, tmp_path, capsys):
        (tmp_path / "data" / "combo_abstraction").mkdir(parents=True)

        assert common._select_abstraction(_make_ctx(tmp_path)) is None
        assert "No combo abstractions found" in capsys.readouterr().out

    def test_it_offers_each_abstraction_plus_cancel(self, tmp_path, monkeypatch):
        base = tmp_path / "data" / "combo_abstraction"
        _write_abstraction(base, "buckets-a", {"config": {"config_name": "quick"}})
        offered: list = []

        def _select(_ctx, _message, choices):
            offered.extend(choices)
            return choices[0].value

        monkeypatch.setattr(common.prompts, "select", _select)

        chosen = common._select_abstraction(_make_ctx(tmp_path))

        assert [c.title for c in offered] == ["buckets-a (quick)", "Cancel"]
        assert chosen is not None
        assert chosen.path.name == "buckets-a"

    def test_cancelling_answers_none(self, tmp_path, monkeypatch):
        base = tmp_path / "data" / "combo_abstraction"
        _write_abstraction(base, "buckets-a", {"config": {"config_name": "quick"}})
        monkeypatch.setattr(common.prompts, "select", lambda _c, _m, choices: choices[-1].value)

        assert common._select_abstraction(_make_ctx(tmp_path)) is None
