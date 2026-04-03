"""The interactive bucket-lookup flow.

It is a REPL over typed cards, so the behaviours that matter are the exits: a
cancelled prompt returns to the street menu, a failed load returns to the menu,
and a mistyped card costs one line rather than the session.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.game.state import Card, Street
from src.interfaces.cli.flows.combo_precompute import lookup
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


class _Abstraction:
    """Buckets a hand by a caller-supplied rule.

    Signatures mirror `BucketingStrategy` exactly, parameter names included:
    structural conformance is by name as well as type, so a double spelling
    `street` as `_street` is not the protocol and `ty` says so.
    """

    def __init__(self, bucket_of: Callable[..., int] = lambda hole, board, street: 7) -> None:
        self._bucket_of = bucket_of
        self.calls: list[tuple] = []

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        self.calls.append((hole_cards, board, street))
        return self._bucket_of(hole_cards, board, street)

    def num_buckets(self, street: Street) -> int:
        return 50


@pytest.fixture
def entry(tmp_path: Path) -> AbstractionEntry:
    path = tmp_path / "buckets-a"
    path.mkdir()
    return AbstractionEntry(path=path, metadata={})


class TestLoadAbstraction:
    def test_a_loaded_abstraction_is_returned(self, monkeypatch, tmp_path):
        loaded = object()
        monkeypatch.setattr(lookup.PostflopPrecomputer, "load", staticmethod(lambda _p: loaded))

        assert lookup._load_abstraction(tmp_path / "buckets-a") is loaded

    def test_a_failed_load_is_reported_and_answers_none(self, monkeypatch, tmp_path, capsys):
        def _boom(_path):
            raise OSError("pickle truncated")

        monkeypatch.setattr(lookup.PostflopPrecomputer, "load", staticmethod(_boom))

        assert lookup._load_abstraction(tmp_path / "buckets-a") is None
        assert "Failed to load: pickle truncated" in capsys.readouterr().out


class TestLookupOnce:
    def test_it_reports_the_bucket_and_the_range(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(lookup.prompts, "text", lambda _c, message, default="": default)
        abstraction = _Abstraction()

        lookup._lookup_once(_make_ctx(tmp_path), abstraction, Street.FLOP)

        out = capsys.readouterr().out
        assert "Bucket: 7" in out
        assert "out of 50 buckets on FLOP" in out

    def test_the_board_example_matches_the_street(self, tmp_path, monkeypatch):
        """A turn prompt must not suggest a three-card board."""
        asked: list[str] = []

        def _text(_ctx, message, default=""):
            asked.append(default)
            return default

        monkeypatch.setattr(lookup.prompts, "text", _text)
        lookup._lookup_once(_make_ctx(tmp_path), _Abstraction(), Street.TURN)

        assert lookup.BOARD_EXAMPLES[Street.TURN] in asked

    def test_cancelling_the_hole_prompt_asks_nothing_further(self, tmp_path, monkeypatch):
        abstraction = _Abstraction()
        monkeypatch.setattr(lookup.prompts, "text", lambda *_a, **_k: None)

        lookup._lookup_once(_make_ctx(tmp_path), abstraction, Street.FLOP)

        assert abstraction.calls == []

    def test_cancelling_the_board_prompt_looks_nothing_up(self, tmp_path, monkeypatch):
        abstraction = _Abstraction()
        answers = iter(["AsKh", None])
        monkeypatch.setattr(lookup.prompts, "text", lambda *_a, **_k: next(answers))

        lookup._lookup_once(_make_ctx(tmp_path), abstraction, Street.FLOP)

        assert abstraction.calls == []

    def test_a_mistyped_card_is_reported_not_raised(self, tmp_path, monkeypatch, capsys):
        answers = iter(["not-a-hand", "QsJhTc"])
        monkeypatch.setattr(lookup.prompts, "text", lambda *_a, **_k: next(answers))

        lookup._lookup_once(_make_ctx(tmp_path), _Abstraction(), Street.FLOP)

        assert "✗ Error:" in capsys.readouterr().out

    def test_an_isomorphic_board_landing_in_the_same_bucket_is_confirmed(
        self, tmp_path, monkeypatch, capsys
    ):
        answers = iter(["AsKh", "QsJhTc", "QhJdTs"])
        monkeypatch.setattr(lookup.prompts, "text", lambda *_a, **_k: next(answers))

        lookup._lookup_once(_make_ctx(tmp_path), _Abstraction(), Street.FLOP)

        assert "same bucket: 7" in capsys.readouterr().out

    def test_a_differing_isomorphic_board_is_flagged(self, tmp_path, monkeypatch, capsys):
        answers = iter(["AsKh", "QsJhTc", "QhJdTs"])
        monkeypatch.setattr(lookup.prompts, "text", lambda *_a, **_k: next(answers))
        seen: list[int] = []

        def _bucket(hole, board, street) -> int:
            seen.append(len(seen))
            return len(seen)

        lookup._lookup_once(_make_ctx(tmp_path), _Abstraction(_bucket), Street.FLOP)

        assert "Different bucket" in capsys.readouterr().out

    def test_an_empty_isomorphic_answer_skips_the_check(self, tmp_path, monkeypatch, capsys):
        answers = iter(["AsKh", "QsJhTc", ""])
        monkeypatch.setattr(lookup.prompts, "text", lambda *_a, **_k: next(answers))
        abstraction = _Abstraction()

        lookup._lookup_once(_make_ctx(tmp_path), abstraction, Street.FLOP)

        assert len(abstraction.calls) == 1
        assert "Isomorphic" not in capsys.readouterr().out


class TestHandleComboTestLookup:
    def test_no_abstraction_selected_returns_without_loading(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lookup, "_select_abstraction", lambda _ctx: None)
        monkeypatch.setattr(lookup, "_load_abstraction", lambda _p: pytest.fail("must not load"))

        lookup.handle_combo_test_lookup(_make_ctx(tmp_path))

    def test_a_failed_load_returns_without_prompting(self, tmp_path, monkeypatch, entry):
        monkeypatch.setattr(lookup, "_select_abstraction", lambda _ctx: entry)
        monkeypatch.setattr(lookup, "_load_abstraction", lambda _p: None)
        monkeypatch.setattr(
            lookup.prompts, "select", lambda *_a, **_k: pytest.fail("must not prompt")
        )

        lookup.handle_combo_test_lookup(_make_ctx(tmp_path))

    def test_back_leaves_the_loop(self, tmp_path, monkeypatch, entry):
        monkeypatch.setattr(lookup, "_select_abstraction", lambda _ctx: entry)
        monkeypatch.setattr(lookup, "_load_abstraction", lambda _p: _Abstraction())
        monkeypatch.setattr(lookup.prompts, "select", lambda *_a, **_k: "Back")
        monkeypatch.setattr(
            lookup, "_lookup_once", lambda *_a: pytest.fail("Back must not look up")
        )

        lookup.handle_combo_test_lookup(_make_ctx(tmp_path))

    def test_a_street_is_looked_up_then_back_exits(self, tmp_path, monkeypatch, entry):
        monkeypatch.setattr(lookup, "_select_abstraction", lambda _ctx: entry)
        monkeypatch.setattr(lookup, "_load_abstraction", lambda _p: _Abstraction())
        choices = iter(["RIVER", "Back"])
        monkeypatch.setattr(lookup.prompts, "select", lambda *_a, **_k: next(choices))
        streets: list[Street] = []
        monkeypatch.setattr(lookup, "_lookup_once", lambda _c, _a, street: streets.append(street))

        lookup.handle_combo_test_lookup(_make_ctx(tmp_path))

        assert streets == [Street.RIVER]

    def test_cancelling_the_street_prompt_also_exits(self, tmp_path, monkeypatch, entry):
        monkeypatch.setattr(lookup, "_select_abstraction", lambda _ctx: entry)
        monkeypatch.setattr(lookup, "_load_abstraction", lambda _p: _Abstraction())
        monkeypatch.setattr(lookup.prompts, "select", lambda *_a, **_k: None)
        monkeypatch.setattr(lookup, "_lookup_once", lambda *_a: pytest.fail("must not look up"))

        lookup.handle_combo_test_lookup(_make_ctx(tmp_path))
