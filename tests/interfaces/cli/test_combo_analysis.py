"""The bucketing-analysis flow.

Three read-only analyses behind two prompts. What is worth pinning is that each
analysis actually asks the abstraction about the street it was given, and that
the dispatch routes to the one that was chosen -- the failure mode here is a
screen that quietly reports on the wrong street or the wrong analysis.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.game.state import Card, Street
from src.interfaces.cli.flows.combo_precompute import analysis
from src.interfaces.cli.flows.combo_precompute.common import AbstractionEntry
from src.interfaces.cli.ui.context import CliContext

PREMIUM = "Premium vs Weak Hands (predefined scenarios)"
RANDOM = "Random Sample (50 random hand/board combos)"
CORRELATION = "Hand Strength Correlation (various equities)"


def _make_ctx(tmp_path: Path) -> CliContext:
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


class _Abstraction:
    """Records every question asked of it; answers a rotating bucket."""

    def __init__(self, *, raises: bool = False) -> None:
        self.streets: list[Street] = []
        self.boards: list[tuple] = []
        self._raises = raises
        self._next = 0

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        if self._raises:
            raise ValueError("no such board")
        self.streets.append(street)
        self.boards.append(board)
        self._next += 1
        return self._next % 7

    def num_buckets(self, street: Street) -> int:
        return 20


@pytest.fixture
def entry(tmp_path: Path) -> AbstractionEntry:
    path = tmp_path / "buckets-a"
    path.mkdir()
    return AbstractionEntry(path=path, metadata={})


class TestPremiumVsWeak:
    @pytest.mark.parametrize(
        ("street", "board_len"),
        [
            pytest.param(Street.FLOP, 3, id="flop-three-card-boards"),
            pytest.param(Street.TURN, 4, id="turn-four-card-boards"),
            pytest.param(Street.RIVER, 5, id="river-five-card-boards"),
        ],
    )
    def test_every_scenario_board_matches_the_street(self, street, board_len, capsys):
        """A turn analysis dealing a three-card board would be measuring the flop."""
        abstraction = _Abstraction()

        analysis._analyze_premium_vs_weak(abstraction, street)

        assert abstraction.boards
        assert {len(b) for b in abstraction.boards} == {board_len}
        assert set(abstraction.streets) == {street}
        assert "Bucket range:" in capsys.readouterr().out

    def test_the_three_categories_are_reported(self, capsys):
        analysis._analyze_premium_vs_weak(_Abstraction(), Street.FLOP)

        out = capsys.readouterr().out
        assert "STRONG HANDS:" in out
        assert "WEAK HANDS:" in out

    def test_a_failing_scenario_is_reported_and_the_rest_continue(self, capsys):
        analysis._analyze_premium_vs_weak(_Abstraction(raises=True), Street.FLOP)

        out = capsys.readouterr().out
        assert "Error testing" in out
        assert "Bucket range:" not in out


class TestRandomSample:
    def test_it_asks_about_the_given_street(self, capsys):
        abstraction = _Abstraction()

        analysis._analyze_random_sample(abstraction, Street.TURN)

        assert set(abstraction.streets) == {Street.TURN}
        assert {len(b) for b in abstraction.boards} == {4}
        assert capsys.readouterr().out


class TestHandStrengthCorrelation:
    def test_it_asks_about_the_given_street(self, capsys):
        abstraction = _Abstraction()

        analysis._analyze_hand_strength_correlation(abstraction, Street.RIVER)

        assert set(abstraction.streets) == {Street.RIVER}
        assert capsys.readouterr().out


class TestDispatch:
    def _wire(self, monkeypatch, entry, *, street: str, choice: str):
        monkeypatch.setattr(analysis, "_select_abstraction", lambda _ctx: entry)
        monkeypatch.setattr(
            analysis.PostflopPrecomputer, "load", staticmethod(lambda _p: _Abstraction())
        )
        answers = iter([street, choice])
        monkeypatch.setattr(analysis.prompts, "select", lambda *_a, **_k: next(answers))
        called: list[str] = []
        for name in (
            "_analyze_premium_vs_weak",
            "_analyze_random_sample",
            "_analyze_hand_strength_correlation",
        ):
            monkeypatch.setattr(analysis, name, lambda _a, _s, _n=name: called.append(_n))
        return called

    def test_no_abstraction_selected_loads_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(analysis, "_select_abstraction", lambda _ctx: None)
        monkeypatch.setattr(
            analysis.PostflopPrecomputer,
            "load",
            staticmethod(lambda _p: pytest.fail("must not load")),
        )

        analysis.handle_combo_analyze_bucketing(_make_ctx(tmp_path))

    def test_a_failed_load_is_reported(self, tmp_path, monkeypatch, entry, capsys):
        monkeypatch.setattr(analysis, "_select_abstraction", lambda _ctx: entry)

        def _boom(_path):
            raise OSError("truncated")

        monkeypatch.setattr(analysis.PostflopPrecomputer, "load", staticmethod(_boom))

        analysis.handle_combo_analyze_bucketing(_make_ctx(tmp_path))

        assert "Failed to load: truncated" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            pytest.param(PREMIUM, "_analyze_premium_vs_weak", id="premium"),
            pytest.param(RANDOM, "_analyze_random_sample", id="random"),
            pytest.param(CORRELATION, "_analyze_hand_strength_correlation", id="correlation"),
        ],
    )
    def test_each_choice_routes_to_its_analysis(
        self, tmp_path, monkeypatch, entry, choice, expected
    ):
        called = self._wire(monkeypatch, entry, street="FLOP", choice=choice)

        analysis.handle_combo_analyze_bucketing(_make_ctx(tmp_path))

        assert called == [expected]

    def test_cancelling_the_street_runs_no_analysis(self, tmp_path, monkeypatch, entry):
        called = self._wire(monkeypatch, entry, street="Cancel", choice=PREMIUM)

        analysis.handle_combo_analyze_bucketing(_make_ctx(tmp_path))

        assert called == []

    def test_backing_out_of_the_analysis_menu_runs_nothing(self, tmp_path, monkeypatch, entry):
        called = self._wire(monkeypatch, entry, street="FLOP", choice="Back")

        analysis.handle_combo_analyze_bucketing(_make_ctx(tmp_path))

        assert called == []
