"""The `abstraction-coupling` command end to end, on a stub abstraction.

`test_coupling.py` pins the arithmetic. What is left to break is the bridge: the
real 1,326-hand enumeration, the streamed universe, and whether the payload the
dial produces is orientated the way the report reads it. A stub bucketer is what
makes that runnable on a laptop -- the real artifact lives on the share, and a
precompute is minutes.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from src.core.game.state import Street
from src.interfaces.commands import abstraction_coupling
from src.interfaces.errors import CommandError

if TYPE_CHECKING:
    from pathlib import Path

COUNTS = {Street.PREFLOP: 169, Street.FLOP: 8, Street.TURN: 12, Street.RIVER: 16}


class StubBucketer:
    """Buckets by a cheap function of the cards, so a universe builds in ms.

    Deliberately board-DEPENDENT: a bucket that ignored the board would make the
    coupling identically zero and the test would pass while measuring nothing.
    """

    def num_buckets(self, street: Street) -> int:
        return COUNTS[street]

    def get_bucket(self, hole_cards, board, street: Street) -> int:
        total = sum(card.rank_eval7() for card in hole_cards)
        total += sum(card.rank_eval7() * (index + 2) for index, card in enumerate(board))
        return total % COUNTS[street]


@pytest.fixture
def args(tmp_path: Path) -> argparse.Namespace:
    (tmp_path / "stub").mkdir()
    return argparse.Namespace(
        abstraction="stub",
        abstractions_dir=str(tmp_path),
        boards=10,
        classes="1,2,5",
        seed=7,
        progress_file="",
    )


@pytest.fixture(autouse=True)
def _stub_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        abstraction_coupling.DenseBucketer, "load", staticmethod(lambda _: StubBucketer())
    )


def test_it_prices_every_averaged_constant(args: argparse.Namespace) -> None:
    payload = abstraction_coupling.run(args)

    names = {gap.name for gap in payload.gaps}
    assert names == {
        "transition:PREFLOP->FLOP",
        "transition:FLOP->TURN",
        "transition:TURN->RIVER",
        "compatible:FLOP",
        "compatible:TURN",
        "compatible:RIVER",
    }
    assert {gap.kind for gap in payload.gaps} == {"coupling", "dispersion"}


def test_the_dial_is_orientated(args: argparse.Namespace) -> None:
    """C=1 is the shipped game, so it recovers nothing; more classes never lose.

    A transposed partition or an inverted residual reads as a plausible curve;
    monotonicity plus a pinned zero is what makes it not.
    """
    payload = abstraction_coupling.run(args)

    for gap in payload.gaps:
        assert gap.recovered[1] == pytest.approx(0.0, abs=1e-9), gap.name
        assert gap.recovered[2] <= gap.recovered[5] + 1e-9, gap.name
        assert gap.recovered[5] <= 1.0 + 1e-9, gap.name


def test_a_board_dependent_abstraction_shows_real_coupling(args: argparse.Namespace) -> None:
    """Guards the test's own power: a zero gap here would make the rest vacuous."""
    payload = abstraction_coupling.run(args)
    transitions = [gap for gap in payload.gaps if gap.kind == "coupling"]
    assert all(gap.relative > 1e-6 for gap in transitions), [
        (gap.name, gap.relative) for gap in transitions
    ]


def test_it_refuses_more_classes_than_boards(args: argparse.Namespace) -> None:
    args.classes = "1,64"
    with pytest.raises(CommandError, match="already recovers everything"):
        abstraction_coupling.run(args)


def test_it_writes_the_progress_file(args: argparse.Namespace, tmp_path: Path) -> None:
    args.progress_file = str(tmp_path / "progress.json")
    abstraction_coupling.run(args)
    assert (tmp_path / "progress.json").read_text().startswith("{")
