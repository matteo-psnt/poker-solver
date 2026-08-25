"""`--expect` is the gate the blueprint's number depends on, so it must fail.

Both defects here were MEASURED against their live server on 2026-09-01, and
both made a broken run look validated.
"""

from __future__ import annotations

import argparse

import pytest

from src.interfaces.commands import benchmark
from src.interfaces.errors import CommandError
from src.interfaces.gtowizard.session import HandRecord, Tally


def _args(**over: object) -> argparse.Namespace:
    base = {
        "agent": "always-fold",
        "hands": 10,
        "concurrency": 1,
        "sigma": 3.0,
        "expect": "always-fold",
        "run": None,
        "runs_dir": "",
        "at": None,
        "no_resolver": False,
        "allow_depth_mismatch": False,
        "log": None,
        "seed": 0,
        "game": "HUNL 200BB",
        "version": 2,
    }
    return argparse.Namespace(**{**base, **over})


def _tally(*records: HandRecord) -> Tally:
    tally = Tally()
    for record in records:
        tally.add(record)
    return tally


def _hand(aivat: float, hand_id: int = 1) -> HandRecord:
    return HandRecord(
        hand_id=hand_id,
        big_blind=100,
        winnings=aivat,
        aivat=aivat,
        decisions=1,
        off_tree=0,
        truncated=False,
    )


def test_a_probe_that_played_nothing_cannot_match() -> None:
    """MEASURED: 2,000 hands all 409'd on their 20-hand cap, and the run printed
    "matches the published check-call at 0.0σ" -- because an empty tally has an
    infinite standard error, which drives the z-score to zero."""
    with pytest.raises(CommandError, match=r"played 0 hands"):
        benchmark._check_expectation(_args(), _tally())


def test_a_handful_of_hands_cannot_match_however_close_they_land() -> None:
    """Two hands that agree make the band arbitrarily tight, which says nothing:
    the standard error is estimated off the same two samples."""
    with pytest.raises(CommandError, match=r"at least 200"):
        benchmark._check_expectation(_args(), _tally(_hand(-60.0), _hand(-66.0, 2)))


def test_a_large_but_hopelessly_noisy_sample_is_refused() -> None:
    """Enough hands, but a band a doubled score would sit inside."""
    spread = [_hand(-63.0 + (400 if i % 2 else -400), i) for i in range(400)]
    with pytest.raises(CommandError, match=r"too noisy"):
        benchmark._check_expectation(_args(), _tally(*spread))
