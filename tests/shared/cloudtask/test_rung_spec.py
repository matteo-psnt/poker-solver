"""`run[:at[:threshold]]`, read the way `deploy.sh` writes it."""

from __future__ import annotations

import pytest

from src.shared.cloudtask.rung_spec import RungSpec, format_ladder, parse_ladder


def test_all_three_fields_are_split():
    """`deploy.sh` once split on the LAST colon and staged a rung named "0.20"."""
    assert RungSpec.parse("run-d50:12000:0.35") == RungSpec("run-d50", 12000, 0.35)


def test_the_rung_and_threshold_are_optional():
    assert RungSpec.parse("run-d50") == RungSpec("run-d50", None, 0.0)
    assert RungSpec.parse("run-d50::0.2") == RungSpec("run-d50", None, 0.2)


def test_a_threshold_in_the_rung_slot_is_refused():
    with pytest.raises(ValueError, match="not an iteration"):
        RungSpec.parse("run-d50:0.35")


def test_a_ladder_round_trips():
    text = "run-a:4000:0.35,run-b::0.2,run-c:12000:0"
    assert format_ladder(parse_ladder(text)) == text


def test_an_empty_ladder_is_refused():
    with pytest.raises(ValueError, match="names no rung"):
        parse_ladder(" , ")
