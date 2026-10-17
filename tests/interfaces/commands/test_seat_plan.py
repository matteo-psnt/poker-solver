"""`seat-plan` resolves the seat file against what the container holds."""

from __future__ import annotations

import pytest

from src.interfaces.commands.seat_plan import plan
from src.interfaces.errors import CommandError

HELD = {
    "deep": {"static-400.ckpt.zst", "static-64200.ckpt.zst", "static-65000.ckpt.zst"},
    "shallow": {"static-12000.ckpt.zst", "static-50000.ckpt.zst"},
}
SPEC = {"run": "deep", "at": "latest", "threshold": 0.35, "rungs": ["shallow:50000:0.20"]}


def held(run: str) -> set[str]:
    return HELD.get(run, set())


def test_latest_is_the_newest_rung_held_and_a_number_pins():
    resolved = plan(SPEC, held)
    assert resolved.base.spec == "deep:65000:0.35"
    assert [seat.spec for seat in resolved.rungs] == ["shallow:50000:0.20"]


def test_the_digest_moves_when_what_is_fielded_moves():
    before = plan(SPEC, held).digest
    HELD["deep"].add("static-65200.ckpt.zst")
    try:
        assert plan(SPEC, held).digest != before
    finally:
        HELD["deep"].discard("static-65200.ckpt.zst")


@pytest.mark.parametrize(
    "rung", ["shallow:40000:0.20", "nosuchrun:latest:0.20", "shallow:latests:0.20"]
)
def test_a_rung_the_container_does_not_hold_is_refused(rung):
    with pytest.raises(CommandError):
        plan({**SPEC, "rungs": [rung]}, held)
