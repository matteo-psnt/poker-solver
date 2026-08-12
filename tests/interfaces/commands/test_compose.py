"""The fan-out both composed surfaces read through.

This machinery was `status`'s alone until the console needed the same three
properties -- independent failure, concurrency, and a context copy per submit.
The isolation tests here came WITH it from `test_status_command.py`: they pin
behaviour, and behaviour that moves module should keep being tested rather than
be re-proved at the new call site by whoever notices first.

The property throughout is ISOLATION. A composed screen whose value is that it
still tells you two things when the third is unavailable is worth nothing if one
expired credential blanks it -- and that failure only appears when something is
already wrong, which is exactly when nobody is in a position to debug the
dashboard.
"""

from __future__ import annotations

import argparse
import threading
from typing import Any

import pytest
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from src.interfaces import telemetry
from src.interfaces.commands._base import Command
from src.interfaces.commands._compose import Part, compose, fan_out, payloads
from src.interfaces.errors import CommandError


def _command(name: str, run: Any) -> Command:
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--limit", type=int, default=10)

    return Command(name=name, add_arguments=add_arguments, run=run, render=lambda _p: None)


def _ok(name: str) -> Command:
    return _command(name, lambda args: {"op": name, "limit": args.limit})


def _raising(name: str, error: BaseException) -> Command:
    def _run(_args):
        raise error

    return _command(name, _run)


def _part(name: str, command: Command, **arguments: Any) -> Part:
    return Part(key=name, command=command, arguments=arguments)


class TestOnePartCannotTakeOutTheOthers:
    def test_a_command_error_becomes_an_unavailable_part(self):
        answered = fan_out([_part("tasks", _raising("tasks", CommandError("share unreachable")))])
        assert answered["tasks"] == {"payload": None, "error": "share unreachable"}

    def test_an_expired_login_reads_as_itself(self):
        """The single most likely failure. It must name the fix, not just fail."""
        answered = fan_out([_part("jobs", _raising("jobs", ClientAuthenticationError("nope")))])
        assert answered["jobs"]["payload"] is None
        assert "az login" in answered["jobs"]["error"]

    def test_an_unreachable_endpoint_is_reported_not_raised(self):
        answered = fan_out([_part("pool", _raising("pool", HttpResponseError("gone")))])
        assert answered["pool"]["payload"] is None
        assert "did not answer" in answered["pool"]["error"]

    def test_a_bug_in_a_part_still_propagates(self):
        """Otherwise this becomes where exceptions go to be quietly rendered as
        'unavailable' -- a dashboard that lies rather than one that breaks."""
        with pytest.raises(ValueError, match="kaboom"):
            fan_out([_part("jobs", _raising("jobs", ValueError("kaboom")))])

    def test_the_other_parts_still_answer(self):
        answered = fan_out(
            [
                _part("pool", _ok("pool")),
                _part("jobs", _raising("jobs", ClientAuthenticationError("nope"))),
                _part("tasks", _ok("tasks")),
            ]
        )
        assert answered["pool"]["payload"] == {"op": "pool", "limit": 10}
        assert answered["tasks"]["payload"] == {"op": "tasks", "limit": 10}
        assert answered["jobs"]["payload"] is None


class TestTheFanOut:
    def test_arguments_reach_the_command_through_its_own_parser(self):
        """`invoke` builds the Namespace from the command's declared flags, so a
        composed caller cannot drift from what the command actually accepts."""
        answered = fan_out([_part("jobs", _ok("jobs"), limit=3)])
        assert answered["jobs"]["payload"] == {"op": "jobs", "limit": 3}

    def test_no_parts_is_not_an_error(self):
        """`ThreadPoolExecutor(max_workers=0)` raises. A view that filtered every
        part away wants an empty answer, not a crash."""
        assert fan_out([]) == {}

    def test_parts_really_do_run_at_once(self):
        """Concurrency is the entire reason this exists, and it is invisible in
        every other assertion here -- a serial fan-out passes them all.

        A barrier rather than a sleep-and-time: three parts that each wait for
        the other two can only all return if all three are in flight together,
        so this fails by TIMING OUT under a serial implementation instead of
        being a threshold someone has to tune.
        """
        barrier = threading.Barrier(3, timeout=4)

        def _waits(_args):
            barrier.wait()
            return {"op": "waited"}

        answered = fan_out([_part(str(i), _command(str(i), _waits)) for i in range(3)])
        assert [answered[str(i)]["payload"] for i in range(3)] == [{"op": "waited"}] * 3

    def test_each_part_keeps_the_calling_thread_s_context(self):
        """The trap `d67411f` paid for once.

        `telemetry._SURFACE` is a ContextVar, and a raw `pool.submit` starts its
        task with a fresh context in which it reverts to its default -- so the
        parts carrying the real Azure cost get filed `unknown` while the thin
        wrapper around them is filed as the surface. Nothing else in this file
        notices; the payloads are identical either way.
        """
        seen: list[str] = []

        def _reads_surface(_args):
            # The ContextVar itself: there is no public accessor, and asserting
            # on the telemetry LOG instead would pass on a fresh context too --
            # the row is still written, just filed under the wrong surface.
            seen.append(telemetry._SURFACE.get())
            return {"op": "read"}

        with telemetry.surface("console"):
            fan_out([_part(str(i), _command(str(i), _reads_surface)) for i in range(3)])

        assert seen == ["console"] * 3


class TestCompose:
    def test_the_payload_carries_the_parts_and_how_long_they_took(self):
        composed = compose("view-now", [_part("pool", _ok("pool"))])
        assert composed["op"] == "view-now"
        assert composed["parts"]["pool"]["payload"] == {"op": "pool", "limit": 10}
        assert isinstance(composed["elapsed_seconds"], float)
        assert composed["at"]

    def test_a_join_sees_the_answered_parts(self):
        composed = compose(
            "view-run",
            [_part("runs", _ok("runs"))],
            join=lambda parts: {"joined": sorted(payloads(parts))},
        )
        assert composed["joined"] == ["runs"]

    def test_a_join_does_not_see_a_failed_part(self):
        """A failed part is ABSENT rather than present-and-None, so a join
        cannot read 'Azure did not answer' as 'there are none'."""
        composed = compose(
            "view-run",
            [
                _part("runs", _ok("runs")),
                _part("tasks", _raising("tasks", CommandError("share unreachable"))),
            ],
            join=lambda parts: {"joined": sorted(payloads(parts))},
        )
        assert composed["joined"] == ["runs"]
        assert composed["parts"]["tasks"]["error"] == "share unreachable"

    def test_a_view_still_reports_a_part_that_failed(self):
        """The join is a convenience over the parts, never a replacement: the
        UI needs the reason in order to grey one panel and keep the rest."""
        composed = compose(
            "view-now",
            [_part("pool", _raising("pool", CommandError("no credentials")))],
            join=lambda parts: {"joined": sorted(payloads(parts))},
        )
        assert composed["joined"] == []
        assert composed["parts"]["pool"]["error"] == "no credentials"
