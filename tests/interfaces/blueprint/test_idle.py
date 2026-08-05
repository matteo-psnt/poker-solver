"""Turning itself off: the mechanism that decides nobody is here any more.

Every test drives an injected clock. A real one would make this suite slow, and
a fast timeout would make it flaky — and the thing being tested is a deadline,
which is exactly the shape that produces tests that pass on a fast machine and
fail on a loaded one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.interfaces.blueprint.app import create_app
from src.interfaces.blueprint.idle import IdleWatch
from tests.test_helpers import build_trained_test_solver


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fired():
    calls: list[int] = []
    return calls


class TestTheDeadline:
    def test_it_does_not_fire_before_the_timeout(self, fired):
        clock = Clock()
        watch = IdleWatch(60, on_expire=lambda: fired.append(1), clock=clock)

        clock.advance(59)

        assert watch.check() is False
        assert fired == []

    def test_it_fires_once_the_timeout_passes(self, fired):
        clock = Clock()
        watch = IdleWatch(60, on_expire=lambda: fired.append(1), clock=clock)

        clock.advance(60)

        assert watch.check() is True
        assert fired == [1]

    def test_it_fires_only_once(self, fired):
        """A second signal would interrupt the graceful drain the first started."""
        clock = Clock()
        watch = IdleWatch(60, on_expire=lambda: fired.append(1), clock=clock)
        clock.advance(120)

        watch.check()
        clock.advance(120)
        watch.check()

        assert fired == [1]

    def test_activity_resets_the_clock(self, fired):
        clock = Clock()
        watch = IdleWatch(60, on_expire=lambda: fired.append(1), clock=clock)

        clock.advance(59)
        watch.touch()
        clock.advance(59)

        assert watch.check() is False
        assert fired == []

    def test_idle_seconds_reports_the_gap(self):
        clock = Clock()
        watch = IdleWatch(60, clock=clock)

        clock.advance(12.5)

        assert watch.idle_seconds() == pytest.approx(12.5)


class TestDisabled:
    """A non-positive timeout means stay up — what a laptop and a test want."""

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_it_never_fires(self, timeout, fired):
        clock = Clock()
        watch = IdleWatch(timeout, on_expire=lambda: fired.append(1), clock=clock)

        clock.advance(10_000)

        assert watch.enabled is False
        assert watch.check() is False
        assert fired == []

    def test_starting_the_thread_is_a_no_op(self):
        watch = IdleWatch(0)
        watch.start()

        assert watch._thread is None


class TestThroughTheApp:
    @pytest.fixture(scope="class")
    def client(self):
        solver = build_trained_test_solver(iterations=4)
        return TestClient(create_app(lambda: solver, run_id="test-run"))

    def test_health_reports_what_the_console_needs(self, client):
        body = client.get("/api/health").json()

        assert body["ready"] is True
        assert body["run"] == "test-run"
        assert body["idle_seconds"] >= 0
        assert body["idle_timeout_seconds"] == 0

    def test_any_request_counts_as_activity(self, client):
        """Including one that refuses — a 422 is still a person at the keyboard."""
        before = client.get("/api/health").json()["idle_seconds"]
        client.get("/api/node", params={"path": "b999999"})
        after = client.get("/api/health").json()["idle_seconds"]

        assert after <= before + 1.0

    def test_sessions_are_counted(self, client):
        start = client.get("/api/health").json()["sessions"]
        client.post("/api/play", json={"human_seat": 0, "seed": 1})

        assert client.get("/api/health").json()["sessions"] == start + 1
