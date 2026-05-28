"""Switching the loaded run in place, which used to mean a three-minute deploy.

The property under test is not "the endpoint returns 200" -- it is that a swap
either lands completely or fails loudly, and never leaves the server answering
for one run out of another run's state.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from src.interfaces.blueprint.app import create_app
from tests.test_helpers import build_trained_test_solver


@pytest.fixture(scope="module")
def solvers():
    """Two distinguishable blueprints, built once — each is seconds, not free."""
    return {
        "first": build_trained_test_solver(iterations=40),
        "second": build_trained_test_solver(iterations=60),
    }


def app_that_can_switch(solvers, *, fail: bool = False, block: threading.Event | None = None):
    """A server whose `load_run` hands back the second solver, or explodes."""

    def load_run(run: str, at_iteration: int | None):
        if block is not None:
            block.wait(timeout=5)
        if fail:
            raise RuntimeError("no such run on the share")
        return solvers["second"], run

    return create_app(lambda: solvers["first"], run_id="first", load_run=load_run)


class TestASwapThatLands:
    def test_the_run_is_replaced(self, solvers):
        client = TestClient(app_that_can_switch(solvers))
        assert client.get("/api/run").json()["run"] == "first"

        started = client.post("/api/load", json={"run": "second"})
        # 202, because the work outlives the request on purpose.
        assert started.status_code == 202

        _settle(client)
        assert client.get("/api/run").json()["run"] == "second"
        assert client.get("/api/health").json()["ready"] is True

    def test_hands_in_progress_do_not_survive_it(self, solvers):
        """The one correctness rule: a hand belongs to the run it was dealt from.

        Carrying a session across a swap would let the bot's next action come
        from a different blueprint, mid-hand, with nothing on screen to say so.
        """
        client = TestClient(app_that_can_switch(solvers))
        session = client.post("/api/play", json={"human_seat": 0}).json()["session"]
        assert client.get(f"/api/play/{session}").status_code == 200

        client.post("/api/load", json={"run": "second"})
        _settle(client)

        assert client.get(f"/api/play/{session}").status_code == 404


class TestASwapThatFails:
    def test_the_reason_is_reported_and_the_server_stays_up(self, solvers):
        client = TestClient(app_that_can_switch(solvers, fail=True))
        client.post("/api/load", json={"run": "nope"})
        _settle(client)

        health = client.get("/api/health").json()
        assert health["ready"] is True
        assert "no such run on the share" in health["last_error"]


class TestRefusals:
    def test_a_server_with_no_way_to_resolve_a_run_says_so(self, solvers):
        """A laptop and a test serve a solver handed to them directly.

        422 with a sentence rather than 404: the endpoint exists, and the reason
        it cannot help is a fact about how this server was started.
        """
        client = TestClient(create_app(lambda: solvers["first"], run_id="first"))
        answer = client.post("/api/load", json={"run": "second"})
        assert answer.status_code == 422
        assert "cannot switch runs" in answer.json()["error"]
        assert client.get("/api/health").json()["can_switch"] is False

    def test_a_second_load_is_refused_while_one_is_running(self, solvers):
        """Refused, not queued: a caller cannot see a queue it is waiting behind."""
        gate = threading.Event()
        client = TestClient(app_that_can_switch(solvers, block=gate))
        try:
            assert client.post("/api/load", json={"run": "second"}).status_code == 202
            assert client.post("/api/load", json={"run": "third"}).status_code == 409
        finally:
            gate.set()
        _settle(client)

    def test_reads_refuse_while_a_swap_is_in_flight(self, solvers):
        """503, not a stale answer: there is no blueprint to answer FROM."""
        gate = threading.Event()
        client = TestClient(app_that_can_switch(solvers, block=gate))
        try:
            client.post("/api/load", json={"run": "second"})
            assert client.get("/api/node").status_code == 503
            # The endpoint the caller watches must keep answering throughout.
            assert client.get("/api/health").json()["loading"] == "second"
        finally:
            gate.set()
        _settle(client)


def _settle(client, tries: int = 100) -> None:
    """Wait for the background swap to finish, via the endpoint a client uses."""
    for _ in range(tries):
        if client.get("/api/health").json()["loading"] is None:
            return
        threading.Event().wait(0.05)
    raise AssertionError("the load never finished")
