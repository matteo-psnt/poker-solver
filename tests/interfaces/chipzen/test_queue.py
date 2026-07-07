"""The queue keeper: what makes the seat wanted rather than merely reachable.

Every test here pins a way the seat goes quiet without dying -- the failure the
supervision work cannot see, because the process stays perfectly healthy while no
match ever arrives.

Driven with `asyncio.run` rather than a plugin: one coroutine under test does not
justify a test-time dependency, and the loop is stopped through the `in_flight`
callback it already calls once per pass.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.interfaces.chipzen import seat as seat_module
from src.interfaces.chipzen.seat import _keep_queued, _rest_base


class TestRestBase:
    def test_the_rest_origin_comes_off_the_lobby_url(self) -> None:
        assert _rest_base("wss://chipzen.ai/ws/external/bot/abc-123") == "https://chipzen.ai"

    def test_staging_is_not_special_cased(self) -> None:
        # The SDK owns the env->host mapping; a second copy is a second thing to
        # get wrong when they add a host.
        assert _rest_base("wss://staging.chipzen.ai/ws/external/bot/x") == (
            "https://staging.chipzen.ai"
        )

    def test_a_plain_ws_url_stays_plain(self) -> None:
        assert _rest_base("ws://localhost:8000/ws/external/bot/x") == "http://localhost:8000"


class _Chipzen:
    """Their matchmaking endpoints, enough of them to drive the keeper."""

    def __init__(self, *, status: str = "idle", expires: bool = False) -> None:
        self.status = status
        self.expires = expires
        self.joins = 0
        self.status_reads = 0
        self.fail_with: Exception | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.fail_with is not None:
            raise self.fail_with
        if request.url.path.endswith("/matchmaking/status"):
            self.status_reads += 1
            return httpx.Response(
                200,
                json={
                    "status": self.status,
                    "position": None,
                    "waiting_seconds": None,
                    # Zero, so the keeper's period collapses to the floor the
                    # fixture has already pinned to nothing.
                    "queue_ttl_seconds": 0,
                },
            )
        if request.url.path.endswith("/matchmaking/join"):
            self.joins += 1
            # Their entry expires on its own; `expires` is whether this fake
            # models that or holds the queued state.
            self.status = "idle" if self.expires else "queued"
            return httpx.Response(200, json={"status": "queued", "position": 1})
        raise AssertionError(f"unexpected request to {request.url}")


@pytest.fixture
def chipzen(monkeypatch):
    """A fake Chipzen, with the keeper's pacing removed."""
    monkeypatch.setattr(seat_module, "_QUEUE_MIN_PERIOD_S", 0.0)
    server = _Chipzen()
    original = httpx.AsyncClient

    def _factory(**kwargs):
        return original(**kwargs, transport=httpx.MockTransport(server.handle))

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return server


def _run(*, playing: int, passes: int) -> None:
    """Let the keeper make `passes` passes, then cancel it from inside."""
    calls = 0

    def _in_flight() -> int:
        nonlocal calls
        calls += 1
        if calls > passes:
            raise asyncio.CancelledError
        return playing

    async def _drive() -> None:
        with pytest.raises(asyncio.CancelledError):
            await _keep_queued("https://chipzen.ai", "tok", _in_flight)

    asyncio.run(_drive())


class TestKeepQueued:
    def test_an_idle_bot_is_put_in_the_queue(self, chipzen) -> None:
        _run(playing=0, passes=1)
        assert chipzen.joins == 1

    def test_it_keeps_rejoining_because_the_entry_expires(self, chipzen) -> None:
        # `queue_ttl_seconds: 60` makes being queued something you keep doing. A
        # keeper that joined once would silently drop out after a minute and look
        # exactly like a healthy idle seat.
        chipzen.expires = True
        _run(playing=0, passes=3)
        assert chipzen.joins == 3

    def test_a_seat_already_playing_is_never_queued_again(self, chipzen) -> None:
        # A second match dealt into this process would share the in-memory table
        # with the first.
        _run(playing=1, passes=3)
        assert chipzen.joins == 0
        assert chipzen.status_reads == 0

    def test_a_bot_already_queued_is_not_joined_twice(self, chipzen) -> None:
        chipzen.status = "queued"
        _run(playing=0, passes=3)
        assert chipzen.joins == 0

    def test_a_queue_that_is_down_does_not_take_the_lobby_with_it(self, chipzen) -> None:
        # The lobby can still be handed a challenge while matchmaking is out, so
        # the keeper must never be the thing that ends the seat.
        chipzen.fail_with = httpx.ConnectError("matchmaking is down")
        _run(playing=0, passes=3)

    def test_a_refused_join_is_survived(self, chipzen) -> None:
        # 4xx from `join` is their business -- a full queue, a rate limit, a bot
        # already entered elsewhere. None of them is a reason to stop holding the
        # seat, and `raise_for_status` would otherwise end the task.
        def _refuse(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/matchmaking/join"):
                return httpx.Response(429, json={"error_code": "RATE_LIMIT"})
            return _Chipzen.handle(chipzen, request)

        chipzen.handle = _refuse  # type: ignore[method-assign]
        _run(playing=0, passes=2)
