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
from src.interfaces.chipzen.seat import _keep_queued, _rest_base, _retry_after


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

    def __init__(
        self,
        *,
        status: str = "idle",
        expires: bool = False,
        waiting: float = 0.0,
        ttl: float = 60.0,
    ) -> None:
        self.status = status
        self.expires = expires
        # Their real numbers: an entry lives 60 s and reports its own age, which
        # is what tells the keeper to refresh before it lapses.
        self.waiting = waiting
        self.ttl = ttl
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
                    "position": 1 if self.status == "queued" else None,
                    "waiting_seconds": self.waiting if self.status == "queued" else None,
                    "queue_ttl_seconds": self.ttl,
                },
            )
        if request.url.path.endswith("/matchmaking/join"):
            self.joins += 1
            # MEASURED 09-01: a join REFRESHES a live entry rather than being
            # refused -- `waiting_seconds` 33 -> 2. That is the whole reason the
            # keeper may refresh instead of waiting for expiry.
            self.waiting = 0.0
            # Their entry expires on its own; `expires` is whether this fake
            # models that or holds the queued state.
            self.status = "idle" if self.expires else "queued"
            return httpx.Response(200, json={"status": "queued", "position": 1})
        raise AssertionError(f"unexpected request to {request.url}")


@pytest.fixture
def chipzen(monkeypatch):
    """A fake Chipzen, with the keeper's pacing removed."""
    monkeypatch.setattr(seat_module, "_QUEUE_POLL_S", 0.0)
    # ...and its rate-limit backoff, which is real seconds by design.
    monkeypatch.setattr(seat_module, "_QUEUE_BACKOFF_START_S", 0.0)
    monkeypatch.setattr(seat_module, "_QUEUE_BACKOFF_MAX_S", 0.0)
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


class TestRateLimitBackoff:
    """A 429 must be paid for in seconds, not in a poll period.

    Measured live on 09-01: a third of joins were rate limited, and because a
    429 fell through to the generic error path the keeper then slept the FULL
    ~30 s poll period each time -- about 14% of wall clock spent unqueued for a
    limit that clears in seconds.
    """

    def test_their_retry_after_is_obeyed(self) -> None:
        assert _retry_after({"retry-after": "3"}, 0.0) == 3.0

    def test_a_date_form_retry_after_falls_back_to_doubling(self) -> None:
        # HTTP allows an HTTP-date here. Parsing it is not worth it; not
        # CRASHING on it is.
        assert _retry_after({"retry-after": "Wed, 01 Sep 2026 06:00:00 GMT"}, 0.0) == (
            seat_module._QUEUE_BACKOFF_START_S
        )

    def test_it_doubles_without_a_header(self) -> None:
        assert _retry_after({}, 0.0) == seat_module._QUEUE_BACKOFF_START_S
        assert _retry_after({}, 2.0) == 4.0

    def test_the_backoff_is_bounded_by_the_period_it_replaces(self) -> None:
        assert _retry_after({}, 1e6) == seat_module._QUEUE_BACKOFF_MAX_S
        assert _retry_after({"retry-after": "99999"}, 0.0) == seat_module._QUEUE_BACKOFF_MAX_S

    def test_a_limited_join_is_retried_rather_than_abandoned(self, chipzen) -> None:
        # The seat must keep asking: a rate limit is the queue working, not the
        # queue being down.
        attempts = 0

        def _limit(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if request.url.path.endswith("/matchmaking/join"):
                attempts += 1
                return httpx.Response(429, json={"error_code": "RATE_LIMIT"})
            return _Chipzen.handle(chipzen, request)

        chipzen.handle = _limit  # type: ignore[method-assign]
        _run(playing=0, passes=3)
        assert attempts == 3


class TestEntryRefresh:
    """The entry is REFRESHED before it lapses, not re-created after it has.

    Joining only on `idle` meant the keeper waited for the 60 s entry to expire
    and then noticed a poll later. Measured live 09-01: 63 s between joins
    against a 60 s TTL, so the seat sat OUT of the queue ~24% of the time and
    every join landed in the window their rate limiter is touchiest.

    A join while queued refreshes rather than being refused -- measured against
    the live endpoint, `waiting_seconds` 33 -> 2 with HTTP 200 -- which is what
    makes refreshing safe.
    """

    def test_an_aging_entry_is_refreshed(self, chipzen) -> None:
        chipzen.status = "queued"
        chipzen.waiting = 20.0  # older than _QUEUE_REFRESH_AFTER_S
        _run(playing=0, passes=1)
        assert chipzen.joins == 1
        assert chipzen.waiting == 0.0

    def test_a_fresh_entry_is_left_alone(self, chipzen) -> None:
        # Refreshing every pass would multiply the join rate against a limiter
        # that already pushes back with 429s.
        chipzen.status = "queued"
        chipzen.waiting = 5.0
        _run(playing=0, passes=3)
        assert chipzen.joins == 0

    def test_the_entry_never_lapses_across_many_passes(self, chipzen) -> None:
        # The whole point: drive the keeper for a while and the queue entry is
        # never once allowed to age out.
        chipzen.status = "queued"
        aged = []

        original = _Chipzen.handle

        def _age(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/matchmaking/status"):
                chipzen.waiting += 20.0  # 20 s of ageing per poll
                aged.append(chipzen.waiting)
            return original(chipzen, request)

        chipzen.handle = _age  # type: ignore[method-assign]
        _run(playing=0, passes=10)
        # 35 s is the shortest entry lifetime measured live; their advertised
        # 60 s ttl is NOT a safe bound and must not be asserted against.
        assert max(aged) < 35.0, f"entry reached {max(aged)}s; live entries died at 12-46s"
