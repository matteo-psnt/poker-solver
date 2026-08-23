"""The memo the console sits behind: expiry, the lock, and what it forgets.

Time is injected rather than slept through -- a TTL test that waits is both slow
and flaky, and the interesting cases (an entry one microsecond past expiry)
cannot be reached by sleeping at all.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

import pytest

from src.interfaces.web.cache import TtlCache

if TYPE_CHECKING:
    from collections.abc import Callable

TTL = 10.0


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    """A monotonic clock the test moves by hand. Returns the setter."""
    now = 1000.0

    def _set(value: float) -> None:
        nonlocal now
        now = value

    monkeypatch.setattr("src.interfaces.web.cache.time.monotonic", lambda: now)
    return _set


@pytest.fixture
def counted() -> Callable[[str], Callable[[], str]]:
    """A producer that records how often it actually ran."""

    def _make(value: str) -> Callable[[], str]:
        calls = 0

        def _produce() -> str:
            nonlocal calls
            calls += 1
            return f"{value}-{calls}"

        return _produce

    return _make


def test_a_hit_does_not_run_the_producer(clock, counted):
    cache = TtlCache(TTL)
    produce = counted("pool")

    assert cache.get("pool", produce) == "pool-1"
    assert cache.get("pool", produce) == "pool-1"


def test_an_expired_entry_is_produced_again(clock, counted):
    cache = TtlCache(TTL)
    produce = counted("pool")
    cache.get("pool", produce)

    clock(1000.0 + TTL)

    assert cache.get("pool", produce) == "pool-2"


def test_distinct_keys_do_not_share_a_value(clock, counted):
    cache = TtlCache(TTL)

    assert cache.get("jobs", counted("jobs")) == "jobs-1"
    assert cache.get("tasks", counted("tasks")) == "tasks-1"


def test_clear_forgets_everything(clock, counted):
    cache = TtlCache(TTL)
    produce = counted("pool")
    cache.get("pool", produce)

    cache.clear()

    assert cache.get("pool", produce) == "pool-2"


def test_expired_entries_are_dropped_not_merely_ignored(clock, counted):
    """The console keys on run id and task id, so the key space is unbounded.

    Ignoring an expired entry is enough to be correct and not enough to be
    bounded: a server left up for days would hold every payload it ever served.
    """
    cache = TtlCache(TTL)
    for index in range(50):
        clock(1000.0 + index)
        cache.get(f"run-{index}", counted("x"))

    assert len(cache) <= 1 + int(TTL)


def test_the_producer_runs_outside_the_lock(clock):
    """Holding the lock across a 2-4s cloud read would serialise every endpoint
    behind the slowest one -- a cache that makes the console feel slower."""
    cache = TtlCache(TTL)

    def _reentrant() -> str:
        # Would deadlock on a non-reentrant lock still held by the outer `get`.
        return cache.get("inner", lambda: "inner-value")

    assert cache.get("outer", _reentrant) == "inner-value"


class TestSingleFlight:
    """One key, one producer. The rest wait for it rather than joining in.

    A page mounts eight queries on the same tick and `refetchInterval` fires
    them together forever after, so "two simultaneous misses both fetch" is not
    the rare case it was defended as -- it is every refresh, multiplied by open
    tabs, against a share where the whole cost is round trips.
    """

    def test_concurrent_misses_on_one_key_produce_once(self, clock):
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def _slow() -> str:
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=2)
            return "value"

        cache = TtlCache(TTL)
        results: list[str] = []
        threads = [
            threading.Thread(target=lambda: results.append(cache.get("pool", _slow)))
            for _ in range(8)
        ]
        threads[0].start()
        assert started.wait(timeout=2)  # the first is committed before the rest arrive
        for thread in threads[1:]:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=2)

        assert calls == 1
        assert results == ["value"] * 8

    def test_different_keys_still_overlap(self, clock):
        """The property single-flight must not cost: `/api/pool` waiting on
        `/api/tasks` would serialise the whole console behind its slowest read."""
        both_inside = threading.Barrier(2, timeout=2)

        def _produce() -> str:
            both_inside.wait()  # deadlocks if the two are serialised
            return "value"

        cache = TtlCache(TTL)
        threads = [
            threading.Thread(target=cache.get, args=(key, _produce)) for key in ("pool", "tasks")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()

    def test_a_failure_releases_the_waiters(self, clock):
        """Otherwise one expired credential parks every other request on that
        key until the server is restarted."""
        cache = TtlCache(TTL)

        def _fails() -> str:
            raise RuntimeError("Azure said no")

        with pytest.raises(RuntimeError):
            cache.get("pool", _fails)

        done = threading.Event()

        def _second() -> None:
            with contextlib.suppress(RuntimeError):
                cache.get("pool", _fails)
            done.set()

        threading.Thread(target=_second).start()
        assert done.wait(timeout=2), "a waiter was left parked by the failed producer"


def test_a_failing_producer_stores_nothing(clock, counted):
    """A refusal must not be memoised: `az login` fixes the cause immediately,
    and serving the stale failure for the whole TTL hides the fix."""
    cache = TtlCache(TTL)

    def _fails() -> str:
        raise RuntimeError("Azure said no")

    with pytest.raises(RuntimeError, match="Azure said no"):
        cache.get("pool", _fails)

    assert cache.get("pool", counted("pool")) == "pool-1"


class TestStaleWhileRevalidate:
    """A polled view answers at once from the expired entry and refreshes behind it."""

    def test_inside_the_grace_the_expired_value_is_served_and_refreshed_once(self, clock, counted):
        cache = TtlCache(TTL)
        produce = counted("now")
        assert cache.get("now", produce, serve_stale_for=30.0) == "now-1"

        clock(1000.0 + TTL + 1)
        assert cache.get("now", produce, serve_stale_for=30.0) == "now-1", "the poll blocked"
        cache.drain(timeout=2)
        assert cache.get("now", produce, serve_stale_for=30.0) == "now-2"

    def test_a_refresh_already_running_is_not_started_twice(self, clock):
        cache = TtlCache(TTL)
        started, release = threading.Event(), threading.Event()
        calls = 0

        def produce() -> str:
            nonlocal calls
            calls += 1
            if calls > 1:
                started.set()
                release.wait(timeout=2)
            return f"v{calls}"

        assert cache.get("now", produce, serve_stale_for=30.0) == "v1"
        clock(1000.0 + TTL + 1)
        for _ in range(5):
            assert cache.get("now", produce, serve_stale_for=30.0) == "v1"
        assert started.wait(timeout=2)
        release.set()
        cache.drain(timeout=2)
        assert calls == 2

    def test_past_the_grace_the_call_blocks_and_a_failure_reaches_the_caller(self, clock):
        """A refresh that keeps failing -- an expired `az login` -- must not
        hide behind an ageing value forever."""
        cache = TtlCache(TTL)
        answers: list = ["first", RuntimeError("az login has expired")]

        def produce() -> str:
            answer = answers[0] if len(answers) == 1 else answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        assert cache.get("now", produce, serve_stale_for=30.0) == "first"
        clock(1000.0 + TTL + 1)
        assert cache.get("now", produce, serve_stale_for=30.0) == "first"
        cache.drain(timeout=2)
        assert cache.get("now", produce, serve_stale_for=30.0) == "first", "still inside the grace"
        clock(1000.0 + TTL + 31)
        with pytest.raises(RuntimeError, match="az login"):
            cache.get("now", produce, serve_stale_for=30.0)

    def test_without_a_grace_an_expired_entry_still_blocks(self, clock, counted):
        """`answer()` endpoints: a bare payload has no timestamp of its own, so
        serving it stale would make the client's fetch time a lie."""
        cache = TtlCache(TTL)
        produce = counted("pool")
        assert cache.get("pool", produce) == "pool-1"
        clock(1000.0 + TTL + 1)
        assert cache.get("pool", produce) == "pool-2"


class TestForce:
    def test_a_forced_get_produces_even_inside_the_ttl(self, clock, counted):
        cache = TtlCache(TTL)
        produce = counted("now")
        assert cache.get("now", produce) == "now-1"
        assert cache.get("now", produce, force=True) == "now-2"
        assert cache.get("now", produce) == "now-2", "the forced value was not stored"

    def test_a_forced_get_joins_a_produce_already_in_flight(self, clock):
        """A refresh running behind a stale serve is as fresh as a new one."""
        cache = TtlCache(TTL)
        started, release = threading.Event(), threading.Event()
        calls = 0

        def produce() -> str:
            nonlocal calls
            calls += 1
            if calls > 1:
                started.set()
                release.wait(timeout=2)
            return f"v{calls}"

        assert cache.get("now", produce, serve_stale_for=30.0) == "v1"
        clock(1000.0 + TTL + 1)
        assert cache.get("now", produce, serve_stale_for=30.0) == "v1"
        assert started.wait(timeout=2)
        forced: list[str] = []
        thread = threading.Thread(
            target=lambda: forced.append(cache.get("now", produce, force=True))
        )
        thread.start()
        release.set()
        thread.join(timeout=2)
        assert forced == ["v2"]
        assert calls == 2
