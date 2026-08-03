"""The memo the console sits behind: expiry, the lock, and what it forgets.

Time is injected rather than slept through -- a TTL test that waits is both slow
and flaky, and the interesting cases (an entry one microsecond past expiry)
cannot be reached by sleeping at all.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.interfaces.web.cache import TtlCache

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
    assert cache.get("legs", counted("legs")) == "legs-1"


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


def test_a_failing_producer_stores_nothing(clock, counted):
    """A refusal must not be memoised: `az login` fixes the cause immediately,
    and serving the stale failure for the whole TTL hides the fix."""
    cache = TtlCache(TTL)

    def _fails() -> str:
        raise RuntimeError("Azure said no")

    with pytest.raises(RuntimeError, match="Azure said no"):
        cache.get("pool", _fails)

    assert cache.get("pool", counted("pool")) == "pool-1"
