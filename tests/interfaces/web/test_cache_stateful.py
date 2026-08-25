"""The memo under a SEQUENCE of requests, rather than one at a time.

`test_cache.py` pins each behaviour on its own: a hit, an expiry, a forced
refresh, a stale serve. What no example test reaches is the order they arrive
in — a stale serve landing on a key a forced get is already producing, a clear
between the two, an expiry that only bites because some other key stored and
pruned. Those interleavings are what a console with eight polling panels
actually generates.

This file exists because mutation testing MEASURED the difference, not because
sequences sound worth covering. Two defects pass all 16 example tests and fail
here:

- the pruning predicate reversed (`stored_at - cached[0] > self._ttl`), which
  drops every FRESH entry and keeps the expired ones;
- the forced-get freshness check widened to `entry[2] != asked`, which lets the
  refresh button serve an entry stored BEFORE the button was pressed.

Both are silent in production: the first only shows as a cache that never hits,
the second as a refresh that changes nothing.

Time is injected, as next door. The clock only moves when a rule moves it, so a
run that fails fails identically on replay.
"""

from __future__ import annotations

from unittest import mock

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from src.interfaces.web.cache import TtlCache

TTL = 10.0
GRACE = 5.0
KEYS = st.sampled_from(["a", "b", "c"])


class TtlCacheMachine(RuleBasedStateMachine):
    """A cache and a model of what it is allowed to say, stepped together."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 1000.0
        self._patch = mock.patch(
            "src.interfaces.web.cache.time.monotonic", side_effect=lambda: self.now
        )
        self._patch.start()
        self.cache = TtlCache(TTL)
        # key -> (created_at, value) for what the cache should be holding.
        self.model: dict[str, tuple[float, str]] = {}
        self.created: dict[str, float] = {}
        self.produced: dict[str, tuple[float, str]] = {}
        self.seq: dict[str, int] = {}
        self.calls = 0

    def teardown(self) -> None:
        self.cache.drain(timeout=5)
        self._patch.stop()

    def _producer(self, key: str):
        """A producer that stamps every value with the instant it was made."""

        def produce() -> str:
            self.calls += 1
            self.seq[key] = self.seq.get(key, 0) + 1
            value = f"{key}-{self.seq[key]}"
            self.created[value] = self.now
            self.produced[key] = (self.now, value)
            return value

        return produce

    def _get(self, key: str, **kwargs) -> tuple[str, bool]:
        before = self.calls
        value = self.cache.get(key, self._producer(key), **kwargs)
        # Any stale serve starts a refresh on a thread; let it land so the next
        # rule sees a settled cache rather than a race.
        self.cache.drain(timeout=5)
        if self.calls != before:
            self.model[key] = self.produced[key]
        return value, self.calls != before

    @rule(key=KEYS)
    def plain_get(self, key: str) -> None:
        """The headline contract: what comes back is never older than the TTL."""
        known = self.model.get(key)
        value, produced = self._get(key)

        assert self.now - self.created[value] < TTL
        if not produced:
            assert known is not None, "served a value it never produced"
            assert value == known[1], "served some other key's value"
            assert self.now - known[0] < TTL, "served an entry past its TTL"

    @rule(key=KEYS)
    def stale_get(self, key: str) -> None:
        """Inside the grace an expired value may be served, but only inside it."""
        value, _ = self._get(key, serve_stale_for=GRACE)

        assert self.now - self.created[value] < TTL + GRACE

    @rule(key=KEYS)
    def forced_get(self, key: str) -> None:
        """The refresh button: nothing stored before the call counts."""
        value, produced = self._get(key, force=True)

        assert produced, "a forced get served something it did not just produce"
        assert self.created[value] == self.now

    @rule(seconds=st.floats(min_value=0.0, max_value=3 * TTL, allow_nan=False))
    def advance(self, seconds: float) -> None:
        self.now += seconds

    @rule()
    def clear(self) -> None:
        self.cache.clear()
        self.model.clear()

    @invariant()
    def nothing_is_left_marked_producing(self) -> None:
        """A key stuck in `_producing` parks every later request on it forever —
        the failure the producer's `except BaseException` exists to prevent. With
        no produce in flight the set must be empty."""
        assert not self.cache._producing

    @invariant()
    def a_store_prunes_everything_it_outlives(self) -> None:
        """Expired entries are DROPPED, not merely ignored: the console puts an
        unbounded key space (`/api/runs/{id}`, `/api/logs/{task}`) in front of a
        server that stays up for days. Pruning happens on store, so the newest
        entry is the reference point — nothing older than a TTL behind it may
        remain."""
        entries = dict(self.cache._entries)
        if not entries:
            return

        newest = max(stored_at for stored_at, _, _ in entries.values())
        for key, (stored_at, _, _) in entries.items():
            assert newest - stored_at < TTL, (
                f"{key} survived a later store by {newest - stored_at}s"
            )

    @invariant()
    def the_store_counter_only_moves_forward(self) -> None:
        """`force` decides what counts as "after I asked" by comparing counters,
        so an entry numbered above the cache's own count would let a forced get
        return something stored before it."""
        entries = dict(self.cache._entries)
        stores = self.cache._stores
        for _, _, number in entries.values():
            assert 0 < number <= stores


TestTtlCacheSequences = TtlCacheMachine.TestCase
TestTtlCacheSequences.settings = settings(max_examples=50, stateful_step_count=30, deadline=None)
# The steps are dict operations against a frozen clock -- under a second in
# practice -- but this is one pytest item covering 1,500 of them, so it gets its
# own budget rather than the suite's 5s.
TestTtlCacheSequences.pytestmark = [pytest.mark.timeout(60)]
