"""The remote-key cache must be bounded, and honest about what it dropped.

Unbounded, this cache converges on the whole tree in EVERY worker, so total
memory scales O(N x workers) instead of O(N). Measured at 315 B/entry that is
~37 GB across 16 workers at 7M infosets, which is what killed three legs on a
32 GB node.

A miss is not an error -- the caller queues an id request and drops one update --
so the cap is a memory/sample-efficiency trade, and the eviction count is the
only way to see when it has been set too low.
"""

from __future__ import annotations

from src.core.game.state import Street
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage.shared_array.remote_cache import UNBOUNDED, RemoteKeyCache


def key(i: int) -> InfoSetKey:
    return InfoSetKey(i & 1, Street.RIVER, f"b0.{i % 100}-c-b{i}", None, i % 600, i % 3)


class TestBounding:
    def test_never_exceeds_capacity(self):
        cache = RemoteKeyCache(capacity=100)
        for i in range(1000):
            cache[key(i)] = i
        assert len(cache) == 100

    def test_bulk_update_overshoot_is_trimmed_in_one_pass(self):
        # `update` arrives from the id-exchange in batches and can overshoot the
        # cap by many entries at once, not one.
        cache = RemoteKeyCache(capacity=50)
        cache.update({key(i): i for i in range(500)})
        assert len(cache) == 50

    def test_evicts_oldest_first_and_keeps_newest(self):
        cache = RemoteKeyCache(capacity=10)
        for i in range(20):
            cache[key(i)] = i
        assert key(0) not in cache, "oldest should have been evicted"
        assert key(19) in cache, "newest must be retained"
        assert cache.get(key(19)) == 19

    def test_zero_capacity_means_unbounded_for_ab_measurement(self):
        cache = RemoteKeyCache(capacity=UNBOUNDED)
        for i in range(5000):
            cache[key(i)] = i
        assert len(cache) == 5000
        assert cache.evictions == 0


class TestEvictionsAreVisible:
    def test_counts_what_was_dropped(self):
        cache = RemoteKeyCache(capacity=100)
        for i in range(250):
            cache[key(i)] = i
        assert cache.evictions == 150

    def test_no_evictions_when_it_fits(self):
        cache = RemoteKeyCache(capacity=1000)
        cache.update({key(i): i for i in range(400)})
        assert cache.evictions == 0

    def test_bulk_overshoot_is_counted_in_full(self):
        cache = RemoteKeyCache(capacity=50)
        cache.update({key(i): i for i in range(500)})
        assert cache.evictions == 450


class TestLookupContract:
    """Whatever the eviction policy, resident keys must answer correctly."""

    def test_miss_returns_none_not_a_raise(self):
        # get_or_create_infoset branches on `is None`; a KeyError here would
        # abort a batch instead of queuing an id request.
        assert RemoteKeyCache(capacity=10).get(key(1)) is None

    def test_resident_keys_round_trip(self):
        cache = RemoteKeyCache(capacity=1000)
        for i in range(100):
            cache[key(i)] = i * 7
        for i in range(100):
            assert cache.get(key(i)) == i * 7
            assert key(i) in cache

    def test_reinsertion_after_eviction_works(self):
        # The recovery path: evicted -> re-requested -> re-inserted.
        cache = RemoteKeyCache(capacity=5)
        for i in range(20):
            cache[key(i)] = i
        assert key(0) not in cache
        cache[key(0)] = 999
        assert cache.get(key(0)) == 999
