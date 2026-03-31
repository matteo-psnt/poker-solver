"""A small time-to-live memo, so open tabs share one cloud sweep.

Deliberately not a poller. An earlier design had a background thread sweeping
every panel on a schedule, because a full sweep cost ~60s and the browser could
not be allowed to trigger one. At 2-4s per read that is no longer true, and a
memo has a property the poller did not: nothing depends on a thread staying
alive, so a dead refresher cannot silently serve yesterday's numbers forever.

Entries expire; they are never invalidated. The freshest thing this can serve is
``ttl`` seconds old, which is the honest bound and is what the UI displays.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable
from typing import Any


class TtlCache:
    """Memoise values for ``ttl`` seconds, keyed by whatever the caller passes."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        self._entries: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable, produce: Callable[[], Any]) -> Any:
        """Return the cached value, or produce and store a fresh one.

        ``produce`` runs OUTSIDE the lock. Holding it across a 4s cloud read
        would serialise every endpoint behind the slowest one -- turning a cache
        meant to reduce load into the thing that makes the console feel slow.
        The cost is that two simultaneous misses on the same key both fetch,
        which is a duplicated read, not a wrong answer.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now - entry[0] < self._ttl:
                return entry[1]

        value = produce()

        with self._lock:
            self._entries[key] = (time.monotonic(), value)
        return value

    def clear(self) -> None:
        """Drop everything. For a forced refresh, and for tests."""
        with self._lock:
            self._entries.clear()
