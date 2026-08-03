"""A small time-to-live memo, so open tabs share one cloud sweep.

Deliberately not a poller. An earlier design had a background thread sweeping
every panel on a schedule, because a full sweep cost ~60s and the browser could
not be allowed to trigger one. At 2-4s per read that is no longer true, and a
memo has a property the poller did not: nothing depends on a thread staying
alive, so a dead refresher cannot silently serve yesterday's numbers forever.

Entries expire; they are never invalidated. The freshest thing this can serve is
``ttl`` seconds old, which is the honest bound and is what the UI displays.
Expired entries are also dropped, not merely ignored -- a per-run and per-task
key space in front of a server that stays up for days is otherwise a slow leak.
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
            # Drop what has expired before inserting. Otherwise the dict only
            # ever grows: `/api/runs/{id}` and `/api/logs/{task}` put an
            # unbounded key space in front of a server that stays up for days,
            # and an entry nobody will read again is still held here with its
            # whole payload.
            stored_at = time.monotonic()
            self._entries = {
                other: cached
                for other, cached in self._entries.items()
                if stored_at - cached[0] < self._ttl
            }
            self._entries[key] = (stored_at, value)
        return value

    def clear(self) -> None:
        """Drop everything. For a forced refresh, and for tests."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """How many entries are held, expired ones included.

        The number a leak shows up in: correctness only needs expired entries to
        be ignored, so nothing else would notice them still being here.
        """
        with self._lock:
            return len(self._entries)
