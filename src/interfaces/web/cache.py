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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable


class TtlCache:
    """Memoise values for ``ttl`` seconds, keyed by whatever the caller passes."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        # A Condition rather than a Lock: it is the same mutual exclusion plus
        # the wait/notify single-flight needs, so there is still one lock here.
        self._lock = threading.Condition()
        self._entries: dict[Hashable, tuple[float, Any]] = {}
        self._producing: set[Hashable] = set()

    def get(self, key: Hashable, produce: Callable[[], Any]) -> Any:
        """Return the cached value, or produce and store a fresh one.

        ``produce`` runs OUTSIDE the lock. Holding it across a cloud read would
        serialise every endpoint behind the slowest one -- turning a cache meant
        to reduce load into the thing that makes the console feel slow.

        SINGLE-FLIGHT per key, though: concurrent misses on one key wait for the
        first producer rather than each starting a read. They were all fetching,
        which was defended as "a duplicated read, not a wrong answer" -- true of
        two, and untrue of what actually happens, which is a page mounting eight
        queries on the same tick and a `refetchInterval` firing them together
        forever after. Different keys still overlap freely; that is the part
        that must not be given up.
        """
        while True:
            now = time.monotonic()
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and now - entry[0] < self._ttl:
                    return entry[1]
                if key in self._producing:
                    self._lock.wait()
                    continue
                self._producing.add(key)
                break

        try:
            value = produce()
        except BaseException:
            # Waiters must be woken on failure too, or one bad credential parks
            # every other request on that key until the server is restarted.
            with self._lock:
                self._producing.discard(key)
                self._lock.notify_all()
            raise

        with self._lock:
            self._producing.discard(key)
            self._lock.notify_all()
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
