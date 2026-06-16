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

A caller may ask for an expired entry to be served STALE while one refresh runs
behind it (``serve_stale_for``). That is still request-driven: nothing sweeps on
a timer, a closed tab stops the refreshes, and a refresher that keeps failing is
noticed, because past the grace the request blocks and fails in the open.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

logger = logging.getLogger(__name__)


class TtlCache:
    """Memoise values for ``ttl`` seconds, keyed by whatever the caller passes."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        # A Condition rather than a Lock: it is the same mutual exclusion plus
        # the wait/notify single-flight needs, so there is still one lock here.
        self._lock = threading.Condition()
        self._entries: dict[Hashable, tuple[float, Any]] = {}
        self._producing: set[Hashable] = set()
        self._refreshers: list[threading.Thread] = []

    def get(
        self, key: Hashable, produce: Callable[[], Any], *, serve_stale_for: float = 0.0
    ) -> Any:
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

        ``serve_stale_for``: an entry past the TTL but within this many further
        seconds is returned AT ONCE and one refresh is started behind it. A
        polling page then never waits on a sweep; what it shows is at most
        ``ttl`` plus one sweep old, and says so through its own timestamp.
        Beyond the grace the call blocks as before, so a refresh that keeps
        failing reaches the caller as a failure within ``ttl + serve_stale_for``.
        """
        while True:
            now = time.monotonic()
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    stored_at, value = entry
                    if now - stored_at < self._ttl:
                        return value
                    if now - stored_at < self._ttl + serve_stale_for:
                        if key not in self._producing:
                            self._producing.add(key)
                            self._refresh_behind(key, produce)
                        return value
                if key in self._producing:
                    self._lock.wait()
                    continue
                self._producing.add(key)
                break
        return self._produce(key, produce)

    def _produce(self, key: Hashable, produce: Callable[[], Any]) -> Any:
        """Run ``produce`` for a key already marked as producing, and store it."""
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

    def _refresh_behind(self, key: Hashable, produce: Callable[[], Any]) -> None:
        """Produce ``key`` on a thread of its own. Caller holds the lock.

        Under a COPY of the caller's context, so a command's telemetry is filed
        under the surface that asked for it rather than a bare thread's default
        -- the same trap `_compose._bound` documents. A failure is logged and
        otherwise dropped: the stale entry stays, and the next request past the
        grace is the one that reports it.
        """
        context = contextvars.copy_context()

        def _run() -> None:
            try:
                context.run(self._produce, key, produce)
            except Exception:  # noqa: BLE001 -- a thread has nobody to raise to
                logger.warning("background refresh of %r failed", key, exc_info=True)

        thread = threading.Thread(target=_run, name=f"refresh:{key!r}", daemon=True)
        self._refreshers = [t for t in self._refreshers if t.is_alive()]
        self._refreshers.append(thread)
        thread.start()

    def drain(self, timeout: float | None = None) -> None:
        """Wait for every background refresh in flight. For tests and shutdown."""
        with self._lock:
            pending = list(self._refreshers)
        for thread in pending:
            thread.join(timeout)

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
