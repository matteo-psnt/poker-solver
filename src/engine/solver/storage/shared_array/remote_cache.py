"""Bounded cache of infoset ids owned by other workers.

WHY BOUNDED. ``remote_keys`` caches key -> global id for infosets this worker
does NOT own, so it converges on holding the ENTIRE tree in every worker rather
than each worker's 1/W share. Total memory therefore scales O(N x W): adding
workers makes each worker bigger, so parallelism degrades the thing it is meant
to improve. Measured against real RSS at 315 B per entry:

    7.24M infosets, 8 workers   ->  1.99 GB per worker of remote keys alone
    node totals: 4w 10.1 GB | 8w 19.2 GB | 16w 37.4 GB

That is what killed three 16-worker legs at ~2.4M infosets on a 32 GB node, and
what would have killed the 8-worker leg around 10-11M.

WHAT AN EVICTION COSTS. A miss is not an error: the caller queues an id request
with the owner and the current update is dropped (``UNKNOWN_ID``). So eviction
trades memory for sample efficiency, against a drop rate that is already high.
Two consequences drive the design:

  * Eviction must be O(1) with NO per-hit bookkeeping. True LRU would reorder on
    every lookup, and lookups are the hot path -- paying a write on every read to
    improve eviction quality is the wrong trade here.
  * Evictions must be COUNTED. A cache thrashing just under its cap looks
    identical to a healthy one from the outside while quietly destroying sample
    efficiency, so the number goes into the metrics row.

Insertion-ordered eviction (FIFO) rather than LRU: Python dicts preserve
insertion order, so the oldest entry is ``next(iter(d))`` and eviction costs one
pop. CFR revisits infosets heavily, so a generous cap keeps the working set
regardless of policy; a cap small enough for FIFO-vs-LRU to matter is a cap that
is already too small, which the eviction counter will show.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.solver.infoset import InfoSetKey

# 0 disables the bound entirely (the historical behaviour), for A/B measurement.
UNBOUNDED = 0


class RemoteKeyCache:
    """Insertion-ordered, bounded map of remote InfoSetKey -> infoset id."""

    __slots__ = ("_data", "_evictions", "capacity")

    def __init__(self, capacity: int = UNBOUNDED) -> None:
        self.capacity = capacity
        self._data: dict[InfoSetKey, int] = {}
        self._evictions = 0

    @property
    def evictions(self) -> int:
        """Entries dropped to stay within capacity, cumulative."""
        return self._evictions

    def get(self, key: InfoSetKey) -> int | None:
        return self._data.get(key)

    def __getitem__(self, key: InfoSetKey) -> int:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[InfoSetKey]:
        return iter(self._data)

    def __setitem__(self, key: InfoSetKey, value: int) -> None:
        self._data[key] = value
        self._trim()

    def update(self, entries: Mapping[InfoSetKey, int]) -> None:
        self._data.update(entries)
        self._trim()

    def _trim(self) -> None:
        if self.capacity <= UNBOUNDED:
            return
        excess = len(self._data) - self.capacity
        if excess <= 0:
            return
        # Oldest-first: dicts iterate in insertion order, so these are the
        # entries resident longest. Materialised BEFORE deleting -- deleting
        # while iterating raises "dictionary changed size during iteration", and
        # a bulk `update` from the id-exchange overshoots by many entries at
        # once, so this path is not rare.
        victims = list(islice(iter(self._data), excess))
        for victim in victims:
            del self._data[victim]
        self._evictions += len(victims)
