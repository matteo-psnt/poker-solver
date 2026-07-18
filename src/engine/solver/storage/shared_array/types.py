"""Typed records shared by shared-array storage collaborators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSetKey

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory


@dataclass(slots=True)
class ExtraAllocation:
    """ID allocation interval assigned to a specific worker."""

    start: int
    end: int
    next: int

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def used(self) -> int:
        return self.next - self.start

    def contains(self, infoset_id: int) -> bool:
        return self.start <= infoset_id < self.end


@dataclass(slots=True)
class ExtraRegion:
    """Metadata for a capacity expansion region split across workers."""

    start: int
    total: int
    base: int
    remainder: int


@dataclass(slots=True)
class SharedArrayMutableState:
    """Mutable runtime state for SharedArrayStorage internals."""

    next_local_id: int
    owned_keys: dict[InfoSetKey, int] = field(default_factory=dict)
    remote_keys: dict[InfoSetKey, int] = field(default_factory=dict)
    legal_actions_cache: dict[int, list[Action]] = field(default_factory=dict)
    pending_id_requests: dict[int, set[InfoSetKey]] = field(default_factory=dict)
    # Keys already sent to their owner and awaiting a response. Gates re-adding
    # to pending_id_requests: without it, every visit to a hot unresolved key
    # re-queues it and each flush re-pickles the same keys to the same owner —
    # enough traffic to collapse multi-worker scaling. Re-armed (moved back to
    # pending) at batch boundaries as a lost-message backstop.
    requested_id_keys: set[InfoSetKey] = field(default_factory=set)
    # Owner side: requesters of keys this worker owns but has not allocated yet.
    # Answered proactively at allocation time (via pending_late_responses), so a
    # request never waits on the requester retrying.
    unanswered_id_requests: dict[InfoSetKey, set[int]] = field(default_factory=dict)
    # Owner side: allocation-time responses queued per requester, flushed with
    # the regular sync cadence.
    pending_late_responses: dict[int, dict[InfoSetKey, int]] = field(default_factory=dict)
    extra_regions: list[ExtraRegion] = field(default_factory=list)
    extra_allocations: list[ExtraAllocation] = field(default_factory=list)
    shm_regrets: SharedMemory | None = None
    shm_strategy: SharedMemory | None = None
    shm_actions: SharedMemory | None = None
    shm_reach: SharedMemory | None = None
    shm_utility: SharedMemory | None = None
