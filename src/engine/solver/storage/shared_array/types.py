"""Typed records shared by shared-array storage collaborators."""

from __future__ import annotations

from collections.abc import Sequence
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
    """Mutable runtime state for SharedArrayStorage internals.

    ID-sync fields, whose invariants are not obvious from their types:

    ``unshipped_keys``
        Keys allocated since the last COLLECT_KEYS. Shares InfoSetKey objects
        with ``owned_keys``, so a teardown that clears one must clear both.
    ``requested_id_keys``
        Sent and awaiting a response. Gates re-queuing (without it, hot
        unresolved keys re-pickle every flush and multi-worker scaling
        collapses). Re-armed into ``pending_id_requests`` at batch boundaries
        as a lost-message backstop.
    ``unanswered_id_requests``
        Owner side: requesters waiting on keys this worker owns but hasn't
        allocated. Answered at allocation time via ``pending_late_responses``.
    ``pending_late_responses``
        Owner side: allocation-time responses per requester, flushed on the
        regular sync cadence.
    """

    next_local_id: int
    owned_keys: dict[InfoSetKey, int] = field(default_factory=dict)
    unshipped_keys: list[tuple[InfoSetKey, int]] = field(default_factory=list)
    remote_keys: dict[InfoSetKey, int] = field(default_factory=dict)
    legal_actions_cache: dict[int, Sequence[Action]] = field(default_factory=dict)
    pending_id_requests: dict[int, set[InfoSetKey]] = field(default_factory=dict)
    requested_id_keys: set[InfoSetKey] = field(default_factory=set)
    unanswered_id_requests: dict[InfoSetKey, set[int]] = field(default_factory=dict)
    pending_late_responses: dict[int, dict[InfoSetKey, int]] = field(default_factory=dict)
    extra_regions: list[ExtraRegion] = field(default_factory=list)
    extra_allocations: list[ExtraAllocation] = field(default_factory=list)
    shm_regrets: SharedMemory | None = None
    shm_strategy: SharedMemory | None = None
    shm_actions: SharedMemory | None = None
    shm_reach: SharedMemory | None = None
    shm_utility: SharedMemory | None = None
