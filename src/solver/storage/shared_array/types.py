"""Typed records shared by shared-array storage collaborators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.bucketing.utils.infoset import InfoSetKey
from src.game.actions import Action

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory


@runtime_checkable
class AllocationLike(Protocol):
    """Protocol for allocation records."""

    start: int
    end: int
    next: int


@runtime_checkable
class RegionLike(Protocol):
    """Protocol for resize region records."""

    start: int
    total: int
    base: int
    remainder: int


@dataclass(slots=True)
class ExtraAllocation:
    """ID allocation interval assigned to a specific worker."""

    start: int
    end: int
    next: int

    def __getitem__(self, key: str) -> int:
        """Backward-compatible dict-like access for tests/callers."""
        return getattr(self, key)

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

    def __iter__(self):
        """Backward-compatible tuple-like behavior for tests/callers."""
        yield self.start
        yield self.total
        yield self.base
        yield self.remainder


@dataclass(slots=True)
class SharedArrayMutableState:
    """Mutable runtime state for SharedArrayStorage internals."""

    next_local_id: int
    owned_keys: dict[InfoSetKey, int] = field(default_factory=dict)
    remote_keys: dict[InfoSetKey, int] = field(default_factory=dict)
    legal_actions_cache: dict[int, list[Action]] = field(default_factory=dict)
    pending_id_requests: dict[int, set[InfoSetKey]] = field(default_factory=dict)
    extra_regions: list[ExtraRegion] = field(default_factory=list)
    extra_allocations: list[ExtraAllocation] = field(default_factory=list)
    shm_regrets: SharedMemory | None = None
    shm_strategy: SharedMemory | None = None
    shm_actions: SharedMemory | None = None
    shm_reach: SharedMemory | None = None
    shm_utility: SharedMemory | None = None


@dataclass(slots=True)
class PendingUpdate:
    """Typed container for one infoset's cross-partition deltas."""

    regret_delta: np.ndarray
    strategy_delta: np.ndarray


class PendingUpdateQueue:
    """Accumulates cross-partition updates with shape validation."""

    def __init__(self):
        self._updates: dict[int, PendingUpdate] = {}

    def buffer(
        self,
        infoset_id: int,
        regret_delta: np.ndarray,
        strategy_delta: np.ndarray,
        *,
        expected_actions: int | None = None,
    ) -> None:
        if regret_delta.ndim != 1 or strategy_delta.ndim != 1:
            raise ValueError("Pending updates must use 1-D regret/strategy arrays")
        if regret_delta.shape != strategy_delta.shape:
            raise ValueError(
                f"Pending update shape mismatch: {regret_delta.shape} vs {strategy_delta.shape}"
            )
        if expected_actions is not None and (
            len(regret_delta) < expected_actions or len(strategy_delta) < expected_actions
        ):
            raise ValueError(
                f"Pending update for infoset {infoset_id} has insufficient actions: "
                f"expected at least {expected_actions}, got {len(regret_delta)}"
            )

        if infoset_id in self._updates:
            current = self._updates[infoset_id]
            self._updates[infoset_id] = PendingUpdate(
                regret_delta=current.regret_delta + regret_delta,
                strategy_delta=current.strategy_delta + strategy_delta,
            )
            return

        self._updates[infoset_id] = PendingUpdate(
            regret_delta=regret_delta.copy(),
            strategy_delta=strategy_delta.copy(),
        )

    def snapshot(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return {
            infoset_id: (update.regret_delta, update.strategy_delta)
            for infoset_id, update in self._updates.items()
        }

    def clear(self) -> None:
        self._updates.clear()
