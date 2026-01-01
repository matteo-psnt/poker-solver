"""Shared-array storage backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage.base import Storage
from src.engine.solver.storage.shared_array.checkpoint import (
    checkpoint_storage,
    load_storage_checkpoint,
)
from src.engine.solver.storage.shared_array.ownership import (
    is_owned_by_id as _is_owned_by_id,
)
from src.engine.solver.storage.shared_array.ownership import (
    owner_for_id as _owner_for_id,
)
from src.engine.solver.storage.shared_array.ownership import (
    owner_for_key as _owner_for_key,
)
from src.engine.solver.storage.shared_array.ownership import (
    stable_hash as _stable_hash,
)
from src.engine.solver.storage.shared_array.types import (
    PendingUpdateQueue,
    SharedArrayMutableState,
)

from . import infoset as infoset_ops
from . import memory as memory_ops
from . import resize as resize_ops
from . import sync as sync_ops

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType


class SharedArrayStorage(Storage):
    """Partitioned storage using flat NumPy arrays in shared memory."""

    SHM_REGRETS = "sas_reg"
    SHM_STRATEGY = "sas_str"
    SHM_ACTIONS = "sas_act"
    SHM_REACH = "sas_reach"
    SHM_UTILITY = "sas_util"

    UNKNOWN_ID = 0
    CAPACITY_THRESHOLD = 0.85
    GROWTH_FACTOR = 2.0

    _get_shm_name = memory_ops.get_shm_name_for_storage
    _create_shared_memory = memory_ops.create_shared_memory
    _attach_shared_memory = memory_ops.attach_shared_memory
    _create_numpy_views = memory_ops.create_numpy_views
    _cleanup_stale_shm = memory_ops.cleanup_stale_shm
    cleanup = memory_ops.cleanup

    get_or_create_infoset = infoset_ops.get_or_create_infoset
    _allocate_id = infoset_ops.allocate_id
    _create_infoset_view = infoset_ops.create_infoset_view
    get_infoset = infoset_ops.get_infoset
    num_infosets = infoset_ops.num_infosets
    iter_infosets = infoset_ops.iter_infosets
    num_owned_infosets = infoset_ops.num_owned_infosets

    get_capacity_usage = resize_ops.get_capacity_usage
    needs_resize = resize_ops.needs_resize
    get_resize_stats = resize_ops.get_resize_stats
    resize = resize_ops.resize
    reattach_after_resize = resize_ops.reattach_after_resize
    _add_extra_region = resize_ops.add_extra_region

    respond_to_id_requests = sync_ops.respond_to_id_requests
    buffer_update = sync_ops.buffer_update
    apply_updates = sync_ops.apply_updates

    checkpoint = checkpoint_storage
    load_checkpoint = load_storage_checkpoint

    def __init__(
        self,
        num_workers: int,
        worker_id: int,
        session_id: str,
        initial_capacity: int = 2_000_000,
        max_actions: int = 10,
        is_coordinator: bool = False,
        checkpoint_dir: Path | None = None,
        ready_event: EventType | None = None,
        load_checkpoint_on_init: bool = True,
        zarr_compression_level: int = 3,
        zarr_chunk_size: int = 10_000,
    ):
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.session_id = session_id[:8]
        self.capacity = initial_capacity
        self.max_actions = max_actions
        self.is_coordinator = is_coordinator
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.ready_event = ready_event
        self.base_capacity = initial_capacity
        self.base_slots_per_worker = (initial_capacity - 1) // num_workers

        self.zarr_compression_level = zarr_compression_level
        self.zarr_chunk_size = zarr_chunk_size

        usable_slots = initial_capacity - 1
        slots_per_worker = usable_slots // num_workers
        self.id_range_start = 1 + worker_id * slots_per_worker
        self.id_range_end = 1 + (worker_id + 1) * slots_per_worker
        self.state = SharedArrayMutableState(
            next_local_id=self.id_range_start,
            pending_id_requests={i: set() for i in range(num_workers)},
        )
        self.update_queue = PendingUpdateQueue()

        self.shared_regrets = np.empty((0, self.max_actions), dtype=np.float64)
        self.shared_strategy_sum = np.empty((0, self.max_actions), dtype=np.float64)
        self.shared_action_counts = np.empty((0,), dtype=np.int32)
        self.shared_reach_counts = np.empty((0,), dtype=np.int64)
        self.shared_cumulative_utility = np.empty((0,), dtype=np.float64)

        if is_coordinator:
            self._create_shared_memory()
        else:
            self._attach_shared_memory()

        if checkpoint_dir and load_checkpoint_on_init:
            self.load_checkpoint()

    def _stable_hash(self, key: InfoSetKey) -> int:
        """Compute stable hash of InfoSetKey."""
        return _stable_hash(key)

    def get_owner(self, key: InfoSetKey) -> int:
        """Determine which worker owns this key."""
        return _owner_for_key(key, self.num_workers)

    def is_owned(self, key: InfoSetKey) -> bool:
        """Check if this worker owns the given infoset key."""
        return self.get_owner(key) == self.worker_id

    def is_owned_by_id(self, infoset_id: int) -> bool:
        """Check if this worker owns the given infoset ID."""
        return _is_owned_by_id(
            infoset_id=infoset_id,
            unknown_id=self.UNKNOWN_ID,
            id_range_start=self.id_range_start,
            id_range_end=self.id_range_end,
            extra_allocations=self.state.extra_allocations,
        )

    def get_owner_by_id(self, infoset_id: int) -> int | None:
        """Determine owner worker for a given infoset ID."""
        return _owner_for_id(
            infoset_id,
            unknown_id=self.UNKNOWN_ID,
            base_slots_per_worker=self.base_slots_per_worker,
            num_workers=self.num_workers,
            extra_regions=self.state.extra_regions,
        )

    def close(self) -> None:
        """Explicitly close shared-memory handles."""
        self.cleanup()

    def __enter__(self) -> SharedArrayStorage:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False

    def __str__(self) -> str:
        return (
            f"SharedArrayStorage(worker={self.worker_id}, "
            f"coordinator={self.is_coordinator}, "
            f"owned={self.num_owned_infosets()}, "
            f"id_range=[{self.id_range_start}, {self.id_range_end}))"
        )
