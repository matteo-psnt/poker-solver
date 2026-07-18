"""Shared-array storage backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.base import Storage
from src.engine.solver.storage.shared_array.checkpoint import (
    checkpoint_storage,
    load_storage_checkpoint,
)
from src.engine.solver.storage.shared_array.ownership import (
    owner_for_key as _owner_for_key,
)
from src.engine.solver.storage.shared_array.ownership import (
    stable_hash as _stable_hash,
)
from src.engine.solver.storage.shared_array.types import SharedArrayMutableState

from . import infoset as infoset_ops
from . import memory as memory_ops
from . import resize as resize_ops
from . import sync as sync_ops

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType


class SharedArrayStorage(Storage):
    """Partitioned storage using flat NumPy arrays in shared memory.

    The backing arrays are declared once in
    :data:`~src.engine.solver.storage.shared_array.specs.ARRAY_SPECS`; the
    annotations below only give the spec-driven ``setattr`` loops static types.
    """

    shared_regrets: np.ndarray
    shared_strategy_sum: np.ndarray
    shared_action_counts: np.ndarray
    shared_reach_counts: np.ndarray
    shared_cumulative_utility: np.ndarray

    UNKNOWN_ID = 0
    CAPACITY_THRESHOLD = 0.85
    GROWTH_FACTOR = 2.0

    get_shm_name = memory_ops.get_shm_name_for_storage
    create_shared_memory = memory_ops.create_shared_memory
    attach_shared_memory = memory_ops.attach_shared_memory
    create_numpy_views = memory_ops.create_numpy_views
    cleanup_stale_shm = memory_ops.cleanup_stale_shm
    cleanup = memory_ops.cleanup

    get_or_create_infoset = infoset_ops.get_or_create_infoset
    allocate_id = infoset_ops.allocate_id
    create_infoset_view = infoset_ops.create_infoset_view
    get_infoset = infoset_ops.get_infoset
    num_infosets = infoset_ops.num_infosets
    iter_infosets = infoset_ops.iter_infosets
    num_owned_infosets = infoset_ops.num_owned_infosets

    get_capacity_usage = resize_ops.get_capacity_usage
    needs_resize = resize_ops.needs_resize
    get_resize_stats = resize_ops.get_resize_stats
    resize = resize_ops.resize
    reattach_after_resize = resize_ops.reattach_after_resize
    add_extra_region = resize_ops.add_extra_region

    respond_to_id_requests = sync_ops.respond_to_id_requests
    rearm_unresolved_id_requests = sync_ops.rearm_unresolved_id_requests

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
        *,
        zarr_compression_level: int,
        zarr_chunk_size: int,
    ):
        """Storage over shared memory.

        ``zarr_*`` are required rather than defaulted: they are owned by
        ``StorageConfig`` and only ever passed through. A default here would be a
        second source of truth that silently wins whenever a caller forgets to
        forward the loaded config — which is exactly how single-worker training
        came to ignore both knobs.
        """
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

        for spec in ARRAY_SPECS:
            setattr(self, spec.attr, np.empty(spec.shape(0, self.max_actions), dtype=spec.dtype))

        if is_coordinator:
            self.create_shared_memory()
        else:
            self.attach_shared_memory()

        if checkpoint_dir and load_checkpoint_on_init:
            self.load_checkpoint()

    def _stable_hash(self, key: InfoSetKey) -> int:
        """Compute stable hash of InfoSetKey."""
        return _stable_hash(key)

    def get_owner(self, key: InfoSetKey) -> int:
        """Determine which worker owns this key."""
        return _owner_for_key(key, self.num_workers)

    def close(self) -> None:
        """Explicitly close shared-memory handles."""
        self.cleanup()

    def __enter__(self) -> SharedArrayStorage:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.cleanup()
        return False

    def __str__(self) -> str:
        return (
            f"SharedArrayStorage(worker={self.worker_id}, "
            f"coordinator={self.is_coordinator}, "
            f"owned={self.num_owned_infosets()}, "
            f"id_range=[{self.id_range_start}, {self.id_range_end}))"
        )
