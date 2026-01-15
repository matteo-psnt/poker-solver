"""Coordinator and lifecycle management for parallel MCCFR workers."""

from __future__ import annotations

import multiprocessing as mp
import uuid
from typing import TYPE_CHECKING, Protocol

from src.engine.solver.storage.shared_array import SharedArrayStorage
from src.shared.config import Config

from . import batch_ops, checkpoint_ops, lifecycle, resize_ops, sync_ops

if TYPE_CHECKING:
    from .batch_ops import BatchResult
    from .checkpoint_ops import CollectedKeys


class MessageQueue(Protocol):
    """Minimal queue interface used by the manager and helper modules."""

    def put(self, item: object) -> None: ...

    def get(self, timeout: float | None = None) -> object: ...


class SharedArrayWorkerManager:
    """
    Manages parallel training with shared array storage.

    ARCHITECTURE:
    - Coordinator creates shared memory once at startup
    - Workers attach to shared memory
    - Ownership by stable hash (xxhash), not Python hash()
    - No global key synchronization
    - ID requests/responses flow directly between workers (batched)
    - Cross-partition updates via per-worker queues
    """

    # Runtime attributes initialized during lifecycle initialization.
    ready_event: object
    storage: SharedArrayStorage
    job_queue: MessageQueue
    result_queue: MessageQueue
    update_queues: list[MessageQueue]
    id_request_queues: list[MessageQueue]
    id_response_queues: list[MessageQueue]

    def __init__(
        self,
        num_workers: int,
        config: Config,
        serialized_action_model: bytes,
        serialized_card_abstraction: bytes,
        session_id: str | None = None,
        base_seed: int = 42,
        initial_capacity: int = 2_000_000,
        max_actions: int = 10,
        checkpoint_dir: str | None = None,
    ):
        """
        Initialize shared array worker manager.

        Args:
            num_workers: Number of workers
            config: Configuration object
            serialized_action_model: Pickled ActionModel
            serialized_card_abstraction: Pickled BucketingStrategy
            session_id: Unique session ID (auto-generated if None)
            base_seed: Base random seed
            initial_capacity: Initial capacity for infoset storage
            max_actions: Maximum actions per infoset
            checkpoint_dir: Optional checkpoint directory
        """
        self.num_workers = num_workers
        self.config = config
        self.serialized_action_model = serialized_action_model
        self.serialized_card_abstraction = serialized_card_abstraction
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.base_seed = base_seed
        self.capacity = initial_capacity
        self.max_actions = max_actions
        self.checkpoint_dir = checkpoint_dir

        self.processes: list[mp.Process] = []

        lifecycle.initialize_runtime(self)
        self._start_workers()

    def _start_workers(self):
        """Start all worker processes."""
        lifecycle.start_workers(self)

    def exchange_ids(self, timeout: float = 60.0, verbose: bool = True) -> dict[str, object]:
        """Trigger batched ID exchange between workers."""
        return sync_ops.exchange_ids(self, timeout=timeout, verbose=verbose)

    def apply_pending_updates(
        self, timeout: float = 60.0, verbose: bool = True
    ) -> dict[str, object]:
        """Trigger workers to apply any pending cross-partition updates."""
        return sync_ops.apply_pending_updates(self, timeout=timeout, verbose=verbose)

    def check_and_resize_if_needed(
        self,
        max_worker_capacity: float,
        timeout: float = 120.0,
        verbose: bool = True,
    ) -> bool:
        """Check if storage needs resizing and perform resize if necessary."""
        return resize_ops.check_and_resize_if_needed(
            self,
            max_worker_capacity=max_worker_capacity,
            timeout=timeout,
            verbose=verbose,
        )

    def resize_storage(
        self,
        new_capacity: int,
        timeout: float = 120.0,
        verbose: bool = True,
    ) -> bool:
        """Resize storage to new capacity (stop-the-world operation)."""
        return resize_ops.resize_storage(
            self, new_capacity=new_capacity, timeout=timeout, verbose=verbose
        )

    def run_batch(
        self,
        iterations_per_worker: list[int],
        batch_id: int = 0,
        start_iteration: int = 0,
        timeout: float = 600.0,
        verbose: bool = True,
        auto_resize: bool = True,
    ) -> BatchResult:
        """Run a batch of iterations across all workers."""
        return batch_ops.run_batch(
            self,
            iterations_per_worker=iterations_per_worker,
            batch_id=batch_id,
            start_iteration=start_iteration,
            timeout=timeout,
            verbose=verbose,
            auto_resize=auto_resize,
        )

    def collect_keys(self, timeout: float = 60.0) -> CollectedKeys:
        """Collect owned keys from all workers for checkpointing."""
        return checkpoint_ops.collect_keys(self, timeout=timeout)

    def checkpoint(self, iteration: int):
        """Save checkpoint to disk."""
        checkpoint_ops.checkpoint(self, iteration=iteration)

    def get_storage(self) -> SharedArrayStorage:
        """Get coordinator's storage instance (for accessing results)."""
        return self.storage

    def shutdown(self):
        """Shutdown all workers cleanly."""
        lifecycle.shutdown(self)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()
        return False
