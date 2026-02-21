"""Coordinator and lifecycle management for parallel MCCFR workers."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import uuid
from pathlib import Path
from typing import Any

from src.bucketing.utils.infoset import InfoSetKey
from src.solver.storage.shared_array import SharedArrayStorage
from src.training.parallel_protocol import JobType
from src.training.parallel_worker import _worker_loop
from src.utils.config import Config


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

    def __init__(
        self,
        num_workers: int,
        config: "Config",
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
        self.capacity = initial_capacity  # Current capacity (grows via resize)
        self.max_actions = max_actions
        self.checkpoint_dir = checkpoint_dir

        # Create synchronization event for shared memory readiness
        # Coordinator sets this after creating memory, workers wait for it
        self.ready_event = mp.Event()

        # Create coordinator storage (creates shared memory)
        print(
            f"[Master] Creating shared memory (session={self.session_id})...",
            flush=True,
        )
        self.storage: SharedArrayStorage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,  # Coordinator uses worker_id 0 for ownership checks
            session_id=self.session_id,
            initial_capacity=initial_capacity,
            max_actions=max_actions,
            is_coordinator=True,  # Create shared memory
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            ready_event=self.ready_event,  # Coordinator signals when memory is ready
            load_checkpoint_on_init=False,
            zarr_compression_level=config.storage.zarr_compression_level,
            zarr_chunk_size=config.storage.zarr_chunk_size,
        )
        total_mb = initial_capacity * max_actions * 4 * 2 // 1024 // 1024
        print(f"[Master] Shared memory created: {total_mb}MB total", flush=True)

        # Communication queues
        self.job_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()

        # Per-worker queues for cross-partition updates
        self.update_queues: list[mp.Queue] = [mp.Queue() for _ in range(num_workers)]

        # Per-worker queues for ID requests/responses
        self.id_request_queues: list[mp.Queue] = [mp.Queue() for _ in range(num_workers)]
        self.id_response_queues: list[mp.Queue] = [mp.Queue() for _ in range(num_workers)]

        # Worker processes
        self.processes: list[mp.Process] = []
        self._fallback_stats_by_worker: dict[int, dict[str, float]] = {}

        # Start workers
        self._start_workers()

    def _start_workers(self):
        """Start all worker processes."""
        print(f"[Master] Starting {self.num_workers} workers...", flush=True)

        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    self.num_workers,
                    self.session_id,
                    self.config,
                    self.serialized_action_model,
                    self.serialized_card_abstraction,
                    self.base_seed,
                    self.job_queue,
                    self.result_queue,
                    self.update_queues,
                    self.id_request_queues,
                    self.id_response_queues,
                    self.capacity,
                    self.max_actions,
                    self.checkpoint_dir,
                    self.ready_event,  # Workers wait on this before attaching
                ),
            )
            p.start()
            self.processes.append(p)

        # Workers will wait for ready_event before attaching to shared memory
        # No need for arbitrary sleep - synchronization is handled by the event
        print(f"[Master] All {self.num_workers} workers started", flush=True)

    def exchange_ids(self, timeout: float = 60.0, verbose: bool = True) -> dict[str, Any]:
        """
        Trigger batched ID exchange between workers.

        Workers send pending ID requests to owners and process responses.
        This is owner-to-requester communication, not global broadcast.

        Returns:
            Dict with exchange stats
        """
        # Tell all workers to exchange IDs
        for _ in range(self.num_workers):
            self.job_queue.put({"type": JobType.EXCHANGE_IDS.value})

        # Wait for acks
        acks = []
        total_owned = 0
        for _ in range(self.num_workers):
            try:
                result = self.result_queue.get(timeout=timeout)
                if result.get("type") == "exchange_ids_ack":
                    acks.append(result)
                    total_owned += result.get("num_owned", 0)
            except queue.Empty:
                raise RuntimeError("Timeout waiting for ID exchange acks")

        if verbose:
            print("[Master] ID exchange complete", flush=True)

        return {"total_owned": total_owned, "acks": acks}

    def apply_pending_updates(self, timeout: float = 60.0, verbose: bool = True) -> dict[str, Any]:
        """
        Trigger workers to apply any pending cross-partition updates.

        Returns:
            Dict with apply stats
        """
        for _ in range(self.num_workers):
            self.job_queue.put({"type": JobType.APPLY_UPDATES.value})

        acks = []
        for _ in range(self.num_workers):
            try:
                result = self.result_queue.get(timeout=timeout)
                if result.get("type") == "apply_updates_ack":
                    acks.append(result)
            except queue.Empty:
                raise RuntimeError("Timeout waiting for apply updates acks")

        if verbose:
            print("[Master] Pending updates applied", flush=True)

        return {"acks": acks}

    def check_and_resize_if_needed(
        self,
        max_worker_capacity: float,
        timeout: float = 120.0,
        verbose: bool = True,
    ) -> bool:
        """
        Check if storage needs resizing and perform resize if necessary.

        This is a stop-the-world operation:
        1. Check if ANY worker is approaching capacity (85% of their local range)
        2. If yes, coordinator resizes storage (2x growth)
        3. Workers reattach to new shared memory

        Args:
            max_worker_capacity: Highest capacity usage among all workers (0.0 to 1.0)
            timeout: Max wait time for worker responses
            verbose: Print progress messages

        Returns:
            True if resize was performed, False otherwise
        """
        if max_worker_capacity < self.storage.CAPACITY_THRESHOLD:
            return False

        new_max = int(self.capacity * self.storage.GROWTH_FACTOR)

        if verbose:
            print(
                "[Master] Storage resize triggered: "
                f"worker at {max_worker_capacity:.1%} capacity, "
                f"growing {self.capacity:,} -> {new_max:,}",
                flush=True,
            )

        return self.resize_storage(new_max, timeout=timeout, verbose=verbose)

    def resize_storage(
        self,
        new_capacity: int,
        timeout: float = 120.0,
        verbose: bool = True,
    ) -> bool:
        """
        Resize storage to new capacity (stop-the-world operation).

        1. Coordinator creates new larger shared memory
        2. Copies existing data to new arrays
        3. Notifies workers to reattach
        4. Waits for all workers to confirm

        Args:
            new_capacity: New capacity for infoset storage
            timeout: Max wait time for worker responses
            verbose: Print progress messages

        Returns:
            True if resize was successful
        """
        resize_start = time.time()

        if verbose:
            print(
                f"[Master] Resizing storage: {self.storage.capacity:,} -> {new_capacity:,}",
                flush=True,
            )

        # Step 1: Coordinator resizes (creates new shared memory, copies data)
        self.storage.resize(new_capacity)
        new_session_id = self.storage.session_id

        # Update our tracking of initial_capacity
        self.capacity = new_capacity

        if verbose:
            print(
                f"[Master] New shared memory created (session={new_session_id}), "
                "notifying workers...",
                flush=True,
            )

        # Step 2: Tell all workers to reattach
        # Include old_capacity so workers can calculate additional slots
        # old_capacity = self.storage.capacity  # This is now new_capacity after resize
        # Actually we need to pass the old value, which we can compute from the difference
        # Since we just did 2x, old = new / 2
        for _ in range(self.num_workers):
            self.job_queue.put(
                {
                    "type": JobType.RESIZE_STORAGE.value,
                    "new_session_id": new_session_id,
                    "new_capacity": new_capacity,
                }
            )

        # Step 3: Wait for all workers to confirm
        acks = []
        for _ in range(self.num_workers):
            try:
                result = self.result_queue.get(timeout=timeout)
                if result.get("type") == "resize_ack":
                    acks.append(result)
                    if verbose:
                        worker_id = result["worker_id"]
                        new_range = result["new_id_range"]
                        print(
                            f"[Master] Worker {worker_id} reattached (range={new_range})",
                            flush=True,
                        )
            except queue.Empty:
                raise RuntimeError(
                    f"Timeout waiting for resize acks ({len(acks)}/{self.num_workers} received)"
                )

        resize_time = time.time() - resize_start

        if verbose:
            print(
                f"[Master] Resize complete in {resize_time:.1f}s, "
                f"new capacity: {new_capacity:,} infosets",
                flush=True,
            )

        return True

    def run_batch(
        self,
        iterations_per_worker: list[int],
        batch_id: int = 0,
        start_iteration: int = 0,
        timeout: float = 600.0,
        verbose: bool = True,
        auto_resize: bool = True,
    ) -> dict[str, Any]:
        """
        Run a batch of iterations across all workers.

        Workers:
        - Write directly to shared arrays for owned infosets
        - Get UNKNOWN_ID (zeros) for non-owned undiscovered keys
        - Send cross-partition updates via queues

        No global key synchronization - ownership determined by stable hash.

        Args:
            iterations_per_worker: Iterations per worker
            batch_id: Batch identifier
            start_iteration: Global iteration offset
            timeout: Max wait time
            verbose: Print progress
            auto_resize: If True, check capacity and resize after batch if needed

        Returns:
            Dict with batch results and utilities (includes 'resized': bool)
        """
        batch_start = time.time()

        # Submit jobs
        iteration_offset = start_iteration
        active_workers = 0
        for worker_id, num_iterations in enumerate(iterations_per_worker):
            if num_iterations > 0:
                self.job_queue.put(
                    {
                        "type": JobType.RUN_ITERATIONS.value,
                        "num_iterations": num_iterations,
                        "batch_id": batch_id,
                        "iteration_offset": iteration_offset,
                    }
                )
                iteration_offset += num_iterations
                active_workers += 1

        # Collect results
        all_utilities = []
        results = []
        errors = []
        total_owned = 0
        max_worker_capacity = 0.0
        interrupted = False

        received = 0
        while received < active_workers:
            try:
                result = self.result_queue.get(timeout=timeout)

                if "error" in result:
                    errors.append(result)
                    print(
                        f"[Master] Worker {result['worker_id']} error: {result['error']}",
                        flush=True,
                    )
                    received += 1
                elif result.get("type") == "iterations_done":
                    results.append(result)
                    all_utilities.extend(result.get("utilities", []))
                    total_owned += result.get("num_owned_infosets", 0)
                    worker_capacity = result.get("capacity_usage", 0.0)
                    max_worker_capacity = max(max_worker_capacity, worker_capacity)
                    fallback_stats = result.get("fallback_stats")
                    if isinstance(fallback_stats, dict):
                        self._fallback_stats_by_worker[result["worker_id"]] = fallback_stats

                    received += 1
                else:
                    if verbose:
                        print(f"[Master] Ignoring unexpected result: {result}", flush=True)

            except queue.Empty:
                raise RuntimeError(
                    f"Timeout waiting for workers ({len(results)}/{active_workers} received)"
                )
            except KeyboardInterrupt:
                interrupted = True
                if verbose:
                    print(
                        "⚠️  Interrupt received; waiting for current batch to finish...",
                        flush=True,
                    )

        batch_time = time.time() - batch_start
        total_iters = sum(iterations_per_worker)

        if verbose:
            print(
                f"[Master] Batch {batch_id} complete in {batch_time:.1f}s "
                f"({total_iters / batch_time:.1f} iter/s)",
                flush=True,
            )

        if errors:
            # Format detailed error message with worker details and tracebacks
            error_details = "\n".join(
                [
                    f"  Worker {e['worker_id']}:\n"
                    f"    Error: {e['error']}\n"
                    f"    Traceback:\n{e.get('traceback', '    (no traceback available)')}"
                    for e in errors
                ]
            )
            raise RuntimeError(
                f"{len(errors)} worker(s) failed during batch {batch_id}:\n{error_details}"
            )

        # Check if resize is needed after batch (based on per-worker capacity)
        resized = False
        if auto_resize:
            resized = self.check_and_resize_if_needed(
                max_worker_capacity=max_worker_capacity,
                timeout=timeout,
                verbose=verbose,
            )

        total_lookups = sum(
            stats.get("total_lookups", 0) for stats in self._fallback_stats_by_worker.values()
        )
        fallback_count = sum(
            stats.get("fallback_count", 0) for stats in self._fallback_stats_by_worker.values()
        )
        fallback_stats = {
            "total_lookups": total_lookups,
            "fallback_count": fallback_count,
            "fallback_rate": (fallback_count / total_lookups) if total_lookups > 0 else 0.0,
        }

        return {
            "utilities": all_utilities,
            "batch_time": batch_time,
            "total_iterations": total_iters,
            "worker_results": results,
            "num_infosets": total_owned,
            "max_worker_capacity": max_worker_capacity,
            "resized": resized,
            "capacity": self.capacity,
            "interrupted": interrupted,
            "fallback_stats": fallback_stats,
        }

    def collect_keys(self, timeout: float = 60.0) -> dict[str, Any]:
        """
        Collect owned keys from all workers for checkpointing.

        Workers send their key→ID mappings to coordinator so the
        coordinator can save a complete checkpoint.

        Returns:
            Dict with aggregated key mappings
        """
        # Tell each worker to send their keys
        for worker_id in range(self.num_workers):
            self.job_queue.put(
                {
                    "type": JobType.COLLECT_KEYS.value,
                    "target_worker": worker_id,
                }
            )

        # Collect responses
        all_owned_keys: dict = {}
        all_legal_actions: dict = {}
        worker_ranges: dict[int, tuple[int, int]] = {}
        id_owners: dict[int, tuple[int, "InfoSetKey"]] = {}

        responses_received = 0
        while responses_received < self.num_workers:
            try:
                result = self.result_queue.get(timeout=timeout)
                if result.get("type") == "keys_collected":
                    owned_keys = result["owned_keys"]
                    legal_actions = result["legal_actions_cache"]
                    worker_id = result["worker_id"]
                    worker_ranges[worker_id] = (
                        result["id_range_start"],
                        result["id_range_end"],
                    )

                    # Merge into coordinator's mappings
                    for key, infoset_id in owned_keys.items():
                        if infoset_id in id_owners:
                            prev_worker, prev_key = id_owners[infoset_id]
                            raise RuntimeError(
                                f"Duplicate infoset_id {infoset_id} across workers "
                                f"(prev worker {prev_worker}, key {prev_key} vs "
                                f"worker {worker_id}, key {key}). "
                                "ID ranges likely overlapping or num_workers changed."
                            )
                        id_owners[infoset_id] = (worker_id, key)
                        all_owned_keys[key] = infoset_id
                    all_legal_actions.update(legal_actions)
                    responses_received += 1
                # Ignore other message types that might be in the queue
            except queue.Empty:
                raise RuntimeError(
                    f"Timeout waiting for key collection ({responses_received}/{self.num_workers} received)"
                )

        # Validate worker ID ranges don't overlap (defensive)
        ranges = sorted(worker_ranges.items(), key=lambda kv: kv[1][0])
        for (wid_a, (start_a, end_a)), (wid_b, (start_b, end_b)) in zip(ranges, ranges[1:]):
            if start_b < end_a:
                raise RuntimeError(
                    f"Worker ID ranges overlap: worker {wid_a} [{start_a},{end_a}) "
                    f"and worker {wid_b} [{start_b},{end_b}). "
                    "This will corrupt checkpoints; ensure consistent num_workers/initial_capacity."
                )

        return {
            "owned_keys": all_owned_keys,
            "legal_actions_cache": all_legal_actions,
        }

    def checkpoint(self, iteration: int):
        """
        Save checkpoint to disk.

        Collects keys from all workers first, then saves the complete state.
        """
        # Collect keys from all workers
        collected = self.collect_keys()

        # Merge into coordinator's storage for checkpointing
        self.storage.state.owned_keys = collected["owned_keys"]
        self.storage.state.legal_actions_cache = collected["legal_actions_cache"]

        # Update next_local_id to reflect total infosets
        if self.storage.state.owned_keys:
            max_id = max(self.storage.state.owned_keys.values())
            self.storage.state.next_local_id = max_id + 1

        if self.config.training.verbose:
            print(
                f"[Master] Checkpointing iter={iteration}: "
                f"{len(self.storage.state.owned_keys):,} infosets "
                f"(max_id={self.storage.state.next_local_id})...",
                flush=True,
            )

        # Now checkpoint with complete key mappings
        self.storage.checkpoint(iteration)

    def get_storage(self) -> SharedArrayStorage:
        """Get coordinator's storage instance (for accessing results)."""
        return self.storage

    def shutdown(self):
        """Shutdown all workers cleanly."""
        print("[Master] Shutting down workers...", flush=True)

        # Send shutdown to all workers
        for _ in range(self.num_workers):
            self.job_queue.put({"type": JobType.SHUTDOWN.value})

        # Wait for processes to exit
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                print(f"[Master] Force terminating worker {p.pid}", flush=True)
                p.terminate()

        self.processes.clear()

        # Cleanup shared memory (coordinator unlinks)
        self.storage.cleanup()

        print("[Master] All workers shut down", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
