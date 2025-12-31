"""
Parallel training infrastructure for MCCFR solver.

High-performance shared array architecture:
- Coordinator creates shared memory once at startup
- Workers attach to shared memory (no recreation)
- Flat NumPy arrays as live data structure
- Lock-free reads, owner-only writes
- Cross-partition updates via queues
"""

import multiprocessing as mp
import pickle
import queue
import random
import time
import traceback
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from src.solver.storage import SharedArrayStorage

import numpy as np


class JobType(Enum):
    """Job types for parallel training."""

    RUN_ITERATIONS = "run_iterations"
    SYNC_KEYS = "sync_keys"  # Sync key mappings across workers
    APPLY_UPDATES = "apply_updates"  # Apply cross-partition updates
    SHUTDOWN = "shutdown"


def _worker_loop(
    worker_id: int,
    num_workers: int,
    session_id: str,
    config_dict: Dict,
    serialized_action_abstraction: bytes,
    serialized_card_abstraction: bytes,
    base_seed: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    update_queues: List[mp.Queue],  # One queue per worker for cross-partition updates
    max_infosets: int,
    max_actions: int,
    checkpoint_dir: str | None = None,
) -> None:
    """
    Worker process for parallel MCCFR training with shared array storage.

    Workers:
    - Attach to shared memory created by coordinator
    - Own infosets where infoset_id % num_workers == worker_id
    - Write directly to shared arrays for owned infosets
    - Buffer cross-partition updates and send to owner via queue

    Args:
        worker_id: This worker's ID (0 to num_workers-1)
        num_workers: Total number of workers
        session_id: Unique session ID for shared memory namespace
        config_dict: Configuration dictionary
        serialized_action_abstraction: Pickled BettingActions
        serialized_card_abstraction: Pickled BucketingStrategy
        base_seed: Base random seed
        job_queue: Queue to receive jobs from coordinator
        result_queue: Queue to return results to coordinator
        update_queues: Per-worker queues for cross-partition updates
        max_infosets: Maximum infosets in shared arrays
        max_actions: Maximum actions per infoset
        checkpoint_dir: Optional checkpoint directory
    """
    try:
        import sys
        from pathlib import Path

        from src.solver.mccfr import MCCFRSolver
        from src.solver.storage import SharedArrayStorage
        from src.utils.config import Config

        print(
            f"[Worker {worker_id}] Initializing (attaching to shared memory)...",
            file=sys.stderr,
            flush=True,
        )

        # Create config
        config = Config.from_dict(config_dict)

        # Deserialize abstractions
        action_abstraction = pickle.loads(serialized_action_abstraction)
        card_abstraction = pickle.loads(serialized_card_abstraction)

        # Create storage (attach to existing shared memory)
        storage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=worker_id,
            session_id=session_id,
            max_infosets=max_infosets,
            max_actions=max_actions,
            is_coordinator=False,  # Worker attaches, doesn't create
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        )

        # Create solver config
        solver_config = config.get_section("game").copy()
        solver_config.update(config.get_section("system"))
        solver_config["seed"] = base_seed + worker_id * 10000

        # CFR+ can be applied directly since each worker owns its infosets
        cfr_plus_enabled = config_dict.get("solver", {}).get("cfr_plus", False)
        solver_config["cfr_plus"] = cfr_plus_enabled

        # Create solver with shared array storage
        solver = MCCFRSolver(
            action_abstraction=action_abstraction,
            card_abstraction=card_abstraction,
            storage=storage,
            config=solver_config,
        )

        print(
            f"[Worker {worker_id}] Ready (attached to shared memory)",
            file=sys.stderr,
            flush=True,
        )

        # Worker loop
        batch_count = 0
        while True:
            # Check for cross-partition updates from other workers
            _process_incoming_updates(worker_id, update_queues[worker_id], storage)

            try:
                job = job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            job_type = JobType(job["type"])

            if job_type == JobType.SHUTDOWN:
                print(
                    f"[Worker {worker_id}] Shutdown signal received",
                    file=sys.stderr,
                    flush=True,
                )
                break

            elif job_type == JobType.SYNC_KEYS:
                # Receive merged key mapping from coordinator
                merged_keys = job.get("key_mapping", {})
                storage.set_key_mapping(merged_keys)

                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "type": "sync_keys_ack",
                        "num_keys": len(merged_keys),
                    }
                )

            elif job_type == JobType.APPLY_UPDATES:
                # Process any pending incoming updates
                _process_incoming_updates(worker_id, update_queues[worker_id], storage)

                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "type": "apply_updates_ack",
                    }
                )

            elif job_type == JobType.RUN_ITERATIONS:
                num_iterations = job["num_iterations"]
                batch_id = job.get("batch_id", batch_count)
                iteration_offset = job.get("iteration_offset", 0)

                # Update solver state
                solver.iteration = iteration_offset

                # Update seed for this batch
                batch_seed = base_seed + worker_id * 10000 + batch_id * 1000
                random.seed(batch_seed)
                np.random.seed(batch_seed)

                # Run iterations
                print(
                    f"[Worker {worker_id}] Running {num_iterations} iterations...",
                    file=sys.stderr,
                    flush=True,
                )

                utilities = []
                iter_start = time.time()

                try:
                    for i in range(num_iterations):
                        # Check for incoming updates periodically
                        if i % 50 == 0:
                            _process_incoming_updates(worker_id, update_queues[worker_id], storage)

                        util = solver.train_iteration()
                        utilities.append(util)

                        if (i + 1) % 100 == 0:
                            elapsed = time.time() - iter_start
                            rate = (i + 1) / elapsed
                            print(
                                f"[Worker {worker_id}] Progress: {i + 1}/{num_iterations} "
                                f"({rate:.1f} iter/s)",
                                file=sys.stderr,
                                flush=True,
                            )

                    iter_time = time.time() - iter_start

                    # Send pending cross-partition updates to owners
                    pending = storage.get_pending_updates()
                    if pending:
                        _send_updates_to_owners(worker_id, num_workers, pending, update_queues)
                        storage.clear_pending_updates()

                    print(
                        f"[Worker {worker_id}] Completed {num_iterations} iterations "
                        f"in {iter_time:.1f}s ({num_iterations / iter_time:.1f} iter/s), "
                        f"owns {storage.num_owned_infosets()} infosets",
                        file=sys.stderr,
                        flush=True,
                    )

                    result_queue.put(
                        {
                            "worker_id": worker_id,
                            "batch_id": batch_id,
                            "type": "iterations_done",
                            "utilities": utilities,
                            "num_owned_infosets": storage.num_owned_infosets(),
                            "new_keys": storage.get_key_mapping(),  # For key sync
                            "iter_time": iter_time,
                        }
                    )

                except Exception as e:
                    print(
                        f"[Worker {worker_id}] Error: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exc()
                    result_queue.put(
                        {
                            "worker_id": worker_id,
                            "batch_id": batch_id,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )

                batch_count += 1

        # Cleanup (just close handles, don't unlink - coordinator does that)
        print(f"[Worker {worker_id}] Cleaning up...", file=sys.stderr, flush=True)
        storage.cleanup()
        print(f"[Worker {worker_id}] Exiting cleanly", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            result_queue.put({"worker_id": worker_id, "error": str(e)}, timeout=5)
        except Exception:
            pass


def _process_incoming_updates(
    worker_id: int, update_queue: mp.Queue, storage: "SharedArrayStorage"
) -> int:
    """
    Process incoming cross-partition updates from other workers.

    Returns number of updates processed.
    """
    count = 0
    while True:
        try:
            update_batch = update_queue.get_nowait()
            storage.apply_updates(update_batch)
            count += len(update_batch)
        except queue.Empty:
            break
    return count


def _send_updates_to_owners(
    sender_id: int,
    num_workers: int,
    updates: Dict[int, Tuple[np.ndarray, np.ndarray]],
    update_queues: List[mp.Queue],
) -> None:
    """
    Send cross-partition updates to their respective owners.

    Groups updates by owner and sends to appropriate queue.
    """
    # Group updates by owner
    updates_by_owner: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {
        i: {} for i in range(num_workers)
    }

    for infoset_id, (regret_delta, strategy_delta) in updates.items():
        owner = infoset_id % num_workers
        if owner != sender_id:  # Don't send to self
            updates_by_owner[owner][infoset_id] = (regret_delta, strategy_delta)

    # Send to each owner's queue
    for owner_id, owner_updates in updates_by_owner.items():
        if owner_updates:
            try:
                update_queues[owner_id].put(owner_updates, timeout=5.0)
            except queue.Full:
                print(
                    f"[Worker {sender_id}] Warning: Update queue for worker {owner_id} is full",
                    flush=True,
                )


class SharedArrayWorkerManager:
    """
    Manages parallel training with shared array storage.

    Architecture:
    - Coordinator (this class) creates shared memory once at startup
    - Workers attach to shared memory
    - Data flows directly through shared arrays (no serialization)
    - Cross-partition updates via per-worker queues
    """

    def __init__(
        self,
        num_workers: int,
        config_dict: Dict,
        serialized_action_abstraction: bytes,
        serialized_card_abstraction: bytes,
        session_id: str | None = None,
        base_seed: int = 42,
        max_infosets: int = 2_000_000,
        max_actions: int = 10,
        checkpoint_dir: str | None = None,
    ):
        """
        Initialize shared array worker manager.

        Args:
            num_workers: Number of workers
            config_dict: Configuration dictionary
            serialized_action_abstraction: Pickled BettingActions
            serialized_card_abstraction: Pickled BucketingStrategy
            session_id: Unique session ID (auto-generated if None)
            base_seed: Base random seed
            max_infosets: Maximum infosets in shared arrays
            max_actions: Maximum actions per infoset
            checkpoint_dir: Optional checkpoint directory
        """
        import uuid
        from pathlib import Path

        from src.solver.storage import SharedArrayStorage

        self.num_workers = num_workers
        self.config_dict = config_dict
        self.serialized_action_abstraction = serialized_action_abstraction
        self.serialized_card_abstraction = serialized_card_abstraction
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.base_seed = base_seed
        self.max_infosets = max_infosets
        self.max_actions = max_actions
        self.checkpoint_dir = checkpoint_dir

        # Create coordinator storage (creates shared memory)
        print(f"[Coordinator] Creating shared memory (session={self.session_id})...", flush=True)
        self.storage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,  # Coordinator uses worker_id 0 for ownership checks
            session_id=self.session_id,
            max_infosets=max_infosets,
            max_actions=max_actions,
            is_coordinator=True,  # Create shared memory
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        )
        print(
            f"[Coordinator] Shared memory created: "
            f"{max_infosets * max_actions * 4 * 2 // 1024 // 1024}MB total",
            flush=True,
        )

        # Communication queues
        self.job_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()

        # Per-worker update queues for cross-partition updates
        self.update_queues: List[mp.Queue] = [mp.Queue() for _ in range(num_workers)]

        # Worker processes
        self.processes: List[mp.Process] = []

        # Merged key mapping (coordinator maintains authoritative copy)
        self._merged_keys: Dict = {}

        # Start workers
        self._start_workers()

    def _start_workers(self):
        """Start all worker processes."""
        print(f"[Coordinator] Starting {self.num_workers} workers...", flush=True)

        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    self.num_workers,
                    self.session_id,
                    self.config_dict,
                    self.serialized_action_abstraction,
                    self.serialized_card_abstraction,
                    self.base_seed,
                    self.job_queue,
                    self.result_queue,
                    self.update_queues,
                    self.max_infosets,
                    self.max_actions,
                    self.checkpoint_dir,
                ),
            )
            p.start()
            self.processes.append(p)

        # Wait a moment for workers to attach
        time.sleep(1.0)
        print(f"[Coordinator] All {self.num_workers} workers started", flush=True)

    def sync_keys(self, timeout: float = 60.0, verbose: bool = True) -> Dict:
        """
        Synchronize key mappings across all workers.

        Collects new keys discovered by each worker and broadcasts
        the merged mapping to all workers.

        Returns:
            Dict with sync stats
        """
        if verbose:
            print("[Coordinator] Syncing key mappings...", flush=True)

        # Broadcast merged keys to all workers
        for _ in range(self.num_workers):
            self.job_queue.put({"type": JobType.SYNC_KEYS.value, "key_mapping": self._merged_keys})

        # Wait for acks
        acks = []
        for _ in range(self.num_workers):
            try:
                result = self.result_queue.get(timeout=timeout)
                if result.get("type") == "sync_keys_ack":
                    acks.append(result)
            except queue.Empty:
                raise RuntimeError("Timeout waiting for key sync acks")

        if verbose:
            print(f"[Coordinator] Keys synced: {len(self._merged_keys)} total", flush=True)

        return {"num_keys": len(self._merged_keys), "acks": acks}

    def apply_pending_updates(self, timeout: float = 60.0, verbose: bool = True) -> Dict:
        """
        Trigger workers to apply any pending cross-partition updates.

        This ensures all buffered updates are processed.

        Returns:
            Dict with apply stats
        """
        if verbose:
            print("[Coordinator] Applying pending updates...", flush=True)

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
            print("[Coordinator] Pending updates applied", flush=True)

        return {"acks": acks}

    def run_batch(
        self,
        iterations_per_worker: List[int],
        batch_id: int = 0,
        start_iteration: int = 0,
        timeout: float = 600.0,
        verbose: bool = True,
    ) -> Dict:
        """
        Run a batch of iterations across all workers.

        Workers write directly to shared arrays for owned infosets.
        Cross-partition updates are sent via queues.

        Args:
            iterations_per_worker: Iterations per worker
            batch_id: Batch identifier
            start_iteration: Global iteration offset
            timeout: Max wait time
            verbose: Print progress

        Returns:
            Dict with batch results and utilities
        """
        batch_start = time.time()

        if verbose:
            total_iters = sum(iterations_per_worker)
            print(
                f"[Coordinator] Running batch {batch_id}: {total_iters} total iterations "
                f"across {self.num_workers} workers",
                flush=True,
            )

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
        new_keys_batch = []

        for _ in range(active_workers):
            try:
                result = self.result_queue.get(timeout=timeout)

                if "error" in result:
                    errors.append(result)
                    print(
                        f"[Coordinator] Worker {result['worker_id']} error: {result['error']}",
                        flush=True,
                    )
                elif result.get("type") == "iterations_done":
                    results.append(result)
                    all_utilities.extend(result.get("utilities", []))

                    # Collect new keys for merging
                    if "new_keys" in result:
                        new_keys_batch.append(result["new_keys"])

                    if verbose:
                        print(
                            f"[Coordinator] Worker {result['worker_id']} done: "
                            f"{result['num_owned_infosets']} owned infosets, "
                            f"{result['iter_time']:.1f}s",
                            flush=True,
                        )

            except queue.Empty:
                raise RuntimeError(
                    f"Timeout waiting for workers ({len(results)}/{active_workers} received)"
                )

        # Merge new keys from all workers
        for keys in new_keys_batch:
            self._merged_keys.update(keys)

        # Update coordinator's storage key mapping
        self.storage.set_key_mapping(self._merged_keys)

        batch_time = time.time() - batch_start
        total_iters = sum(iterations_per_worker)

        if verbose:
            print(
                f"[Coordinator] Batch {batch_id} complete in {batch_time:.1f}s "
                f"({total_iters / batch_time:.1f} iter/s), "
                f"{self.storage.num_infosets()} total infosets",
                flush=True,
            )

        if errors:
            raise RuntimeError(f"Workers failed: {errors}")

        return {
            "utilities": all_utilities,
            "batch_time": batch_time,
            "total_iterations": total_iters,
            "worker_results": results,
            "num_infosets": self.storage.num_infosets(),
        }

    def checkpoint(self, iteration: int):
        """Save checkpoint to disk (coordinator writes shared arrays)."""
        self.storage.checkpoint(iteration)

    def get_storage(self) -> "SharedArrayStorage":
        """Get coordinator's storage instance (for accessing results)."""
        return self.storage

    def shutdown(self):
        """Shutdown all workers cleanly."""
        print("[Coordinator] Shutting down workers...", flush=True)

        # Send shutdown to all workers
        for _ in range(self.num_workers):
            self.job_queue.put({"type": JobType.SHUTDOWN.value})

        # Wait for processes to exit
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                print(f"[Coordinator] Force terminating worker {p.pid}", flush=True)
                p.terminate()

        self.processes.clear()

        # Cleanup shared memory (coordinator unlinks)
        self.storage.cleanup()

        print("[Coordinator] All workers shut down", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
