"""
Parallel training infrastructure for MCCFR solver.

Manages persistent worker pools, job distribution, and result merging
for multi-process training.
"""

import multiprocessing as mp
import pickle
import queue
import random
import struct
import time
import traceback
from enum import Enum
from multiprocessing import shared_memory
from typing import Any, Dict, List, Union

import numpy as np


class JobType(Enum):
    """Types of jobs that can be sent to worker processes."""

    RUN_ITERATIONS = "run_iterations"
    BROADCAST_REGRETS = "broadcast_regrets"  # Sync master regrets to workers
    SHUTDOWN = "shutdown"


def _serialize_infosets_to_shm(infoset_data: Dict, shm_name: str):
    """
    Serialize infosets to shared memory buffer for zero-copy transfer.

    Format:
    - 4 bytes: num_infosets (uint32)
    - For each infoset:
      - 4 bytes: key_length (uint32)
      - key_length bytes: pickled key
      - 4 bytes: num_actions (uint32)
      - num_actions * 4 bytes: regrets (float32)
      - num_actions * 4 bytes: strategies (float32)
      - 4 bytes: reach_count (uint32)
      - 8 bytes: cumulative_utility (float64)

    Args:
        infoset_data: Dictionary mapping InfoSetKey -> data dict
        shm_name: Name for the shared memory segment

    Returns:
        tuple: (SharedMemory object, total size in bytes)
    """
    # Calculate total size needed
    total_size = 4  # num_infosets
    key_bytes_list = []
    for key, data in infoset_data.items():
        key_bytes = pickle.dumps(key)
        key_bytes_list.append((key, key_bytes))
        total_size += 4 + len(key_bytes)  # key_length + key
        total_size += 4  # num_actions
        total_size += len(data["regrets"]) * 4  # regrets (float32)
        total_size += len(data["strategy_sum"]) * 4  # strategies (float32)
        total_size += 4  # reach_count (uint32)
        total_size += 8  # cumulative_utility (float64)

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=total_size, name=shm_name)

    try:
        # Write data
        buf = shm.buf
        offset = 0

        # Write num_infosets
        struct.pack_into("I", buf, offset, len(infoset_data))
        offset += 4

        # Write each infoset
        for key, key_bytes in key_bytes_list:
            data = infoset_data[key]

            # Write key
            struct.pack_into("I", buf, offset, len(key_bytes))
            offset += 4
            buf[offset : offset + len(key_bytes)] = key_bytes
            offset += len(key_bytes)

            # Write regrets/strategies
            num_actions = len(data["regrets"])
            struct.pack_into("I", buf, offset, num_actions)
            offset += 4

            regrets_bytes = np.array(data["regrets"], dtype=np.float32).tobytes()
            buf[offset : offset + len(regrets_bytes)] = regrets_bytes
            offset += len(regrets_bytes)

            strategies_bytes = np.array(data["strategy_sum"], dtype=np.float32).tobytes()
            buf[offset : offset + len(strategies_bytes)] = strategies_bytes
            offset += len(strategies_bytes)

            # Write metadata
            struct.pack_into("I", buf, offset, data["reach_count"])
            offset += 4
            struct.pack_into("d", buf, offset, data["cumulative_utility"])
            offset += 8

        return shm, total_size

    except Exception:
        # Cleanup on error
        shm.close()
        shm.unlink()
        raise


def _deserialize_infosets_from_shm(shm_name: str, shm_size: int) -> Dict:
    """
    Deserialize infosets from shared memory buffer.

    Args:
        shm_name: Name of the shared memory segment
        shm_size: Expected size of the shared memory (for validation)

    Returns:
        Dict mapping InfoSetKey -> data dict with regrets, strategy_sum, etc.
    """
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = shm.buf
        offset = 0

        # Read num_infosets
        num_infosets = struct.unpack_from("I", buf, offset)[0]
        offset += 4

        infoset_data = {}

        # Read each infoset
        for _ in range(num_infosets):
            # Read key
            key_length = struct.unpack_from("I", buf, offset)[0]
            offset += 4
            key_bytes = bytes(buf[offset : offset + key_length])
            key = pickle.loads(key_bytes)
            offset += key_length

            # Read arrays
            num_actions = struct.unpack_from("I", buf, offset)[0]
            offset += 4

            regrets = np.frombuffer(buf, dtype=np.float32, count=num_actions, offset=offset).copy()
            offset += num_actions * 4

            strategies = np.frombuffer(
                buf, dtype=np.float32, count=num_actions, offset=offset
            ).copy()
            offset += num_actions * 4

            # Read metadata
            reach_count = struct.unpack_from("I", buf, offset)[0]
            offset += 4
            cumulative_utility = struct.unpack_from("d", buf, offset)[0]
            offset += 8

            infoset_data[key] = {
                "regrets": regrets,
                "strategy_sum": strategies,
                "reach_count": reach_count,
                "cumulative_utility": cumulative_utility,
                # Note: legal_actions not included in shm transfer (reconstructed from storage)
            }

        return infoset_data

    finally:
        # Always cleanup shared memory
        if shm is not None:
            shm.close()
            shm.unlink()


def _persistent_worker_loop(
    worker_id: int,
    config_dict: Dict,
    serialized_action_abstraction: bytes,
    serialized_card_abstraction: bytes,
    base_seed: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Persistent worker process that runs in a loop processing jobs.

    This worker:
    1. Initializes solver once at startup (avoiding repeated process spawning overhead)
    2. Loops waiting for jobs from the master process
    3. Executes jobs (run N iterations) and returns results
    4. Exits cleanly on shutdown signal

    Args:
        worker_id: Unique worker identifier
        config_dict: Configuration dictionary
        serialized_action_abstraction: Pickled BettingActions
        serialized_card_abstraction: Pickled BucketingStrategy
        base_seed: Base random seed for this worker
        job_queue: Queue to receive job specifications
        result_queue: Queue to return results
    """
    try:
        import sys

        from src.solver.mccfr import MCCFRSolver
        from src.solver.storage import InMemoryStorage
        from src.utils.config import Config

        print(f"[Worker {worker_id}] Initializing...", file=sys.stderr, flush=True)

        # ONE-TIME INITIALIZATION (avoids per-batch overhead)
        # Recreate config
        config = Config.from_dict(config_dict)
        print(f"[Worker {worker_id}] Config loaded", file=sys.stderr, flush=True)

        # Deserialize pre-built abstractions
        print(
            f"[Worker {worker_id}] Deserializing action abstraction ({len(serialized_action_abstraction)} bytes)...",
            file=sys.stderr,
            flush=True,
        )
        action_abstraction = pickle.loads(serialized_action_abstraction)
        print(
            f"[Worker {worker_id}] Deserializing card abstraction ({len(serialized_card_abstraction)} bytes)...",
            file=sys.stderr,
            flush=True,
        )
        card_abstraction = pickle.loads(serialized_card_abstraction)
        print(f"[Worker {worker_id}] Abstractions loaded", file=sys.stderr, flush=True)

        # Workers always use in-memory storage
        storage = InMemoryStorage()

        # Create solver config with unique seed
        solver_config = config.get_section("game").copy()
        solver_config.update(config.get_section("system"))
        solver_config["seed"] = base_seed  # Base seed per worker

        # Workers must use vanilla CFR (no regret flooring) for mathematically correct merging
        # CFR+ flooring will be applied at master after merge
        solver_config["cfr_plus"] = False

        # Build solver once
        print(f"[Worker {worker_id}] Creating solver...", file=sys.stderr, flush=True)
        solver = MCCFRSolver(
            action_abstraction=action_abstraction,
            card_abstraction=card_abstraction,
            storage=storage,
            config=solver_config,
        )
        print(f"[Worker {worker_id}] Solver created, ready for jobs", file=sys.stderr, flush=True)

        # WORKER LOOP: Process jobs until shutdown
        while True:
            # Wait for job from master with timeout to check for termination
            try:
                job = job_queue.get(timeout=1.0)
            except queue.Empty:
                # Check if we should still be alive
                continue

            job_type = JobType(job["type"])

            if job_type == JobType.SHUTDOWN:
                # Clean shutdown
                print(
                    f"[Worker {worker_id}] Received shutdown signal, exiting...",
                    file=sys.stderr,
                    flush=True,
                )
                break

            elif job_type == JobType.BROADCAST_REGRETS:
                # Load master's current regrets into worker storage
                # This allows workers to sample from the learned strategy
                print(
                    f"[Worker {worker_id}] Receiving regret broadcast from master...",
                    file=sys.stderr,
                    flush=True,
                )

                # Clear existing storage before loading broadcast
                storage.clear()

                # Deserialize from shared memory if using shm transfer
                if "shm_name" in job:
                    try:
                        print(
                            f"[Worker {worker_id}] Loading from shared memory: {job['shm_name']} ({job['shm_size']} bytes)...",
                            file=sys.stderr,
                            flush=True,
                        )
                        infoset_data = _deserialize_infosets_from_shm(
                            job["shm_name"], job["shm_size"]
                        )

                        # Merge legal_actions back into infoset_data
                        legal_actions_map = job.get("legal_actions_map", {})
                        for key, data in infoset_data.items():
                            data["legal_actions"] = legal_actions_map.get(key, [])

                        # Load into worker storage
                        for key, data in infoset_data.items():
                            infoset = storage.get_or_create_infoset(key, data["legal_actions"])
                            infoset.regrets = np.array(data["regrets"], dtype=np.float32)
                            infoset.strategy_sum = np.array(data["strategy_sum"], dtype=np.float32)
                            infoset.reach_count = data["reach_count"]
                            infoset.cumulative_utility = data["cumulative_utility"]

                        print(
                            f"[Worker {worker_id}] Loaded {len(infoset_data)} infosets from master",
                            file=sys.stderr,
                            flush=True,
                        )
                    except Exception as e:
                        print(
                            f"[Worker {worker_id}] ERROR loading regrets: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Continue with empty storage if broadcast fails
                        storage.clear()

                # Send acknowledgment back
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "type": "broadcast_ack",
                        "num_infosets_loaded": len(storage.infosets)
                        if hasattr(storage, "infosets")
                        else storage.num_infosets(),
                    }
                )

            elif job_type == JobType.RUN_ITERATIONS:
                # Extract job parameters
                num_iterations = job["num_iterations"]
                batch_id = job.get("batch_id", 0)
                iteration_offset = job.get("iteration_offset", 0)

                # Do NOT clear storage - workers should have received regret broadcast
                # Storage now contains master's current regrets for correct MCCFR sampling
                solver.iteration = iteration_offset
                solver.total_utility = 0.0

                # Update seed for this batch (ensures different rollouts per batch)
                batch_seed = base_seed + batch_id * 1000
                random.seed(batch_seed)
                np.random.seed(batch_seed)

                # Run iterations
                utilities = []
                try:
                    import sys

                    print(
                        f"[Worker {worker_id}] Starting {num_iterations} iterations...",
                        file=sys.stderr,
                        flush=True,
                    )
                    for i in range(num_iterations):
                        util = solver.train_iteration()
                        utilities.append(util)
                        # Progress update every 50 iterations
                        if (i + 1) % 50 == 0:
                            print(
                                f"[Worker {worker_id}] Progress: {i + 1}/{num_iterations} iterations",
                                file=sys.stderr,
                                flush=True,
                            )
                    print(
                        f"[Worker {worker_id}] Completed {num_iterations} iterations, discovered {len(storage.infosets)} infosets",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as e:
                    # Report error back to master
                    print(
                        f"[Worker {worker_id}] ERROR during iterations: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result_queue.put(
                        {
                            "worker_id": worker_id,
                            "batch_id": batch_id,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    continue

                # Extract infoset data for merging
                print(
                    f"[Worker {worker_id}] Extracting {len(storage.infosets)} infosets...",
                    file=sys.stderr,
                    flush=True,
                )
                infoset_data = {}
                legal_actions_map = {}  # Store separately (not in shared memory)
                for key, infoset in storage.infosets.items():
                    infoset_data[key] = {
                        "regrets": infoset.regrets.copy(),
                        "strategy_sum": infoset.strategy_sum.copy(),
                        "reach_count": infoset.reach_count,
                        "cumulative_utility": infoset.cumulative_utility,
                    }
                    legal_actions_map[key] = infoset.legal_actions

                # Return results using shared memory for large arrays
                print(
                    f"[Worker {worker_id}] Serializing {len(infoset_data)} infosets to shared memory...",
                    file=sys.stderr,
                    flush=True,
                )

                # Serialize to shared memory
                shm_name = f"worker_{worker_id}_batch_{batch_id}"
                shm = None
                try:
                    shm, shm_size = _serialize_infosets_to_shm(infoset_data, shm_name)

                    print(
                        f"[Worker {worker_id}] Sending results to master (shm: {shm_size} bytes, ~{len(infoset_data)} infosets)...",
                        file=sys.stderr,
                        flush=True,
                    )

                    # Use a separate thread to put results with timeout detection
                    # This prevents deadlock if the queue buffer fills up
                    import threading

                    result_sent = threading.Event()
                    put_error = []

                    def _put_result():
                        try:
                            result_queue.put(
                                {
                                    "worker_id": worker_id,
                                    "batch_id": batch_id,
                                    "utilities": utilities,
                                    "shm_name": shm_name,
                                    "shm_size": shm_size,
                                    "legal_actions_map": legal_actions_map,
                                    "num_infosets": len(infoset_data),
                                }
                            )
                            result_sent.set()
                        except Exception as e:
                            put_error.append(e)
                            result_sent.set()

                    put_thread = threading.Thread(target=_put_result, daemon=True)
                    put_thread.start()

                    # Wait with timeout - if it blocks too long, continue anyway
                    # The daemon thread will be abandoned but worker can still exit
                    if not result_sent.wait(timeout=120):
                        print(
                            f"[Worker {worker_id}] WARNING: result_queue.put() blocked for >120s! "
                            f"Queue may be full or master stopped listening. "
                            f"Abandoning result send and continuing...",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Don't wait indefinitely - allow worker to continue/exit
                        # Shared memory will be cleaned up by master during shutdown
                    elif put_error:
                        raise put_error[0]
                    else:
                        print(
                            f"[Worker {worker_id}] Results sent successfully",
                            file=sys.stderr,
                            flush=True,
                        )

                finally:
                    # Worker closes shared memory (master will unlink)
                    if shm is not None:
                        shm.close()

        print(f"[Worker {worker_id}] Exiting cleanly", file=sys.stderr, flush=True)

    except Exception as e:
        # Report error and exit - but use try-except in case queue is closed
        print(f"[Worker {worker_id}] Fatal error: {e}", file=sys.stderr, flush=True)
        try:
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "error": str(e),
                },
                timeout=5,
            )
        except Exception:
            # Queue is closed or full - just exit
            print(
                f"[Worker {worker_id}] Could not report error to master, exiting",
                file=sys.stderr,
                flush=True,
            )


class WorkerManager:
    """
    Manages a persistent pool of worker processes for parallel training.

    This manager:
    - Starts N worker processes once at initialization
    - Distributes jobs to workers via queues
    - Collects results from workers
    - Handles clean shutdown

    Benefits over per-batch process spawning:
    - Eliminates process startup overhead
    - Avoids repeated serialization/deserialization
    - Provides stable, predictable throughput
    """

    def __init__(
        self,
        num_workers: int,
        config_dict: Dict,
        serialized_action_abstraction: bytes,
        serialized_card_abstraction: bytes,
        base_seed: int = 42,
    ):
        """
        Initialize worker pool.

        Args:
            num_workers: Number of worker processes
            config_dict: Configuration dictionary
            serialized_action_abstraction: Pickled BettingActions
            serialized_card_abstraction: Pickled BucketingStrategy
            base_seed: Base random seed (each worker gets base_seed + worker_id)
        """
        self.num_workers = num_workers
        self.config_dict = config_dict
        self.serialized_action_abstraction = serialized_action_abstraction
        self.serialized_card_abstraction = serialized_card_abstraction
        self.base_seed = base_seed

        # Communication queues
        self.job_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()

        # Worker processes
        self.processes: List[mp.Process] = []

        # Track active shared memory objects for cleanup
        self._active_shm_names: List[str] = []

        # Start workers
        self._start_workers()

    def _start_workers(self):
        """Start all worker processes."""
        for worker_id in range(self.num_workers):
            # Each worker gets a unique seed
            worker_seed = self.base_seed + worker_id * 10000

            p = mp.Process(
                target=_persistent_worker_loop,
                args=(
                    worker_id,
                    self.config_dict,
                    self.serialized_action_abstraction,
                    self.serialized_card_abstraction,
                    worker_seed,
                    self.job_queue,
                    self.result_queue,
                ),
            )
            p.start()
            self.processes.append(p)

    def submit_batch(
        self,
        iterations_per_worker: List[int],
        batch_id: int = 0,
        start_iteration: int = 0,
        timeout_seconds: float | None = None,
        verbose: bool = False,
        storage=None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Submit a batch of jobs to workers and collect results.

        With incremental merging enabled (storage provided):
        - Merges each worker's results immediately upon arrival
        - Returns dict with utilities: {"utilities": [list of floats]}

        Without storage (legacy batch merge mode):
        - Collects all worker results
        - Returns list of result dicts (unused in current codebase)

        Args:
            iterations_per_worker: List of iteration counts (one per worker)
            batch_id: Batch identifier for seeding
            start_iteration: Global iteration offset for this batch
            timeout_seconds: Optional total timeout for the batch (None = wait indefinitely)
            verbose: Whether to print periodic wait warnings
            storage: Optional master storage to merge into as results arrive (enables incremental merging)

        Returns:
            Dict with utilities (incremental merge mode) or List of worker results (legacy mode)
        """
        # Extract CFR+ setting from config for merge operation
        # Default to False if not specified
        cfr_plus_enabled = self.config_dict.get("system", {}).get("cfr_plus", False)

        # Broadcast master regrets to workers before running iterations
        # This ensures workers sample from the learned strategy (correct MCCFR)
        if storage is not None and hasattr(storage, "infosets"):
            num_infosets = (
                storage.num_infosets()
                if hasattr(storage, "num_infosets")
                else len(storage.infosets)
            )
            if num_infosets > 0:
                # Only broadcast if we have learned something
                if verbose:
                    print(
                        f"[Master] Broadcasting {num_infosets} infosets to workers...", flush=True
                    )
                self.broadcast_regrets(storage, verbose=verbose)
            elif verbose:
                print(
                    "[Master] No infosets to broadcast, workers will start with empty storage",
                    flush=True,
                )

        # Submit jobs to all workers
        iteration_offset = start_iteration
        for worker_id, num_iterations in enumerate(iterations_per_worker):
            if num_iterations > 0:
                job = {
                    "type": JobType.RUN_ITERATIONS.value,
                    "num_iterations": num_iterations,
                    "batch_id": batch_id,
                    "iteration_offset": iteration_offset,
                }
                self.job_queue.put(job)
                iteration_offset += num_iterations

        # Collect results (merge incrementally if storage provided)
        worker_results: List[Dict] = []
        all_utilities: List[float] = []
        num_active_workers = sum(1 for n in iterations_per_worker if n > 0)
        results_received = 0

        print(f"[Master] Waiting for {num_active_workers} worker results...", flush=True)

        start_time = time.time()
        last_warning = start_time
        warn_after = 60.0
        warn_interval = 30.0

        while results_received < num_active_workers:
            try:
                result = self.result_queue.get(timeout=5)

                results_received += 1
                print(
                    f"[Master] Received result from worker {result.get('worker_id', '?')} "
                    f"({results_received}/{num_active_workers})",
                    flush=True,
                )

                if "error" in result:
                    traceback_text = result.get("traceback", "")
                    raise RuntimeError(
                        f"Worker {result['worker_id']} error: {result['error']}\n{traceback_text}"
                    )

                # Deserialize from shared memory if using shm transfer
                if "shm_name" in result:
                    print(
                        f"[Master] Deserializing from shared memory: {result['shm_name']} ({result['shm_size']} bytes)...",
                        flush=True,
                    )
                    try:
                        infoset_data = _deserialize_infosets_from_shm(
                            result["shm_name"], result["shm_size"]
                        )
                        # Merge legal_actions back into infoset_data
                        legal_actions_map = result.get("legal_actions_map", {})
                        for key, data in infoset_data.items():
                            data["legal_actions"] = legal_actions_map.get(key, [])
                        # Inject infoset_data into result for merge function
                        result["infoset_data"] = infoset_data
                        # Remove from tracking since it's been cleaned up
                        if result["shm_name"] in self._active_shm_names:
                            self._active_shm_names.remove(result["shm_name"])
                    except Exception as e:
                        print(
                            f"[Master] ERROR deserializing shared memory: {e}",
                            flush=True,
                        )
                        raise

                # Incremental merge: merge this worker's results immediately
                if storage is not None:
                    merge_start = time.time()
                    merge_worker_results(storage, [result], cfr_plus=cfr_plus_enabled)
                    merge_time = time.time() - merge_start
                    if verbose:
                        print(
                            f"[Master] Merged worker {result.get('worker_id', '?')} results in {merge_time:.2f}s",
                            flush=True,
                        )
                    # Collect utilities for metrics
                    all_utilities.extend(result.get("utilities", []))
                else:
                    # Traditional mode: collect all results for batch merge
                    worker_results.append(result)

            except queue.Empty as e:
                dead = [p for p in self.processes if not p.is_alive()]
                if dead:
                    details = ", ".join(f"pid={p.pid}, exitcode={p.exitcode}" for p in dead)
                    raise RuntimeError(
                        f"Worker process exited unexpectedly while waiting for results: {details}"
                    ) from e
                elapsed = time.time() - start_time
                if timeout_seconds is not None and elapsed >= timeout_seconds:
                    raise TimeoutError(
                        "Timed out waiting for worker results. "
                        "Workers are alive but not responding."
                    ) from e
                if verbose and elapsed >= warn_after and (elapsed - last_warning) >= warn_interval:
                    print(
                        f"Waiting for worker results... {elapsed:.0f}s elapsed "
                        f"({results_received}/{num_active_workers} received)"
                    )
                    last_warning = elapsed
            except Exception:
                raise

        if storage is not None:
            print(
                f"[Master] All {num_active_workers} worker results merged incrementally",
                flush=True,
            )
            # Return utilities directly (incremental merge mode)
            return {"utilities": all_utilities}
        else:
            print(
                f"[Master] All {num_active_workers} results collected, returning to caller...",
                flush=True,
            )
            # Return batch results (legacy batch merge mode - unused)
            return worker_results

    def broadcast_regrets(self, storage, verbose: bool = False):
        """
        Broadcast master's current regrets to all workers.

        This allows workers to sample from the current learned strategy (correct MCCFR).

        Args:
            storage: Master storage containing current regrets
            verbose: Whether to print progress
        """
        if verbose:
            print("[Master] Broadcasting regrets to workers...", flush=True)

        broadcast_start = time.time()

        # Serialize master's infosets
        infoset_data = {}
        for key in storage.infosets:
            infoset = storage.get_infoset(key)
            if infoset is not None:
                infoset_data[key] = {
                    "legal_actions": list(infoset.legal_actions),
                    "regrets": infoset.regrets.tolist(),
                    "strategy_sum": infoset.strategy_sum.tolist(),
                    "reach_count": infoset.reach_count,
                    "cumulative_utility": infoset.cumulative_utility,
                }

        if verbose:
            print(
                f"[Master] Serializing {len(infoset_data)} infosets for broadcast...",
                flush=True,
            )

        # Use shared memory for efficient transfer
        import uuid

        shm_name = f"broadcast_regrets_{uuid.uuid4().hex[:8]}"

        try:
            # Serialize to shared memory
            shm, shm_size = _serialize_infosets_to_shm(infoset_data, shm_name)
            self._active_shm_names.append(shm_name)

            if verbose:
                print(
                    f"[Master] Broadcast data serialized ({shm_size} bytes), sending to workers...",
                    flush=True,
                )

            # Extract legal_actions for separate transfer (can't pickle Action objects in shm)
            legal_actions_map = {key: data["legal_actions"] for key, data in infoset_data.items()}

            # Send broadcast job to all workers
            for worker_id in range(self.num_workers):
                self.job_queue.put(
                    {
                        "type": JobType.BROADCAST_REGRETS.value,
                        "shm_name": shm_name,
                        "shm_size": shm_size,
                        "legal_actions_map": legal_actions_map,
                    }
                )

            # Wait for acknowledgments from all workers
            acks_received = 0
            while acks_received < self.num_workers:
                try:
                    result = self.result_queue.get(timeout=30)
                    if result.get("type") == "broadcast_ack":
                        acks_received += 1
                        if verbose:
                            print(
                                f"[Master] Worker {result['worker_id']} acknowledged broadcast "
                                f"({result.get('num_infosets_loaded', 0)} infosets loaded) "
                                f"[{acks_received}/{self.num_workers}]",
                                flush=True,
                            )
                except queue.Empty:
                    raise RuntimeError(
                        f"Timeout waiting for broadcast acknowledgments "
                        f"({acks_received}/{self.num_workers} received)"
                    )

            # Cleanup shared memory
            shm.close()
            shm.unlink()
            if shm_name in self._active_shm_names:
                self._active_shm_names.remove(shm_name)

            broadcast_time = time.time() - broadcast_start
            if verbose:
                print(
                    f"[Master] Broadcast complete in {broadcast_time:.2f}s "
                    f"({len(infoset_data)} infosets, {shm_size / 1024 / 1024:.1f} MB)",
                    flush=True,
                )

        except Exception as e:
            # Cleanup on error
            try:
                if shm_name in self._active_shm_names:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                    self._active_shm_names.remove(shm_name)
            except Exception:
                pass
            raise RuntimeError(f"Failed to broadcast regrets: {e}") from e

    def shutdown(self):
        """Shutdown all workers cleanly."""
        print("[Master] Shutting down workers...", flush=True)

        # Send shutdown signal to all workers
        for _ in range(self.num_workers):
            self.job_queue.put({"type": JobType.SHUTDOWN.value})

        # Drain result queue to prevent workers from blocking on put()
        print("[Master] Draining result queue...", flush=True)
        drained_results = 0
        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                drained_results += 1
                # Clean up any shared memory from unprocessed results
                if "shm_name" in result:
                    try:
                        shm = shared_memory.SharedMemory(name=result["shm_name"])
                        shm.close()
                        shm.unlink()
                        print(
                            f"[Master] Cleaned up leaked shared memory: {result['shm_name']}",
                            flush=True,
                        )
                    except FileNotFoundError:
                        pass  # Already cleaned up
                    except Exception as e:
                        print(
                            f"[Master] Warning: Could not cleanup shared memory {result['shm_name']}: {e}",
                            flush=True,
                        )
            except queue.Empty:
                break

        if drained_results > 0:
            print(f"[Master] Drained {drained_results} pending results from queue", flush=True)

        # Wait for all processes to terminate with longer timeout
        print("[Master] Waiting for workers to exit...", flush=True)
        for i, p in enumerate(self.processes):
            p.join(timeout=10)  # Increased timeout from 5 to 10 seconds
            if p.is_alive():
                print(f"Warning: Process {p.pid} (worker {i}) did not terminate, killing it")
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    print(f"Warning: Process {p.pid} still alive after terminate, using kill")
                    p.kill()
                    p.join()

        # Clean up any remaining shared memory objects
        if self._active_shm_names:
            print(
                f"[Master] Cleaning up {len(self._active_shm_names)} remaining shared memory objects...",
                flush=True,
            )
            for shm_name in self._active_shm_names[:]:
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                    print(f"[Master] Cleaned up shared memory: {shm_name}", flush=True)
                except FileNotFoundError:
                    pass  # Already cleaned up
                except Exception as e:
                    print(f"[Master] Warning: Could not cleanup {shm_name}: {e}", flush=True)
            self._active_shm_names.clear()

        # Close and join queues
        self.job_queue.close()
        self.job_queue.join_thread()
        self.result_queue.close()
        self.result_queue.join_thread()

        print("[Master] Shutdown complete.", flush=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.shutdown()


def _fast_action_sum(worker_infosets, action_index, num_actions):
    """
    Fast action alignment and summation with minimized hash lookups.

    Optimized to cache index lookups and use direct array indexing.
    """
    sum_regrets = np.zeros(num_actions, dtype=np.float32)
    sum_strategies = np.zeros(num_actions, dtype=np.float32)

    for data in worker_infosets:
        actions = data["legal_actions"]
        regrets = data["regrets"]
        strategies = data["strategy_sum"]

        # Cache index lookups (minimize hashing)
        indices = [action_index[action] for action in actions]

        # Direct numpy array operations (no loops)
        n_regrets = min(len(indices), len(regrets))
        n_strategies = min(len(indices), len(strategies))

        for i in range(n_regrets):
            sum_regrets[indices[i]] += regrets[i]
        for i in range(n_strategies):
            sum_strategies[indices[i]] += strategies[i]

    return sum_regrets, sum_strategies


def merge_worker_results(solver_storage, worker_results: List[Dict], cfr_plus: bool = False):
    """
    Merge worker results into master storage.

    For each infoset:
    - Sum regrets across workers (CFR theory: regrets accumulate additively)
    - Sum strategy counts (CFR theory: strategies accumulate additively)
    - Sum reach counts (total visits across all workers)
    - Sum cumulative utilities (for proper averaging later)
    - Apply CFR+ flooring if enabled (after merge, not during worker updates)

    Args:
        solver_storage: Master storage to merge into
        worker_results: List of result dicts from workers
        cfr_plus: If True, apply CFR+ regret flooring after merging
    """
    merge_start_time = time.time()
    print(f"[Master] Starting merge of {len(worker_results)} worker results...", flush=True)

    # Collect all infoset keys (use set union for speed)
    all_keys = set().union(*(result.get("infoset_data", {}).keys() for result in worker_results))

    print(
        f"[Master] Merging {len(all_keys)} unique infosets...",
        flush=True,
    )

    # Track dirty keys for batch marking at end (use set to avoid duplicates)
    dirty_keys = set()

    # Merge each infoset
    merged_count = 0
    merge_loop_start = time.time()
    for key in all_keys:
        merged_count += 1
        if merged_count % 100000 == 0:
            elapsed = time.time() - merge_loop_start
            print(
                f"[Master] Merge progress: {merged_count}/{len(all_keys)} infosets "
                f"({elapsed:.1f}s, {merged_count / elapsed:.0f}/s)...",
                flush=True,
            )

        # Collect data from all workers for this key
        worker_infosets = []
        for result in worker_results:
            if "infoset_data" not in result:
                continue
            data = result["infoset_data"].get(key)
            if data is None:
                continue
            worker_infosets.append(data)

        if not worker_infosets:
            continue

        # Build unified action list using master (if present) + any new worker actions
        existing_infoset = solver_storage.get_infoset(key)
        if existing_infoset is not None:
            legal_actions = list(existing_infoset.legal_actions)
        else:
            legal_actions = list(worker_infosets[0]["legal_actions"])

        action_index = {action: idx for idx, action in enumerate(legal_actions)}
        for data in worker_infosets:
            for action in data["legal_actions"]:
                if action not in action_index:
                    action_index[action] = len(legal_actions)
                    legal_actions.append(action)

        # Get or create infoset
        if existing_infoset is not None:
            infoset = existing_infoset
        else:
            infoset = solver_storage.get_or_create_infoset(key, legal_actions)

        # Ensure master infoset has correct size (may need extending with newly discovered actions)
        if infoset.num_actions < len(legal_actions):
            target_size = len(legal_actions)
            infoset.legal_actions = legal_actions
            infoset.num_actions = target_size
            if hasattr(solver_storage, "infoset_actions"):
                solver_storage.infoset_actions[key] = legal_actions

            pad_size = target_size - len(infoset.regrets)
            infoset.regrets = np.pad(
                infoset.regrets, (0, pad_size), mode="constant", constant_values=0
            )
            infoset.strategy_sum = np.pad(
                infoset.strategy_sum, (0, pad_size), mode="constant", constant_values=0
            )

        # Sum regrets/strategies aligned by action identity
        sum_regrets = np.zeros(len(legal_actions), dtype=np.float32)
        sum_strategies = np.zeros(len(legal_actions), dtype=np.float32)

        for data in worker_infosets:
            actions = data["legal_actions"]
            regrets = data["regrets"]
            strategies = data["strategy_sum"]

            # Direct summation with minimal overhead
            for idx, action in enumerate(actions):
                target_idx = action_index[action]
                if idx < len(regrets):
                    sum_regrets[target_idx] += regrets[idx]
                if idx < len(strategies):
                    sum_strategies[target_idx] += strategies[idx]

        # Update master infoset with summed values
        infoset.regrets += sum_regrets
        infoset.strategy_sum += sum_strategies

        # Apply CFR+ flooring after merge (mathematically correct)
        # Workers use vanilla CFR (no flooring) so regrets sum correctly
        if cfr_plus:
            infoset.regrets = np.maximum(0, infoset.regrets)

        # Sum reach counts (total visits across all workers)
        infoset.reach_count += sum(data["reach_count"] for data in worker_infosets)

        # Sum cumulative utilities (for proper averaging later)
        infoset.cumulative_utility += sum(data["cumulative_utility"] for data in worker_infosets)

        # Track for batch dirty marking
        dirty_keys.add(key)

    # Batch mark dirty at end
    if dirty_keys:
        # Check if storage has bulk dirty marking support
        if hasattr(solver_storage, "mark_dirty_batch"):
            solver_storage.mark_dirty_batch(dirty_keys)
        else:
            for key in dirty_keys:
                solver_storage.mark_dirty(key)

    total_merge_time = time.time() - merge_start_time
    print(
        f"[Master] Merge complete! Merged {len(all_keys)} infosets in {total_merge_time:.2f}s "
        f"(throughput: {len(all_keys) / total_merge_time:.0f} infosets/s)",
        flush=True,
    )
