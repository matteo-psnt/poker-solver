"""
Parallel training infrastructure for MCCFR solver.

Manages persistent worker pools, job distribution, and result merging
for multi-process training.
"""

import multiprocessing as mp
import pickle
import queue
import random
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np


class JobType(Enum):
    """Types of jobs that can be sent to worker processes."""

    RUN_ITERATIONS = "run_iterations"
    SHUTDOWN = "shutdown"


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
            # Wait for job from master
            job = job_queue.get()

            job_type = JobType(job["type"])

            if job_type == JobType.SHUTDOWN:
                # Clean shutdown
                break

            elif job_type == JobType.RUN_ITERATIONS:
                # Extract job parameters
                num_iterations = job["num_iterations"]
                batch_id = job.get("batch_id", 0)
                iteration_offset = job.get("iteration_offset", 0)

                # Reset worker state per job to avoid double-counting across batches
                storage.clear()
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
                for key, infoset in storage.infosets.items():
                    infoset_data[key] = {
                        "regrets": infoset.regrets.copy(),
                        "strategy_sum": infoset.strategy_sum.copy(),
                        "legal_actions": infoset.legal_actions,
                        "reach_count": infoset.reach_count,
                        "cumulative_utility": infoset.cumulative_utility,
                    }

                # Return results
                print(
                    f"[Worker {worker_id}] Sending results to master (result size: ~{len(infoset_data)} infosets)...",
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
                                "infoset_data": infoset_data,
                                "num_infosets": len(infoset_data),
                            }
                        )
                        result_sent.set()
                    except Exception as e:
                        put_error.append(e)
                        result_sent.set()

                put_thread = threading.Thread(target=_put_result, daemon=True)
                put_thread.start()

                # Wait with timeout
                if not result_sent.wait(timeout=120):
                    print(
                        f"[Worker {worker_id}] WARNING: result_queue.put() blocked for >120s! "
                        f"Queue may be full. Consider reducing batch size.",
                        file=sys.stderr,
                        flush=True,
                    )
                    # Wait indefinitely now
                    result_sent.wait()

                if put_error:
                    raise put_error[0]

                print(
                    f"[Worker {worker_id}] Results sent successfully",
                    file=sys.stderr,
                    flush=True,
                )

    except Exception as e:
        # Report error and exit
        result_queue.put(
            {
                "worker_id": worker_id,
                "error": str(e),
            }
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

                # Incremental merge: merge this worker's results immediately
                if storage is not None:
                    merge_start = time.time()
                    merge_worker_results(storage, [result])
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

    def shutdown(self):
        """Shutdown all workers cleanly."""
        print("[Master] Shutting down workers...", flush=True)
        # Send shutdown signal to all workers
        for _ in range(self.num_workers):
            self.job_queue.put({"type": JobType.SHUTDOWN.value})

        # Wait for all processes to terminate
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"Warning: Process {p.pid} did not terminate, killing it")
                p.terminate()
                p.join()

        # Clean up queues
        self.job_queue.close()
        self.result_queue.close()
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


def merge_worker_results(solver_storage, worker_results: List[Dict]):
    """
    Merge worker results into master storage.

    For each infoset:
    - Sum regrets across workers (CFR theory: regrets accumulate additively)
    - Sum strategy counts (CFR theory: strategies accumulate additively)
    - Sum reach counts (total visits across all workers)
    - Sum cumulative utilities (for proper averaging later)

    Args:
        solver_storage: Master storage to merge into
        worker_results: List of result dicts from workers
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
