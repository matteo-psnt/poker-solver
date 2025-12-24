"""
Parallel training infrastructure for MCCFR solver.

Manages persistent worker pools, job distribution, and result merging
for multi-process training.
"""

import multiprocessing as mp
import pickle
import random
import traceback
from enum import Enum
from typing import Dict, List

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
        from src.solver.mccfr import MCCFRSolver
        from src.solver.storage import InMemoryStorage
        from src.utils.config import Config

        # ONE-TIME INITIALIZATION (avoids per-batch overhead)
        # Recreate config
        config = Config.from_dict(config_dict)

        # Deserialize pre-built abstractions
        action_abstraction = pickle.loads(serialized_action_abstraction)
        card_abstraction = pickle.loads(serialized_card_abstraction)

        # Workers always use in-memory storage
        storage = InMemoryStorage()

        # Create solver config with unique seed
        solver_config = config.get_section("game").copy()
        solver_config.update(config.get_section("system"))
        solver_config["seed"] = base_seed  # Base seed per worker

        # Build solver once
        solver = MCCFRSolver(
            action_abstraction=action_abstraction,
            card_abstraction=card_abstraction,
            storage=storage,
            config=solver_config,
        )

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

                # Update seed for this batch (ensures different rollouts per batch)
                batch_seed = base_seed + batch_id * 1000
                random.seed(batch_seed)
                np.random.seed(batch_seed)

                # Run iterations
                utilities = []
                try:
                    for _ in range(num_iterations):
                        util = solver.train_iteration()
                        utilities.append(util)
                except Exception as e:
                    # Report error back to master
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
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "batch_id": batch_id,
                        "utilities": utilities,
                        "infoset_data": infoset_data,
                        "num_infosets": len(infoset_data),
                    }
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

    def submit_batch(self, iterations_per_worker: List[int], batch_id: int = 0) -> List[Dict]:
        """
        Submit a batch of jobs to workers and collect results.

        Args:
            iterations_per_worker: List of iteration counts (one per worker)
            batch_id: Batch identifier for seeding

        Returns:
            List of result dictionaries from workers
        """
        # Submit jobs to all workers
        for worker_id, num_iterations in enumerate(iterations_per_worker):
            if num_iterations > 0:
                job = {
                    "type": JobType.RUN_ITERATIONS.value,
                    "num_iterations": num_iterations,
                    "batch_id": batch_id,
                }
                self.job_queue.put(job)

        # Collect results
        worker_results = []
        num_active_workers = sum(1 for n in iterations_per_worker if n > 0)

        for _ in range(num_active_workers):
            try:
                result = self.result_queue.get(timeout=60)

                if "error" in result:
                    print(f"Worker {result['worker_id']} error: {result['error']}")
                    continue

                worker_results.append(result)

            except Exception as e:
                print(f"Failed to get result from worker: {e}")

        return worker_results

    def shutdown(self):
        """Shutdown all workers cleanly."""
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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.shutdown()


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
    # Collect all infoset keys
    all_keys = set()
    for result in worker_results:
        if "infoset_data" in result:
            all_keys.update(result["infoset_data"].keys())

    # Merge each infoset
    for key in all_keys:
        # Collect data from all workers
        regrets_list = []
        strategies_list = []
        reach_counts_list = []
        cumulative_utilities_list = []
        legal_actions = None

        for result in worker_results:
            if "infoset_data" not in result:
                continue

            if key in result["infoset_data"]:
                data = result["infoset_data"][key]
                regrets_list.append(data["regrets"])
                strategies_list.append(data["strategy_sum"])
                reach_counts_list.append(data["reach_count"])
                cumulative_utilities_list.append(data["cumulative_utility"])
                if legal_actions is None:
                    legal_actions = data["legal_actions"]

        if not regrets_list:
            continue

        # Get or create infoset in master storage
        # legal_actions is guaranteed to be set if regrets_list is non-empty
        assert legal_actions is not None

        # Handle action-set size mismatches via padding
        # This occurs in MCCFR due to sampling variance, action abstraction edge cases,
        # or stack-size dependent legal actions. Padding with zeros is correct per CFR theory:
        # missing actions contribute zero regret.
        shapes = [r.shape for r in regrets_list]
        if len(set(shapes)) > 1:
            print(f"Action set size mismatch for infoset {key}: {shapes}")

            # Find maximum action count and collect all action lists
            max_actions = max(r.shape[0] for r in regrets_list)
            all_action_lists = []
            for result in worker_results:
                if key in result.get("infoset_data", {}):
                    worker_actions = result["infoset_data"][key]["legal_actions"]
                    all_action_lists.append(worker_actions)

            # Check action type consistency for overlapping positions (best-effort)
            # Due to stack-dependent legality, the same abstract infoset can map to
            # different concrete action sets. We use the longest list and warn if unstable.
            for i in range(min(len(al) for al in all_action_lists)):
                action_types_at_i = {al[i].type for al in all_action_lists if i < len(al)}
                if len(action_types_at_i) > 1:
                    # Different action types at same position - indicates stack-dependent actions
                    action_examples = [al[i] for al in all_action_lists if i < len(al)][:3]
                    print(
                        f"  Warning: Inconsistent action types at position {i}: {action_examples}. "
                        f"Using longest action list for {key}."
                    )
                    break  # Only warn once per infoset

            # Use the longest action list (most complete discovery)
            # This is a heuristic: we assume the worker that saw more actions
            # encountered a more permissive game state (larger stacks, deeper tree)
            legal_actions = max(all_action_lists, key=len)

            # Pad all arrays to max size with zeros
            # Padding is purely numerical - we do NOT invent actions by duplication
            padded_regrets = []
            padded_strategies = []

            for regrets, strategies in zip(regrets_list, strategies_list):
                if regrets.shape[0] < max_actions:
                    # Pad with zeros (missing actions have zero regret/strategy)
                    pad_size = max_actions - regrets.shape[0]
                    regrets = np.pad(regrets, (0, pad_size), mode="constant", constant_values=0)
                    strategies = np.pad(
                        strategies, (0, pad_size), mode="constant", constant_values=0
                    )

                padded_regrets.append(regrets)
                padded_strategies.append(strategies)

            regrets_list = padded_regrets
            strategies_list = padded_strategies

        infoset = solver_storage.get_or_create_infoset(key, legal_actions)

        # Ensure master infoset has correct size (may need extending with newly discovered actions)
        if infoset.num_actions < regrets_list[0].shape[0]:
            # Master infoset has fewer actions than merged data
            # This happens when workers discovered additional legal actions
            target_size = regrets_list[0].shape[0]

            # Extend legal_actions with newly discovered actions
            infoset.legal_actions = legal_actions
            infoset.num_actions = target_size

            # Pad master infoset's arrays
            pad_size = target_size - len(infoset.regrets)
            infoset.regrets = np.pad(
                infoset.regrets, (0, pad_size), mode="constant", constant_values=0
            )
            infoset.strategy_sum = np.pad(
                infoset.strategy_sum, (0, pad_size), mode="constant", constant_values=0
            )

        # Sum regrets (CFR theory: regrets accumulate additively across all samples)
        if len(regrets_list) == 1:
            sum_regrets = regrets_list[0]
        else:
            sum_regrets = np.sum(regrets_list, axis=0)

        infoset.regrets += sum_regrets.astype(np.float32)

        # Sum strategies (CFR theory: strategies are accumulated)
        if len(strategies_list) == 1:
            sum_strategies = strategies_list[0]
        else:
            sum_strategies = np.sum(strategies_list, axis=0)

        infoset.strategy_sum += sum_strategies.astype(np.float32)

        # Sum reach counts (total visits across all workers)
        infoset.reach_count += sum(reach_counts_list)

        # Sum cumulative utilities (for proper averaging later)
        infoset.cumulative_utility += sum(cumulative_utilities_list)
