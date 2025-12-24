"""
Training orchestration for MCCFR solver.

Manages the complete training loop: solver creation, iteration execution,
checkpointing, metrics tracking, and progress reporting.

Supports both sequential and parallel (multiprocessing) training modes.
"""

import multiprocessing as mp
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from src.training import components
from src.training.metrics import MetricsTracker
from src.training.run_tracker import RunTracker
from src.utils.config import Config


def _worker_process(
    worker_id: int,
    num_iterations: int,
    config_dict: Dict,
    serialized_action_abstraction: bytes,
    serialized_card_abstraction: bytes,
    seed: int,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process that runs MCCFR iterations independently.

    Each worker:
    1. Deserializes pre-built abstractions (much faster than rebuilding)
    2. Creates its own solver with independent storage
    3. Runs N iterations with unique random seed
    4. Returns accumulated regrets/strategies for merging

    Args:
        worker_id: Unique worker identifier
        num_iterations: Number of iterations for this worker
        config_dict: Configuration dictionary
        serialized_action_abstraction: Pickled BettingActions
        serialized_card_abstraction: Pickled BucketingStrategy
        seed: Random seed for this worker
        result_queue: Queue to return results
    """
    try:
        from src.solver.mccfr import MCCFRSolver
        from src.solver.storage import InMemoryStorage

        # Recreate config
        config = Config.from_dict(config_dict)

        # Deserialize pre-built abstractions (much faster than rebuilding from disk)
        action_abstraction = pickle.loads(serialized_action_abstraction)
        card_abstraction = pickle.loads(serialized_card_abstraction)

        # Workers always use in-memory storage
        storage = InMemoryStorage()

        # Create solver config with unique seed
        solver_config = config.get_section("game").copy()
        solver_config.update(config.get_section("system"))
        solver_config["seed"] = seed  # Unique seed per worker

        # Build solver
        solver = MCCFRSolver(
            action_abstraction=action_abstraction,
            card_abstraction=card_abstraction,
            storage=storage,
            config=solver_config,
        )

        # Run iterations
        utilities = []
        for _ in range(num_iterations):
            util = solver.train_iteration()
            utilities.append(util)

        # Extract infoset data for merging
        infoset_data = {}
        for key, infoset in storage.infosets.items():
            infoset_data[key] = {
                "regrets": infoset.regrets.copy(),
                "strategy_sum": infoset.strategy_sum.copy(),
                "legal_actions": infoset.legal_actions,
            }

        # Return results
        result_queue.put(
            {
                "worker_id": worker_id,
                "utilities": utilities,
                "infoset_data": infoset_data,
                "num_infosets": len(infoset_data),
            }
        )

    except Exception as e:
        result_queue.put(
            {
                "worker_id": worker_id,
                "error": str(e),
            }
        )


class TrainingSession:
    """
    Orchestrates MCCFR training.

    Manages solver initialization, training loop, checkpointing,
    metrics tracking, and progress reporting.

    Supports both sequential and parallel training modes.
    """

    def __init__(self, config: Config, run_id: Optional[str] = None):
        """
        Initialize trainer from configuration.

        Args:
            config: Configuration object
            run_id: Optional run ID (for resuming or explicit naming)
        """
        self.config = config

        # Determine run directory
        runs_base_dir = Path(config.get("training.runs_dir", "data/runs"))
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run-{timestamp}"

        self.run_dir = runs_base_dir / run_id

        # Initialize run tracker
        self.run_tracker = RunTracker(
            run_dir=self.run_dir,
            config_name=config.get("system.config_name", "default"),
            config=config.to_dict(),
        )

        # Build components using shared builders
        # These may fail and we don't want directories created if they do
        try:
            self.action_abstraction = components.build_action_abstraction(config)
            self.card_abstraction = components.build_card_abstraction(
                config, prompt_user=False, auto_compute=False
            )

            # Storage needs the directory to exist
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.storage = components.build_storage(config, run_dir=self.run_dir)
            self.solver = components.build_solver(
                config, self.action_abstraction, self.card_abstraction, self.storage
            )

            # Initialize metrics tracker
            self.metrics = MetricsTracker(window_size=config.get("training.log_frequency", 100))
        except Exception:
            # Don't create run metadata if initialization fails
            self.run_tracker.mark_failed(cleanup_if_empty=True)
            raise

    def _merge_worker_results(self, worker_results: List[Dict]):
        """
        Merge worker results into master storage.

        For each infoset:
        - Average regrets across workers
        - Sum strategy counts

        Args:
            worker_results: List of result dicts from workers
        """
        # Collect all infoset keys
        all_keys = set()
        for result in worker_results:
            if "infoset_data" in result:
                all_keys.update(result["infoset_data"].keys())

        # Merge each infoset
        for key in all_keys:
            # Collect regrets and strategies from all workers
            regrets_list = []
            strategies_list = []
            legal_actions = None

            for result in worker_results:
                if "infoset_data" not in result:
                    continue

                if key in result["infoset_data"]:
                    data = result["infoset_data"][key]
                    regrets_list.append(data["regrets"])
                    strategies_list.append(data["strategy_sum"])
                    if legal_actions is None:
                        legal_actions = data["legal_actions"]

            if not regrets_list:
                continue

            # Verify all arrays are same shape (should be for same infoset)
            shapes = [r.shape for r in regrets_list]
            if len(set(shapes)) > 1:
                print(f"Action set size mismatch for infoset {key}: {shapes}")

                # Different workers saw different action sets - skip this infoset
                # This can happen due to race conditions in independent rollouts
                continue

            # Get or create infoset in master storage
            infoset = self.solver.storage.get_or_create_infoset(key, legal_actions)

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

    def _train_parallel(
        self,
        num_iterations: int,
        num_workers: int,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Run parallel training using multiprocessing.

        Args:
            num_iterations: Total iterations to run
            num_workers: Number of parallel workers
            batch_size: Iterations per batch (default: num_workers * 10)

        Returns:
            Training results dictionary
        """
        if batch_size is None:
            # Each worker does ~10 iterations per batch
            batch_size = num_workers * 10

        checkpoint_freq = self.config.get("training.checkpoint_frequency", 100)
        verbose = self.config.get("training.verbose", True)

        # Initialize run tracker (creates directory and .run.json)
        self.run_tracker.initialize()

        if verbose:
            print("\n🚀 Parallel Training")
            print(f"   Workers: {num_workers}")
            print(f"   Iterations: {num_iterations}")
            print(f"   Batch size: {batch_size}")
            print(f"   Expected speedup: {num_workers:.1f}x\n")

        training_start_time = time.time()
        completed_iterations = 0

        # Serialize abstractions once (avoids rebuilding in each worker)
        serialized_action_abstraction = pickle.dumps(self.solver.action_abstraction)
        serialized_card_abstraction = pickle.dumps(self.solver.card_abstraction)
        if verbose:
            print(
                f"   Serialized abstractions: {len(serialized_action_abstraction) + len(serialized_card_abstraction):,} bytes"
            )

        # Progress bar for batches
        num_batches = (num_iterations + batch_size - 1) // batch_size
        if verbose:
            batch_iterator: Union[range, tqdm] = tqdm(
                range(num_batches),
                desc="Training batches",
                unit="batch",
            )
        else:
            batch_iterator = range(num_batches)

        try:
            for batch_idx in batch_iterator:
                # Determine iterations for this batch
                remaining = num_iterations - completed_iterations
                current_batch_size = min(batch_size, remaining)

                # Split work among workers
                iters_per_worker = current_batch_size // num_workers
                extra_iters = current_batch_size % num_workers

                # Start worker processes
                result_queue: mp.Queue = mp.Queue()
                processes = []

                for worker_id in range(num_workers):
                    # Assign iterations to this worker
                    worker_iters = iters_per_worker + (1 if worker_id < extra_iters else 0)

                    if worker_iters == 0:
                        continue

                    # Unique seed per worker
                    worker_seed = (completed_iterations + worker_id) * 1000 + batch_idx

                    p = mp.Process(
                        target=_worker_process,
                        args=(
                            worker_id,
                            worker_iters,
                            self.config.to_dict(),
                            serialized_action_abstraction,
                            serialized_card_abstraction,
                            worker_seed,
                            result_queue,
                        ),
                    )
                    p.start()
                    processes.append(p)

                # Collect results
                worker_results = []
                for _ in range(len(processes)):
                    result = result_queue.get()

                    if "error" in result:
                        print(f"Worker {result['worker_id']} error: {result['error']}")
                        continue

                    worker_results.append(result)

                    # Log utilities
                    for util in result["utilities"]:
                        completed_iterations += 1
                        self.metrics.log_iteration(
                            iteration=completed_iterations,
                            utility=util,
                            num_infosets=result["num_infosets"],
                        )

                # Wait for all processes
                for p in processes:
                    p.join()

                # Merge worker results into master storage
                self._merge_worker_results(worker_results)

                # Update solver iteration count
                self.solver.iteration = completed_iterations

                # Update progress bar
                if verbose and isinstance(batch_iterator, tqdm):
                    summary = self.metrics.get_summary()
                    batch_iterator.set_postfix(
                        {
                            "iters": completed_iterations,
                            "util": f"{summary['avg_utility']:+.2f}",
                            "infosets": f"{summary['avg_infosets']:.0f}",
                        }
                    )

                # Checkpoint if needed
                if completed_iterations % checkpoint_freq < batch_size:
                    elapsed_time = time.time() - training_start_time
                    self.run_tracker.update(
                        iterations=completed_iterations,
                        runtime_seconds=elapsed_time,
                        num_infosets=self.solver.num_infosets(),
                    )
                    self.storage.checkpoint(completed_iterations)

        except KeyboardInterrupt:
            if verbose:
                print("\n⚠️  Training interrupted by user")

        finally:
            # Final checkpoint
            elapsed_time = time.time() - training_start_time
            self.run_tracker.update(
                iterations=completed_iterations,
                runtime_seconds=elapsed_time,
                num_infosets=self.solver.num_infosets(),
            )
            self.storage.checkpoint(completed_iterations)
            self.run_tracker.mark_completed()

            if verbose:
                print("\n✅ Training complete!")
                print(f"   Iterations: {completed_iterations}")
                print(f"   Time: {elapsed_time:.1f}s")
                if completed_iterations > 0:
                    print(f"   Speed: {completed_iterations / elapsed_time:.2f} iter/s")
                    print(f"   Expected speedup: ~{num_workers:.1f}x (parallel)")

        # Return final results
        return {
            "total_iterations": completed_iterations,
            "final_infosets": self.solver.num_infosets(),
            "avg_utility": self.metrics.get_avg_utility(),
            "elapsed_time": elapsed_time,
        }

    def train(
        self,
        num_iterations: Optional[int] = None,
        resume: bool = False,
        use_parallel: bool = True,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Run training loop.

        Args:
            num_iterations: Number of iterations (overrides config if provided)
            resume: Whether to resume from latest checkpoint
            use_parallel: Whether to use parallel training (default: True)
            num_workers: Number of parallel workers (default: CPU count)
            batch_size: Iterations per batch for parallel mode (default: num_workers * 10)

        Returns:
            Training results dictionary
        """
        # Get iteration count
        if num_iterations is None:
            num_iterations = int(self.config.get("training.num_iterations", 1000))

        # Resume from snapshot if requested
        if resume:
            print(f"Resuming from run: {self.run_dir.name}")
            start_iteration = self.run_tracker.metadata.get("iterations", 0)
            if start_iteration > 0:
                print(f"Continuing from iteration {start_iteration}")
                num_iterations = max(num_iterations, start_iteration)

        # Use parallel training if requested
        if use_parallel:
            if num_workers is None:
                num_workers = mp.cpu_count()
            return self._train_parallel(num_iterations, num_workers, batch_size)

        # Sequential training
        checkpoint_freq = self.config.get("training.checkpoint_frequency", 100)
        log_freq = self.config.get("training.log_frequency", 10)
        verbose = self.config.get("training.verbose", True)

        start_iteration = self.run_tracker.metadata.get("iterations", 0)

        # Initialize run tracker (creates directory and .run.json)
        self.run_tracker.initialize()

        # Track training time
        training_start_time = time.time()

        # Training loop
        if verbose:
            iterator: Union[range, tqdm] = tqdm(
                range(start_iteration, num_iterations),
                desc="Training",
                unit="iter",
            )
        else:
            iterator = range(start_iteration, num_iterations)

        try:
            for i in iterator:
                # Run one iteration
                utility = self.solver.train_iteration()

                # Log metrics
                self.metrics.log_iteration(
                    iteration=i + 1,
                    utility=utility,
                    num_infosets=self.solver.num_infosets(),
                )

                # Periodic logging
                if (i + 1) % log_freq == 0 and verbose:
                    summary = self.metrics.get_summary()
                    if isinstance(iterator, tqdm):
                        iterator.set_postfix(
                            {
                                "util": f"{summary['avg_utility']:+.2f}",
                                "infosets": f"{summary['avg_infosets']:.0f}",
                            }
                        )

                # Periodic snapshotting
                if (i + 1) % checkpoint_freq == 0:
                    elapsed_time = time.time() - training_start_time

                    # Update progress
                    self.run_tracker.update(
                        iterations=i + 1,
                        runtime_seconds=elapsed_time,
                        num_infosets=self.solver.num_infosets(),
                    )

                    # Trigger storage checkpoint
                    self.storage.checkpoint(i + 1)

            # Final checkpoint
            elapsed_time = time.time() - training_start_time

            # Update final stats
            self.run_tracker.update(
                iterations=num_iterations,
                runtime_seconds=elapsed_time,
                num_infosets=self.solver.num_infosets(),
            )

            # Save final checkpoint
            self.storage.checkpoint(num_iterations)

            # Mark run as completed
            self.run_tracker.mark_completed()

            # Print final summary
            if verbose:
                self.metrics.print_summary()

            return {
                "total_iterations": num_iterations,
                "final_infosets": self.solver.num_infosets(),
                "avg_utility": self.metrics.get_avg_utility(),
                "elapsed_time": elapsed_time,
            }

        except KeyboardInterrupt:
            # User interrupted - save progress before exiting
            if verbose:
                print("\n⚠️  Training interrupted by user")

            elapsed_time = time.time() - training_start_time
            current_iter = i + 1 if "i" in locals() else start_iteration  # type: ignore

            # Save progress
            self.run_tracker.update(
                iterations=current_iter,
                runtime_seconds=elapsed_time,
                num_infosets=self.solver.num_infosets(),
            )
            self.storage.checkpoint(current_iter)

            if verbose:
                print(f"✅ Progress saved at iteration {current_iter}")

            raise

        except Exception:
            # Mark run as failed on exception
            self.run_tracker.mark_failed()
            raise

    def evaluate(
        self,
        num_samples: int = 10000,
        num_rollouts_per_infoset: int = 100,
    ) -> Dict:
        """
        Evaluate current solver using exploitability estimation.

        Args:
            num_samples: Number of game samples for exploitability estimation
            num_rollouts_per_infoset: Number of rollouts per infoset for BR approximation

        Returns:
            Evaluation metrics including exploitability estimates
        """
        from src.evaluation.exploitability import compute_exploitability
        from src.solver.mccfr import MCCFRSolver

        # Cast to MCCFRSolver for type checking
        if not isinstance(self.solver, MCCFRSolver):
            raise TypeError("Exploitability computation requires MCCFRSolver")

        results = compute_exploitability(
            self.solver,
            num_samples=num_samples,
            num_rollouts_per_infoset=num_rollouts_per_infoset,
        )

        return {
            "num_infosets": self.solver.num_infosets(),
            "exploitability_mbb": results["exploitability_mbb"],
            "std_error_mbb": results["std_error_mbb"],
            "confidence_95_mbb": results["confidence_95_mbb"],
        }

    def __str__(self) -> str:
        """String representation."""
        return f"TrainingSession(solver={self.solver}, config={self.config})"
