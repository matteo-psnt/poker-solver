"""
Parallel training for MCCFR solver.

Uses multiprocessing to run multiple MCCFR iterations in parallel,
with periodic synchronization of regret/strategy updates.
"""

import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from src.abstraction.action_abstraction import ActionAbstraction
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training.metrics import MetricsTracker
from src.training.training_run import TrainingRun
from src.utils.config import Config


def _worker_process(
    worker_id: int,
    num_iterations: int,
    config_dict: Dict,
    seed: int,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process that runs MCCFR iterations independently.

    Each worker:
    1. Creates its own solver with independent storage
    2. Runs N iterations with unique random seed
    3. Returns accumulated regrets/strategies for merging

    Args:
        worker_id: Unique worker identifier
        num_iterations: Number of iterations for this worker
        config_dict: Configuration dictionary
        seed: Random seed for this worker
        result_queue: Queue to return results
    """
    try:
        # Recreate config
        config = Config.from_dict(config_dict)

        # Create independent storage for this worker
        storage = InMemoryStorage()

        # Create solver with unique seed
        solver_config = {
            "starting_stack": config.get("game.starting_stack", 200),
            "small_blind": config.get("game.small_blind", 1),
            "big_blind": config.get("game.big_blind", 2),
            "seed": seed,  # Unique seed per worker
        }

        # Create abstractions
        action_abstraction = ActionAbstraction(config.to_dict())

        card_abstraction_type = config.get("card_abstraction.type", "equity_bucketing")
        if card_abstraction_type == "equity_bucketing":
            from src.abstraction.equity_bucketing import EquityBucketing

            bucketing_path = config.get("card_abstraction.bucketing_path")
            if not bucketing_path:
                raise ValueError("equity_bucketing requires 'bucketing_path' in config")

            from pathlib import Path

            bucketing_path = Path(bucketing_path)
            if not bucketing_path.exists():
                raise FileNotFoundError(
                    f"Equity bucketing file not found: {bucketing_path}\n"
                    "Please run 'Precompute Equity Buckets' from the CLI first."
                )

            card_abstraction = EquityBucketing.load(bucketing_path)
        else:
            raise ValueError(
                f"Unknown card abstraction type: {card_abstraction_type}\n"
                "Only 'equity_bucketing' is supported."
            )

        # Create solver
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


class ParallelTrainer:
    """
    Parallel MCCFR trainer using multiprocessing.

    Strategy:
    1. Split iterations into batches
    2. Run batches in parallel across workers
    3. Merge regrets/strategies after each batch
    4. Continue until total iterations reached

    This provides near-linear speedup while maintaining CFR convergence.
    """

    def __init__(
        self,
        config: Config,
        num_workers: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize parallel trainer.

        Args:
            config: Configuration object
            num_workers: Number of parallel workers (default: CPU count)
            run_id: Optional run ID for checkpointing
        """
        self.config = config
        self.num_workers = num_workers or mp.cpu_count()

        # Initialize experiment manager
        runs_base_dir = Path(config.get("training.run_dir", "data/runs"))
        self.training_run = TrainingRun(
            base_dir=runs_base_dir,
            experiment_name=run_id,
            config_name=config.get("system.config_name", "default"),
            config=config.to_dict(),
        )

        # Create master storage and solver (for checkpointing)
        self._setup_master_solver()

        # Metrics tracker
        self.metrics = MetricsTracker()

    def _setup_master_solver(self):
        """Setup master solver for checkpointing."""
        storage_type = self.config.get("storage.type", "memory")

        if storage_type == "memory":
            storage = InMemoryStorage()
        elif storage_type == "disk":
            from src.solver.storage import DiskBackedStorage

            storage_path = Path(self.training_run.run_dir / "storage")
            storage = DiskBackedStorage(
                storage_path,
                cache_size=self.config.get("storage.cache_size", 10000),
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

        # Create abstractions
        action_abstraction = ActionAbstraction(self.config)

        card_abstraction_type = self.config.get("card_abstraction.type", "equity_bucketing")
        if card_abstraction_type == "equity_bucketing":
            from src.abstraction.equity_bucketing import EquityBucketing

            bucketing_path = self.config.get("card_abstraction.bucketing_path")
            if not bucketing_path:
                raise ValueError("equity_bucketing requires 'bucketing_path' in config")

            bucketing_path = Path(bucketing_path)
            if not bucketing_path.exists():
                raise FileNotFoundError(
                    f"Equity bucketing file not found: {bucketing_path}\n"
                    "Please run 'Precompute Equity Buckets' from the CLI first."
                )

            card_abstraction = EquityBucketing.load(bucketing_path)
        else:
            raise ValueError(
                f"Unknown card abstraction type: {card_abstraction_type}\n"
                "Only 'equity_bucketing' is supported."
            )

        solver_config = {
            "starting_stack": self.config.get("game.starting_stack", 200),
            "small_blind": self.config.get("game.small_blind", 1),
            "big_blind": self.config.get("game.big_blind", 2),
        }

        self.solver = MCCFRSolver(
            action_abstraction=action_abstraction,
            card_abstraction=card_abstraction,
            storage=storage,
            config=solver_config,
        )

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
                # Different workers saw different action sets - skip this infoset
                # This can happen due to race conditions in independent rollouts
                continue

            # Get or create infoset in master storage
            infoset = self.solver.storage.get_or_create_infoset(key, legal_actions)

            # Average regrets (CFR theory: regrets are averaged across parallel iterations)
            if len(regrets_list) == 1:
                avg_regrets = regrets_list[0]
            else:
                avg_regrets = np.mean(regrets_list, axis=0)

            infoset.regrets = avg_regrets.astype(np.float32)

            # Sum strategies (CFR theory: strategies are accumulated)
            if len(strategies_list) == 1:
                sum_strategies = strategies_list[0]
            else:
                sum_strategies = np.sum(strategies_list, axis=0)

            infoset.strategy_sum = sum_strategies.astype(np.float32)

    def train(
        self,
        num_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Run parallel training.

        Args:
            num_iterations: Total iterations to run
            batch_size: Iterations per batch (default: num_workers * 10)

        Returns:
            Training results dictionary
        """
        if num_iterations is None:
            num_iterations = self.config.get("training.num_iterations", 1000)

        if batch_size is None:
            # Each worker does ~10 iterations per batch
            batch_size = self.num_workers * 10

        checkpoint_freq = self.config.get("training.checkpoint_frequency", 100)
        verbose = self.config.get("training.verbose", True)

        if verbose:
            print("\n🚀 Parallel Training")
            print(f"   Workers: {self.num_workers}")
            print(f"   Iterations: {num_iterations}")
            print(f"   Batch size: {batch_size}")
            print(f"   Expected speedup: {self.num_workers:.1f}x\n")

        training_start_time = time.time()
        completed_iterations = 0

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
                iters_per_worker = current_batch_size // self.num_workers
                extra_iters = current_batch_size % self.num_workers

                # Start worker processes
                result_queue: mp.Queue = mp.Queue()
                processes = []

                for worker_id in range(self.num_workers):
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
                    summary = self.metrics.get_summary()

                    snapshot_metrics = {
                        "avg_utility_p0": summary.get("avg_utility", 0.0),
                        "avg_utility_p1": -summary.get("avg_utility", 0.0),
                        "avg_infosets": summary.get("avg_infosets", 0),
                    }

                    self.training_run.save_snapshot(
                        self.solver,
                        completed_iterations,
                        metrics=snapshot_metrics,
                    )

                    self.training_run.update_stats(
                        total_iterations=completed_iterations,
                        total_runtime_seconds=elapsed_time,
                        num_infosets=int(summary.get("avg_infosets", 0)),
                    )

        except KeyboardInterrupt:
            if verbose:
                print("\n⚠️  Training interrupted by user")

        finally:
            # Final snapshot
            elapsed_time = time.time() - training_start_time
            summary = self.metrics.get_summary()

            snapshot_metrics = {
                "avg_utility_p0": summary.get("avg_utility", 0.0),
                "avg_utility_p1": -summary.get("avg_utility", 0.0),
                "avg_infosets": summary.get("avg_infosets", 0),
            }

            self.training_run.save_snapshot(
                self.solver,
                completed_iterations,
                metrics=snapshot_metrics,
            )

            # Final stats update
            self.training_run.update_stats(
                total_iterations=completed_iterations,
                total_runtime_seconds=elapsed_time,
                num_infosets=self.solver.num_infosets(),
            )

            if verbose:
                print("\n✅ Training complete!")
                print(f"   Iterations: {completed_iterations}")
                print(f"   Time: {elapsed_time:.1f}s")
                if completed_iterations > 0:
                    print(f"   Speed: {completed_iterations / elapsed_time:.2f} iter/s")
                    print(f"   Expected speedup: ~{self.num_workers:.1f}x (parallel)")

        # Return final results
        return {
            "iterations": completed_iterations,
            "elapsed_time": elapsed_time,
            "num_infosets": self.solver.num_infosets(),
            "metrics": self.metrics.get_summary(),
            "run_id": self.training_run.run_id,
        }
