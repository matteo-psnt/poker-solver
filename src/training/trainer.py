"""
Training orchestration for MCCFR solver.

Manages the complete training loop: solver creation, iteration execution,
checkpointing, metrics tracking, and progress reporting.

Supports both sequential and parallel (multiprocessing) training modes.
"""

import json
import multiprocessing as mp
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from tqdm import tqdm

from src.training import components
from src.training.metrics import MetricsTracker
from src.training.parallel import WorkerManager, merge_worker_results
from src.training.run_tracker import RunTracker
from src.utils.config import Config


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

    @classmethod
    def resume(
        cls, run_dir: Union[str, Path], checkpoint_id: Optional[int] = None
    ) -> "TrainingSession":
        """
        Resume training from a checkpoint.

        Loads the latest (or specified) checkpoint from the run directory,
        restoring solver state, storage mappings, and iteration counter.

        Args:
            run_dir: Path to the run directory
            checkpoint_id: Optional specific checkpoint iteration to load (default: latest)

        Returns:
            TrainingSession configured to continue from checkpoint

        Raises:
            FileNotFoundError: If run directory or checkpoint artifacts don't exist
            ValueError: If checkpoint is invalid or incomplete
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")

        # Load run metadata
        metadata_file = run_path / ".run.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Run metadata not found: {metadata_file}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Find checkpoint to load
        checkpoint_iter = (
            checkpoint_id if checkpoint_id is not None else cls._find_latest_checkpoint(run_path)
        )

        if checkpoint_iter is None:
            raise FileNotFoundError(f"No checkpoint found in {run_path}")

        # Verify checkpoint artifacts exist
        required_files = ["regrets.h5", "strategies.h5", "key_mapping.pkl"]
        missing_files = [f for f in required_files if not (run_path / f).exists()]

        if missing_files:
            raise ValueError(
                f"Checkpoint is incomplete. Missing files: {missing_files}\n"
                f"Required: {required_files}"
            )

        # Reconstruct config from metadata
        config = Config.from_dict(metadata.get("config", {}))

        # Create a new session instance with the same run_id
        session = cls(config, run_id=run_path.name)

        # The storage should already have loaded the checkpoint data in its __init__
        # via _load_metadata(), but we need to verify it worked
        if session.storage.num_infosets() == 0:
            raise ValueError(
                f"Failed to load checkpoint data. Storage has 0 infosets.\n"
                f"Expected to load from: {run_path}"
            )

        # Restore solver iteration counter
        session.solver.iteration = checkpoint_iter

        # Mark run as resumed
        session.run_tracker.mark_resumed()

        print(f"✅ Resumed from checkpoint at iteration {checkpoint_iter}")
        print(f"   Loaded {session.storage.num_infosets()} infosets")

        return session

    @staticmethod
    def _find_latest_checkpoint(run_dir: Path) -> Optional[int]:
        """
        Find the latest checkpoint iteration in run directory.

        Looks for checkpoint artifacts and returns the highest iteration number.
        Currently uses the iteration count from metadata.

        Args:
            run_dir: Path to run directory

        Returns:
            Latest checkpoint iteration number, or None if no checkpoint found
        """
        metadata_file = run_dir / ".run.json"
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            iterations = metadata.get("iterations", 0)
            return iterations if iterations > 0 else None
        except Exception:
            return None
            raise

    def _train_parallel(
        self,
        num_iterations: int,
        num_workers: int,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Run parallel training using persistent worker pool.

        Uses WorkerManager to maintain persistent workers across batches,
        eliminating process spawning overhead.

        Args:
            num_iterations: Total iterations to run
            num_workers: Number of parallel workers
            batch_size: Iterations per batch (default: num_workers * 10)

        Returns:
            Training results dictionary
        """
        if batch_size is None:
            batch_size = 200 * num_workers

        checkpoint_freq = self.config.get("training.checkpoint_frequency", 100)
        verbose = self.config.get("training.verbose", True)

        # Initialize run tracker (creates directory and .run.json)
        self.run_tracker.initialize()

        if verbose:
            print("\n🚀 Parallel Training")
            print(f"   Workers: {num_workers}")
            print(f"   Iterations: {num_iterations}")
            print(f"   Batch size: {batch_size}")
            print(f"   Expected speedup: {num_workers:.1f}x")

        training_start_time = time.time()
        completed_iterations = 0

        # Serialize abstractions once (sent to workers at startup)
        serialized_action_abstraction = pickle.dumps(self.solver.action_abstraction)
        serialized_card_abstraction = pickle.dumps(self.solver.card_abstraction)

        if verbose:
            abstraction_size = len(serialized_action_abstraction) + len(serialized_card_abstraction)
            print(f"   Serialized abstractions: {abstraction_size:,} bytes")
            print("   Starting worker pool...")

        pool_start_time = time.time()

        # Create persistent worker pool (workers initialized once)
        with WorkerManager(
            num_workers=num_workers,
            config_dict=self.config.to_dict(),
            serialized_action_abstraction=serialized_action_abstraction,
            serialized_card_abstraction=serialized_card_abstraction,
            base_seed=self.config.get("system.seed", 42) or 42,
        ) as worker_manager:
            pool_init_time = time.time() - pool_start_time

            if verbose:
                print(f"   Worker pool ready ({pool_init_time:.2f}s)\n")

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
                    iters_per_worker_base = current_batch_size // num_workers
                    extra_iters = current_batch_size % num_workers

                    # Build iteration list for each worker
                    iterations_per_worker = [
                        iters_per_worker_base + (1 if i < extra_iters else 0)
                        for i in range(num_workers)
                    ]

                    # Submit batch to persistent workers (no process spawning!)
                    worker_results = worker_manager.submit_batch(
                        iterations_per_worker=iterations_per_worker, batch_id=batch_idx
                    )

                    # Collect utilities from all workers
                    batch_utilities = []
                    for result in worker_results:
                        batch_utilities.extend(result["utilities"])

                    # Merge worker results into master storage
                    merge_worker_results(self.solver.storage, worker_results)

                    # Update solver iteration count
                    self.solver.iteration = completed_iterations + len(batch_utilities)

                    # Log metrics using master solver state after merge
                    # This ensures num_infosets reflects the actual merged solver, not worker-local counts
                    master_num_infosets = self.solver.num_infosets()

                    # Create infoset sampler for solver-quality metrics
                    def sample_infosets(n: int):
                        """Sample n random infosets from solver storage."""
                        if hasattr(self.solver.storage, "infosets"):
                            all_infosets = list(self.solver.storage.infosets.values())
                            if len(all_infosets) <= n:
                                return all_infosets
                            indices = np.random.choice(len(all_infosets), size=n, replace=False)
                            return [all_infosets[i] for i in indices]
                        return []

                    for util in batch_utilities:
                        completed_iterations += 1
                        self.metrics.log_iteration(
                            iteration=completed_iterations,
                            utility=util,
                            num_infosets=master_num_infosets,
                            infoset_sampler=sample_infosets,
                        )

                    # Update progress bar with solver-quality metrics
                    if verbose and isinstance(batch_iterator, tqdm):
                        compact_summary = self.metrics.get_compact_summary()
                        batch_iterator.set_postfix_str(
                            f"iter={completed_iterations} | {compact_summary}"
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
                        print(f"   Expected speedup: ~{num_workers:.1f}x (persistent pool)")

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
            resume: (DEPRECATED) Only adjusts loop bounds, does not load checkpoint state.
                Use TrainingSession.resume() classmethod instead for proper checkpoint restoration.
            use_parallel: Whether to use parallel training (default: True)
            num_workers: Number of parallel workers (default: CPU count)
            batch_size: Iterations per batch for parallel mode (default: num_workers * 10)

        Returns:
            Training results dictionary

        Note:
            For proper resume functionality that restores solver state from checkpoints,
            use the TrainingSession.resume() classmethod instead of resume=True parameter.
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

        # Create infoset sampler for solver-quality metrics
        def sample_infosets(n: int):
            """Sample n random infosets from solver storage."""
            if hasattr(self.solver.storage, "infosets"):
                all_infosets = list(self.solver.storage.infosets.values())
                if len(all_infosets) <= n:
                    return all_infosets
                indices = np.random.choice(len(all_infosets), size=n, replace=False)
                return [all_infosets[i] for i in indices]
            return []

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

                # Log metrics with solver-quality indicators
                self.metrics.log_iteration(
                    iteration=i + 1,
                    utility=utility,
                    num_infosets=self.solver.num_infosets(),
                    infoset_sampler=sample_infosets,
                )

                # Periodic logging
                if (i + 1) % log_freq == 0 and verbose:
                    summary = self.metrics.get_summary()
                    if isinstance(iterator, tqdm):
                        # Use compact summary for progress bar
                        compact_summary = self.metrics.get_compact_summary()
                        iterator.set_postfix_str(compact_summary)

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
