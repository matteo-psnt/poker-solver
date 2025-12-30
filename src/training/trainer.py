"""
Training orchestration for MCCFR solver.

Manages the complete training loop: solver creation, iteration execution,
checkpointing, metrics tracking, and progress reporting.

Training uses parallel multiprocessing with hash-partitioned shared memory.
"""

import concurrent.futures
import multiprocessing as mp
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.exploitability import compute_exploitability
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.helpers import get_missing_checkpoint_files
from src.training import components
from src.training.metrics import MetricsTracker
from src.training.parallel_manager import SharedArrayWorkerManager
from src.training.run_tracker import RunTracker
from src.utils.config import Config


class TrainingSession:
    """
    Orchestrates MCCFR training.

    Manages solver initialization, training loop, checkpointing,
    metrics tracking, and progress reporting.

    Training uses parallel multiprocessing with hash-partitioned shared memory.
    """

    def __init__(
        self,
        config: Config,
        run_id: str | None = None,
        run_tracker: RunTracker | None = None,
    ):
        """
        Initialize trainer from configuration.

        Args:
            config: Configuration object
            run_id: Optional run ID (for resuming or explicit naming)
        """
        self.config = config

        # Determine run directory
        self.run_tracker = run_tracker
        if self.run_tracker:
            self.run_dir = self.run_tracker.run_dir
        else:
            runs_base_dir = Path(self.config.training.runs_dir)
            if run_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"run-{timestamp}"
            self.run_dir = runs_base_dir / run_id

        self._checkpoint_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending_checkpoint: concurrent.futures.Future[float] | None = None

        # Build components using shared builders
        # These may fail and we don't want directories created if they do
        try:
            self.action_abstraction = components.build_action_abstraction(config)
            self.card_abstraction = components.build_card_abstraction(
                config, prompt_user=False, auto_compute=False
            )
            action_config_hash = self.action_abstraction.get_config_hash()

            if self.run_tracker is None:
                self.run_tracker = RunTracker(
                    run_dir=self.run_dir,
                    config_name=self.config.system.config_name,
                    config=config,
                    action_config_hash=action_config_hash,
                )
            else:
                self.run_tracker.verify_action_config_hash(action_config_hash)

            # Storage needs the directory to exist
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.storage = components.build_storage(
                config,
                run_dir=self.run_dir,
                run_metadata=self.run_tracker.metadata,
            )
            self.solver = components.build_solver(
                config, self.action_abstraction, self.card_abstraction, self.storage
            )

            # Initialize metrics tracker
            self.metrics = MetricsTracker()

            # Initialize checkpoint executor (used for async checkpoints)
            if self.config.storage.checkpoint_enabled:
                self._checkpoint_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="checkpoint"
                )
        except Exception:
            # Don't create run metadata if initialization fails
            if self.run_tracker is not None:
                self.run_tracker.mark_failed(cleanup_if_empty=True)
            raise

    @classmethod
    def resume(cls, run_dir: str | Path, checkpoint_id: int | None = None) -> "TrainingSession":
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
        run_tracker = RunTracker.load(run_path)
        metadata = run_tracker.metadata

        # Find checkpoint to load
        if checkpoint_id is not None:
            checkpoint_iter = checkpoint_id
        else:
            checkpoint_iter = metadata.iterations if metadata.iterations > 0 else None

        if checkpoint_iter is None:
            raise FileNotFoundError(f"No checkpoint found in {run_path}")

        # Verify checkpoint artifacts exist
        missing_files = get_missing_checkpoint_files(run_path)
        if missing_files:
            raise ValueError(f"Checkpoint is incomplete. Missing files: {missing_files}")

        # Reconstruct config from metadata
        if metadata.config is None:
            raise ValueError(f"Missing config in run metadata: {run_tracker.metadata_path}")
        config = metadata.config

        # Create a new session instance with the same run_id
        session = cls(config, run_id=run_path.name, run_tracker=run_tracker)

        # The storage should already have loaded the checkpoint data in its __init__
        # via _load_metadata(), but we need to verify it worked
        if session.storage.num_infosets() == 0:
            raise ValueError(
                f"Failed to load checkpoint data. Storage has 0 infosets.\n"
                f"Expected to load from: {run_path}"
            )

        # Restore solver iteration counter
        assert checkpoint_iter is not None
        session.solver.iteration = checkpoint_iter

        # Mark run as resumed
        assert session.run_tracker is not None
        session.run_tracker.mark_resumed()

        print(f"✅ Resumed from checkpoint at iteration {checkpoint_iter}")

        return session

    def __del__(self):
        """Cleanup on deletion."""
        self._shutdown_checkpoint_executor()

    @property
    def verbose(self) -> bool:
        """Get verbose setting from config."""
        return self.config.training.verbose

    def _checkpoint_enabled(self) -> bool:
        return self.config.storage.checkpoint_enabled

    def _checkpoint_frequency(self) -> int:
        return self.config.training.checkpoint_frequency

    def _should_checkpoint(self, iteration: int, batch_size: int) -> bool:
        if not self._checkpoint_enabled() or iteration == 0:
            return False
        return iteration % self._checkpoint_frequency() < batch_size

    def _async_checkpoint(
        self,
        worker_manager: SharedArrayWorkerManager,
        iteration: int,
        total_infosets: int,
        storage_capacity: int,
        training_start_time: float,
    ) -> None:
        if not self._checkpoint_enabled() or self._checkpoint_executor is None:
            return
        if self._pending_checkpoint is not None and not self._pending_checkpoint.done():
            if self.verbose:
                print("[Master] Previous checkpoint still running; skipping", flush=True)
            return

        self._pending_checkpoint = self._checkpoint_executor.submit(
            self._checkpoint_with_timing,
            worker_manager,
            iteration,
            total_infosets,
            storage_capacity,
            training_start_time,
        )

    def _wait_for_checkpoint(self) -> None:
        if self._pending_checkpoint is None:
            return
        if self.verbose:
            print("[Master] Waiting for background checkpoint to complete...", flush=True)
        try:
            self._pending_checkpoint.result()
        except Exception as exc:
            print(f"[Master] ERROR: Background checkpoint failed: {exc}", flush=True)
            raise
        finally:
            self._pending_checkpoint = None

    def _shutdown_checkpoint_executor(self) -> None:
        if self._checkpoint_executor is None:
            return
        self._wait_for_checkpoint()
        self._checkpoint_executor.shutdown(wait=True)

    def _checkpoint_with_timing(
        self,
        worker_manager: SharedArrayWorkerManager,
        iteration: int,
        total_infosets: int,
        storage_capacity: int,
        training_start_time: float,
    ) -> float:
        start = time.time()
        worker_manager.checkpoint(iteration)

        elapsed_time = time.time() - training_start_time
        if self.run_tracker is not None:
            self.run_tracker.update(
                iterations=iteration,
                runtime_seconds=elapsed_time,
                num_infosets=total_infosets,
                storage_capacity=storage_capacity,
            )

        checkpoint_time = time.time() - start
        if self.verbose:
            print(
                f"[Master] Checkpoint saved at iter={iteration} in {checkpoint_time:.2f}s",
                flush=True,
            )
        return checkpoint_time

    def _print_training_header(
        self,
        num_workers: int,
        num_iterations: int,
        batch_size: int,
        initial_capacity: int,
        max_actions: int,
    ) -> None:
        if not self.verbose:
            return
        print("\n🚀 Shared Array Parallel Training")
        print(f"   Workers: {num_workers}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Batch size: {batch_size}")
        print(f"   Initial capacity: {initial_capacity:,}")
        print(f"   Max actions: {max_actions}")
        print("   Mode: Live shared memory arrays")

    def _update_progress_bar(
        self,
        progress_bar: tqdm,
        iteration: int,
        total_infosets: int,
        capacity_usage: float,
    ) -> None:
        if not self.verbose or not isinstance(progress_bar, tqdm):
            return
        compact_summary = self.metrics.get_compact_summary()
        progress_bar.set_postfix_str(
            f"iter={iteration} infosets={total_infosets} "
            f"cap={capacity_usage:.0%} | {compact_summary}"
        )

    def _print_final_summary(
        self,
        total_iterations: int,
        total_infosets: int,
        elapsed_time: float,
        interrupted: bool,
        fallback_stats: dict[str, float] | None = None,
    ) -> None:
        if not self.verbose:
            return
        if interrupted:
            print("🟡 Training interrupted")
        else:
            print("✅ Shared Array Training complete!")

        print(f"   Iterations: {total_iterations}")
        print(f"   Infosets: {total_infosets:,}")
        print(f"   Time: {elapsed_time:.1f}s")

        if total_iterations > 0:
            print(f"   Speed: {total_iterations / elapsed_time:.2f} iter/s")

        if fallback_stats:
            total_lookups = int(fallback_stats.get("total_lookups", 0))
            fallback_count = int(fallback_stats.get("fallback_count", 0))
            if total_lookups > 0:
                fallback_rate = fallback_stats.get("fallback_rate", 0.0) * 100
                print(
                    f"   Abstraction fallbacks: {fallback_count:,}/{total_lookups:,} "
                    f"({fallback_rate:.2f}%)"
                )

    def _get_training_config(self, num_workers: int, batch_size: int | None) -> dict[str, Any]:
        """Parse training configuration with defaults."""
        assert self.run_tracker is not None
        if batch_size is None:
            batch_size = self.config.training.iterations_per_worker * num_workers

        # Determine initial capacity: use run metadata if resuming, else config
        initial_capacity = self.config.storage.initial_capacity
        stored_capacity = self.run_tracker.metadata.resolve_initial_capacity(initial_capacity)
        if stored_capacity != initial_capacity:
            print(
                f"[Resume] Using stored capacity {stored_capacity:,} "
                f"(config initial_capacity={initial_capacity:,})"
            )
        initial_capacity = stored_capacity

        return {
            "batch_size": batch_size,
            "verbose": self.config.training.verbose,
            "initial_capacity": initial_capacity,
            "max_actions": self.config.storage.max_actions,
            "checkpoint_enabled": self.config.storage.checkpoint_enabled,
        }

    def _train_partitioned(
        self,
        num_iterations: int,
        num_workers: int,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Run parallel training using shared array storage.

        Architecture:
        - Coordinator creates shared memory once at startup
        - Workers attach to shared memory (no recreation)
        - Flat NumPy arrays as live data structure
        - Lock-free reads, owner-only writes
        - Cross-partition updates via queues

        Args:
            num_iterations: Total iterations to run
            num_workers: Number of parallel workers
            batch_size: Iterations per batch

        Returns:
            Training results dictionary
        """
        assert self.run_tracker is not None
        # Parse configuration
        config = self._get_training_config(num_workers, batch_size)
        batch_size_val = config["batch_size"]
        verbose = config["verbose"]
        initial_capacity = config["initial_capacity"]
        max_actions = config["max_actions"]
        checkpoint_enabled = config["checkpoint_enabled"]

        # Initialize run tracker
        self.run_tracker.initialize()

        # Display header
        self._print_training_header(
            num_workers, num_iterations, batch_size_val, initial_capacity, max_actions
        )

        training_start_time = time.time()
        start_iteration = self.solver.iteration

        # Serialize abstractions
        serialized_action_abstraction = pickle.dumps(self.solver.action_abstraction)
        serialized_card_abstraction = pickle.dumps(self.solver.card_abstraction)

        if verbose:
            abstraction_size = len(serialized_action_abstraction) + len(serialized_card_abstraction)
            print(f"   Serialized abstractions: {abstraction_size:,} bytes")

        pool_start_time = time.time()

        # Create shared array worker manager
        with SharedArrayWorkerManager(
            num_workers=num_workers,
            config=self.config,
            serialized_action_abstraction=serialized_action_abstraction,
            serialized_card_abstraction=serialized_card_abstraction,
            session_id=self.run_dir.name,
            base_seed=self.config.system.seed
            if self.config.system.seed is not None
            else random.randint(0, 2**31 - 1),
            initial_capacity=initial_capacity,
            max_actions=max_actions,
            checkpoint_dir=str(self.run_dir) if checkpoint_enabled else None,
        ) as worker_manager:
            pool_init_time = time.time() - pool_start_time

            if verbose:
                print(f"   Worker pool ready ({pool_init_time:.2f}s)\n")

            # Run training loop
            num_batches = (num_iterations + batch_size_val - 1) // batch_size_val
            batch_iterator = tqdm(
                range(num_batches),
                desc="Training batches",
                unit="batch",
                disable=not verbose,
            )

            completed_iterations = 0
            total_infosets = 0
            interrupted = False
            fallback_stats: dict[str, float] | None = None
            last_capacity: int | None = None

            try:
                for batch_idx in batch_iterator:
                    # Wait for any pending checkpoint
                    self._wait_for_checkpoint()

                    # Determine iterations for this batch
                    remaining = num_iterations - completed_iterations
                    current_batch_size = min(batch_size_val, remaining)

                    # Split work among workers
                    iters_per_worker_base = current_batch_size // num_workers
                    extra_iters = current_batch_size % num_workers
                    iterations_per_worker = [
                        iters_per_worker_base + (1 if i < extra_iters else 0)
                        for i in range(num_workers)
                    ]

                    # Run batch (data flows directly through shared arrays)
                    batch_result = worker_manager.run_batch(
                        iterations_per_worker=iterations_per_worker,
                        batch_id=batch_idx,
                        start_iteration=start_iteration + completed_iterations,
                        verbose=verbose,
                    )

                    # Update state
                    batch_utilities = batch_result["utilities"]
                    completed_iterations += len(batch_utilities)
                    total_infosets = batch_result.get("num_infosets", 0)
                    max_worker_capacity = batch_result.get("max_worker_capacity", 0.0)
                    last_capacity = batch_result.get("capacity", last_capacity)

                    if "fallback_stats" in batch_result:
                        fallback_stats = batch_result["fallback_stats"]

                    if batch_result.get("interrupted"):
                        interrupted = True

                    # Exchange IDs and apply updates between batches
                    # (owner-to-requester, not global broadcast)
                    if batch_idx < num_batches - 1:
                        inter_batch_timeout = max(60.0, batch_result["batch_time"] * 2.0)
                        worker_manager.exchange_ids(
                            timeout=inter_batch_timeout,
                            verbose=verbose,
                        )
                        worker_manager.apply_pending_updates(
                            timeout=inter_batch_timeout,
                            verbose=verbose,
                        )

                    # Log metrics for each iteration in batch
                    for i, util in enumerate(batch_utilities):
                        iter_num = (
                            start_iteration + completed_iterations - len(batch_utilities) + i + 1
                        )
                        self.metrics.log_iteration(
                            iteration=iter_num,
                            utility=util,
                            num_infosets=total_infosets,
                            infoset_sampler=None,
                        )

                    # Update progress bar
                    self._update_progress_bar(
                        batch_iterator,
                        start_iteration + completed_iterations,
                        total_infosets,
                        max_worker_capacity,
                    )

                    # Checkpoint if needed
                    if self._should_checkpoint(
                        start_iteration + completed_iterations, batch_size_val
                    ):
                        self._async_checkpoint(
                            worker_manager=worker_manager,
                            iteration=start_iteration + completed_iterations,
                            total_infosets=total_infosets,
                            storage_capacity=last_capacity or 0,
                            training_start_time=training_start_time,
                        )

                    if interrupted:
                        break

            except KeyboardInterrupt:
                interrupted = True

            # Save checkpoint on interrupt if we haven't just saved
            if interrupted and self._checkpoint_enabled() and completed_iterations > 0:
                if verbose:
                    print("[Master] Saving checkpoint...", flush=True)
                self._async_checkpoint(
                    worker_manager=worker_manager,
                    iteration=start_iteration + completed_iterations,
                    total_infosets=total_infosets,
                    storage_capacity=last_capacity or 0,
                    training_start_time=training_start_time,
                )
                # Wait for checkpoint to complete before returning
                self._wait_for_checkpoint()

            elapsed_time = time.time() - training_start_time

            # Wait for any pending checkpoint
            self._wait_for_checkpoint()

            # Update solver iteration
            self.solver.iteration = start_iteration + completed_iterations

            # Update run tracker
            storage_capacity = last_capacity or initial_capacity
            if interrupted:
                self.run_tracker.mark_interrupted()
            else:
                self.run_tracker.mark_completed()

            self.run_tracker.update(
                iterations=self.solver.iteration,
                runtime_seconds=elapsed_time,
                num_infosets=total_infosets,
                storage_capacity=storage_capacity,
            )

            # Print final summary
            self._print_final_summary(
                self.solver.iteration,
                total_infosets,
                elapsed_time,
                interrupted,
                fallback_stats,
            )

        return {
            "total_iterations": completed_iterations,
            "final_infosets": total_infosets,
            "avg_utility": self.metrics.get_avg_utility(),
            "elapsed_time": elapsed_time,
            "interrupted": interrupted,
        }

    def train(
        self,
        num_iterations: int | None = None,
        num_workers: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Run parallel training using hash-partitioned infosets.

        Training always uses parallel mode with SharedArrayStorage. For sequential
        behavior, simply use num_workers=1.

        Args:
            num_iterations: Number of iterations (overrides config if provided)
            num_workers: Number of parallel workers (default: CPU count, use 1 for sequential)
            batch_size: Iterations per batch for parallel mode

        Returns:
            Training results dictionary with:
                - total_iterations: Total iterations completed
                - final_infosets: Number of infosets discovered
                - avg_utility: Average utility over training
                - elapsed_time: Total training time in seconds

        Note:
            To resume from a checkpoint, use TrainingSession.resume() instead of creating
            a new session. This ensures proper state restoration from disk.
        """
        # Get iteration count
        if num_iterations is None:
            num_iterations = self.config.training.num_iterations

        # Default to CPU count for num_workers
        if num_workers is None:
            num_workers = mp.cpu_count()

        return self._train_partitioned(num_iterations, num_workers, batch_size)

    def evaluate(
        self,
        num_samples: int = 10000,
        num_rollouts_per_infoset: int = 100,
    ) -> dict[str, Any]:
        """
        Evaluate current solver using exploitability estimation.

        Args:
            num_samples: Number of game samples for exploitability estimation
            num_rollouts_per_infoset: Number of rollouts per infoset for BR approximation

        Returns:
            Evaluation metrics including exploitability estimates
        """

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
