"""
Training orchestration for MCCFR solver.

Manages the complete training loop: solver creation, iteration execution,
checkpointing, metrics tracking, and progress reporting.

Training uses parallel multiprocessing with hash-partitioned shared memory.
"""

import concurrent.futures
import json
import multiprocessing as mp
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tqdm import tqdm

from src.solver.storage.helpers import CHECKPOINT_REQUIRED_FILES, get_missing_checkpoint_files
from src.solver.storage.shared_array import SharedArrayStorage
from src.training import components
from src.training.metrics import MetricsTracker
from src.training.parallel import SharedArrayWorkerManager
from src.training.run_tracker import RunTracker
from src.utils.config import Config


class TrainingSession:
    """
    Orchestrates MCCFR training.

    Manages solver initialization, training loop, checkpointing,
    metrics tracking, and progress reporting.

    Training uses parallel multiprocessing with hash-partitioned shared memory.
    """

    def __init__(self, config: Config, run_id: Optional[str] = None):
        """
        Initialize trainer from configuration.

        Args:
            config: Configuration object
            run_id: Optional run ID (for resuming or explicit naming)
        """
        self.config = config
        self._fallback_stats: Optional[Dict[str, float]] = None

        # Determine run directory
        runs_base_dir = Path(self.config.training.runs_dir)
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run-{timestamp}"

        self.run_dir = runs_base_dir / run_id

        # Initialize run tracker
        self.run_tracker = RunTracker(
            run_dir=self.run_dir,
            config_name=self.config.system.config_name,
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
            self.metrics = MetricsTracker(window_size=self.config.training.log_frequency)

            # Initialize async checkpointing (single background thread)
            self._checkpoint_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="checkpoint"
            )
            self._pending_checkpoint: Optional[concurrent.futures.Future[float]] = None
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
        missing_files = get_missing_checkpoint_files(run_path)
        if missing_files:
            raise ValueError(
                f"Checkpoint is incomplete. Missing files: {missing_files}\n"
                f"Required: {list(CHECKPOINT_REQUIRED_FILES)}"
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

        Looks for checkpoint artifacts and returns the most recent checkpoint iteration.
        Uses the iteration count from metadata (only advanced on checkpoint).

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

    def _async_checkpoint(self, iteration: int, training_start_time: float):
        """
        Non-blocking checkpoint save.

        Ensures previous checkpoint completes before starting new one.
        Submits new checkpoint to background thread.

        Args:
            iteration: Current iteration number
            training_start_time: Training start time for calculating elapsed time
        """
        # Wait for previous checkpoint to finish (if any)
        if self._pending_checkpoint is not None:
            try:
                # Non-blocking check
                self._pending_checkpoint.result(timeout=0.1)
                if self.verbose:
                    print("[Master] Previous checkpoint completed", flush=True)
            except concurrent.futures.TimeoutError:
                # Still running, will wait before next checkpoint
                pass

        # Submit new checkpoint in background
        self._pending_checkpoint = self._checkpoint_executor.submit(
            self._checkpoint_with_timing, iteration, training_start_time
        )
        if self.verbose:
            print("[Master] Checkpoint started in background (non-blocking)...", flush=True)

    def _checkpoint_with_timing(self, iteration: int, training_start_time: float):
        """
        Execute checkpoint with timing (runs in background thread).

        Args:
            iteration: Current iteration number
            training_start_time: Training start time for calculating elapsed time

        Returns:
            Elapsed time for checkpoint operation
        """
        start = time.time()
        elapsed_time = time.time() - training_start_time
        self.run_tracker.update(
            iterations=iteration,
            runtime_seconds=elapsed_time,
            num_infosets=self.solver.num_infosets(),
        )
        self.storage.checkpoint(iteration)
        checkpoint_time = time.time() - start
        if self.verbose:
            print(f"[Background] Checkpoint saved in {checkpoint_time:.2f}s", flush=True)
        return checkpoint_time

    def _wait_for_checkpoint(self):
        """Wait for pending checkpoint to complete (called at shutdown)."""
        if self._pending_checkpoint is not None:
            if self.verbose:
                print("[Master] Waiting for background checkpoint to complete...", flush=True)
            try:
                elapsed = self._pending_checkpoint.result()  # Block until done
                if self.verbose:
                    print(f"[Master] Background checkpoint completed ({elapsed:.2f}s)", flush=True)
            except Exception as e:
                print(f"[Master] Warning: Background checkpoint failed: {e}", flush=True)
            finally:
                self._pending_checkpoint = None

    def _shutdown_checkpoint_executor(self):
        """Shutdown the checkpoint executor (called at end of training)."""
        if hasattr(self, "_checkpoint_executor"):
            self._checkpoint_executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup on deletion."""
        self._shutdown_checkpoint_executor()

    @property
    def verbose(self) -> bool:
        """Get verbose setting from config."""
        return self.config.training.verbose

    def _get_training_config(self, num_workers: int, batch_size: Optional[int]) -> Dict[str, Any]:
        """Parse training configuration with defaults."""
        if batch_size is None:
            batch_size = self.config.training.iterations_per_worker * num_workers

        # Use max_infosets from checkpoint if it's larger (handles resume after resize)
        max_infosets = self.config.storage.max_infosets
        if self.run_dir and self.run_dir.exists():
            checkpoint_info = SharedArrayStorage.get_checkpoint_info(self.run_dir)
            if checkpoint_info and checkpoint_info.get("max_infosets"):
                checkpoint_max = checkpoint_info["max_infosets"]
                if checkpoint_max > max_infosets:
                    print(
                        f"[Resume] Using checkpoint max_infosets={checkpoint_max:,} "
                        f"(config had {max_infosets:,})"
                    )
                    max_infosets = checkpoint_max

        return {
            "batch_size": batch_size,
            "checkpoint_freq": self.config.training.checkpoint_frequency,
            "verbose": self.config.training.verbose,
            "max_infosets": max_infosets,
            "max_actions": self.config.storage.max_actions,
            "checkpoint_enabled": self.config.storage.checkpoint_enabled,
        }

    def _print_training_header(
        self,
        num_workers: int,
        num_iterations: int,
        batch_size: int,
        max_infosets: int,
        max_actions: int,
    ) -> None:
        """Display training configuration header."""
        print("\n🚀 Shared Array Parallel Training")
        print(f"   Workers: {num_workers}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max infosets: {max_infosets:,}")
        print(f"   Max actions: {max_actions}")
        print("   Mode: Live shared memory arrays")

    def _save_checkpoint(
        self,
        worker_manager,
        completed_iterations: int,
        total_infosets: int,
        training_start_time: float,
    ) -> None:
        """Save a checkpoint and update run metadata."""
        if not self.config.storage.checkpoint_enabled or completed_iterations == 0:
            return

        checkpoint_start = time.time()
        worker_manager.checkpoint(completed_iterations)
        elapsed = time.time() - training_start_time
        self.run_tracker.update(
            iterations=completed_iterations,
            runtime_seconds=elapsed,
            num_infosets=total_infosets,
        )
        checkpoint_time = time.time() - checkpoint_start
        print(
            f"[Coordinator] Checkpoint saved at iteration {completed_iterations} "
            f"in {checkpoint_time:.2f}s"
        )

    def _run_training_loop(
        self,
        worker_manager,
        num_iterations: int,
        num_workers: int,
        batch_size: int,
        checkpoint_freq: int,
        checkpoint_enabled: bool,
        verbose: bool,
        training_start_time: float,
    ) -> tuple[int, int, bool]:
        """
        Execute the main training loop with batching and checkpointing.

        Returns:
            Tuple of (completed_iterations, total_infosets, interrupted)
        """
        completed_iterations = 0
        total_infosets = 0
        interrupted = False

        # Progress tracking
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
                iterations_per_worker = [
                    iters_per_worker_base + (1 if i < extra_iters else 0)
                    for i in range(num_workers)
                ]

                # Run batch (data flows directly through shared arrays)
                batch_result = worker_manager.run_batch(
                    iterations_per_worker=iterations_per_worker,
                    batch_id=batch_idx,
                    start_iteration=completed_iterations,
                    verbose=verbose,
                )

                batch_utilities = batch_result["utilities"]
                completed_iterations += len(batch_utilities)
                total_infosets = batch_result.get("num_infosets", 0)
                max_worker_capacity = batch_result.get("max_worker_capacity", 0.0)
                if "fallback_stats" in batch_result:
                    self._fallback_stats = batch_result["fallback_stats"]
                if batch_result.get("interrupted"):
                    interrupted = True

                # Exchange IDs and apply updates between batches
                # (owner-to-requester, not global broadcast)
                if batch_idx < num_batches - 1:
                    worker_manager.exchange_ids(verbose=verbose)
                    worker_manager.apply_pending_updates(verbose=verbose)

                for i, util in enumerate(batch_utilities):
                    iter_num = completed_iterations - len(batch_utilities) + i + 1
                    self.metrics.log_iteration(
                        iteration=iter_num,
                        utility=util,
                        num_infosets=total_infosets,
                        infoset_sampler=None,
                    )

                # Update progress bar
                if verbose and isinstance(batch_iterator, tqdm):
                    compact_summary = self.metrics.get_compact_summary()
                    batch_iterator.set_postfix_str(
                        f"iter={completed_iterations} infosets={total_infosets} "
                        f"cap={max_worker_capacity:.0%} | {compact_summary}"
                    )

                # Checkpoint if needed
                if checkpoint_enabled and completed_iterations % checkpoint_freq < batch_size:
                    self._save_checkpoint(
                        worker_manager=worker_manager,
                        completed_iterations=completed_iterations,
                        total_infosets=total_infosets,
                        training_start_time=training_start_time,
                    )

                if interrupted:
                    break

        except KeyboardInterrupt:
            interrupted = True
            if verbose:
                print("\n⚠️  Training interrupted by user", flush=True)

        # Save checkpoint on interrupt if we haven't just saved
        if interrupted and checkpoint_enabled and completed_iterations > 0:
            if verbose:
                print("[Coordinator] Saving checkpoint...", flush=True)
            self._save_checkpoint(
                worker_manager=worker_manager,
                completed_iterations=completed_iterations,
                total_infosets=total_infosets,
                training_start_time=training_start_time,
            )

        return completed_iterations, total_infosets, interrupted

    def _finalize_training(
        self,
        completed_iterations: int,
        total_infosets: int,
        elapsed_time: float,
        verbose: bool,
        interrupted: bool,
        checkpoint_enabled: bool,
    ) -> None:
        """Update trackers and print final training summary."""
        if interrupted:
            self.run_tracker.mark_interrupted()
            if verbose:
                print("\n🟡 Training interrupted")
        else:
            if not checkpoint_enabled:
                self.run_tracker.update(
                    iterations=completed_iterations,
                    runtime_seconds=elapsed_time,
                    num_infosets=total_infosets,
                )
            self.run_tracker.mark_completed()
            if verbose:
                print("\n✅ Shared Array Training complete!")

        if verbose:
            print(f"   Iterations: {completed_iterations}")
            print(f"   Infosets: {total_infosets:,}")
            print(f"   Time: {elapsed_time:.1f}s")
            if completed_iterations > 0:
                print(f"   Speed: {completed_iterations / elapsed_time:.2f} iter/s")
        if self._fallback_stats:
            total_lookups = int(self._fallback_stats.get("total_lookups", 0))
            fallback_count = int(self._fallback_stats.get("fallback_count", 0))
            if total_lookups > 0:
                fallback_rate = self._fallback_stats.get("fallback_rate", 0.0) * 100
                print(
                    f"   Abstraction fallbacks: {fallback_count:,}/{total_lookups:,} "
                    f"({fallback_rate:.2f}%)"
                )

    def _train_partitioned(
        self,
        num_iterations: int,
        num_workers: int,
        batch_size: Optional[int] = None,
    ) -> Dict:
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
        # Parse configuration
        config = self._get_training_config(num_workers, batch_size)
        batch_size_val = config["batch_size"]
        checkpoint_freq = config["checkpoint_freq"]
        verbose = config["verbose"]
        max_infosets = config["max_infosets"]
        max_actions = config["max_actions"]
        checkpoint_enabled = config["checkpoint_enabled"]

        # Initialize run tracker
        self.run_tracker.initialize()

        # Display header
        if verbose:
            self._print_training_header(
                num_workers, num_iterations, batch_size_val, max_infosets, max_actions
            )

        training_start_time = time.time()
        completed_iterations = 0
        total_infosets = 0

        # Serialize abstractions
        serialized_action_abstraction = pickle.dumps(self.solver.action_abstraction)
        serialized_card_abstraction = pickle.dumps(self.solver.card_abstraction)

        if verbose:
            abstraction_size = len(serialized_action_abstraction) + len(serialized_card_abstraction)
            print(f"   Serialized abstractions: {abstraction_size:,} bytes")
            print("   Starting worker pool...")

        pool_start_time = time.time()

        # Create shared array worker manager
        with SharedArrayWorkerManager(
            num_workers=num_workers,
            config=self.config,
            serialized_action_abstraction=serialized_action_abstraction,
            serialized_card_abstraction=serialized_card_abstraction,
            session_id=self.run_dir.name,
            base_seed=self.config.system.seed or 42,
            max_infosets=max_infosets,
            max_actions=max_actions,
            checkpoint_dir=str(self.run_dir) if checkpoint_enabled else None,
        ) as worker_manager:
            pool_init_time = time.time() - pool_start_time

            if verbose:
                print(f"   Worker pool ready ({pool_init_time:.2f}s)\n")

            # Run training loop
            interrupted = False
            try:
                completed_iterations, total_infosets, interrupted = self._run_training_loop(
                    worker_manager=worker_manager,
                    num_iterations=num_iterations,
                    num_workers=num_workers,
                    batch_size=batch_size_val,
                    checkpoint_freq=checkpoint_freq,
                    checkpoint_enabled=checkpoint_enabled,
                    verbose=verbose,
                    training_start_time=training_start_time,
                )
            finally:
                elapsed_time = time.time() - training_start_time
                self._finalize_training(
                    completed_iterations,
                    total_infosets,
                    elapsed_time,
                    verbose,
                    interrupted,
                    checkpoint_enabled,
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
        num_iterations: Optional[int] = None,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
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
