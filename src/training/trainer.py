"""
Training orchestration for MCCFR solver.

Manages the complete training loop: solver creation, iteration execution,
checkpointing, metrics tracking, and progress reporting.

Training uses parallel multiprocessing with hash-partitioned shared memory.
"""

import multiprocessing as mp
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.exploitability import compute_exploitability
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.helpers import get_missing_checkpoint_files
from src.training import components
from src.training.checkpoint_manager import CheckpointManager
from src.training.metrics import MetricsTracker
from src.training.metrics_reporter import MetricsReporter
from src.training.orchestrator import TrainingOrchestrator
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

            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(
                config=self.config,
                run_tracker=self.run_tracker,
                verbose=self.verbose,
            )

            # Initialize metrics reporter
            self.metrics_reporter = MetricsReporter(
                metrics=self.metrics,
                verbose=self.verbose,
            )

            # Initialize training orchestrator
            self.orchestrator = TrainingOrchestrator(
                config=self.config,
                checkpoint_manager=self.checkpoint_manager,
                metrics_reporter=self.metrics_reporter,
                verbose=self.verbose,
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
        if hasattr(self, "checkpoint_manager"):
            self.checkpoint_manager.shutdown()

    @property
    def verbose(self) -> bool:
        """Get verbose setting from config."""
        return self.config.training.verbose

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
            "checkpoint_freq": self.config.training.checkpoint_frequency,
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
        if verbose:
            self.metrics_reporter.print_training_header(
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

            # Run training loop via orchestrator
            results = self.orchestrator.run_training(
                worker_manager=worker_manager,
                num_iterations=num_iterations,
                num_workers=num_workers,
                batch_size=batch_size_val,
                start_iteration=start_iteration,
                training_start_time=training_start_time,
            )

            # Wait for any pending checkpoint
            self.checkpoint_manager.wait_for_checkpoint()

            # Update solver iteration
            self.solver.iteration = start_iteration + results["total_iterations"]

            # Update run tracker
            storage_capacity = results.get("last_capacity", initial_capacity)
            if results["interrupted"]:
                self.run_tracker.mark_interrupted()
            else:
                self.run_tracker.mark_completed()

            self.run_tracker.update(
                iterations=self.solver.iteration,
                runtime_seconds=results["elapsed_time"],
                num_infosets=results["final_infosets"],
                storage_capacity=storage_capacity,
            )

            # Print final summary
            self.metrics_reporter.print_final_summary(
                self.solver.iteration,
                results["final_infosets"],
                results["elapsed_time"],
                results["interrupted"],
                results.get("fallback_stats"),
            )

        return {
            "total_iterations": results["total_iterations"],
            "final_infosets": results["final_infosets"],
            "avg_utility": self.metrics.get_avg_utility(),
            "elapsed_time": results["elapsed_time"],
            "interrupted": results["interrupted"],
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
