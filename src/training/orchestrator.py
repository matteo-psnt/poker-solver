"""
Training orchestration for MCCFR solver.

Coordinates the core training loop, batch execution, and component coordination.
"""

import time
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from src.training.checkpoint_manager import CheckpointManager
from src.training.metrics_reporter import MetricsReporter
from src.utils.config import Config

if TYPE_CHECKING:
    from src.training.parallel import SharedArrayWorkerManager


class TrainingOrchestrator:
    """
    Orchestrates the core training loop.

    Responsibilities:
    - Execute training batches
    - Coordinate checkpoint/metrics/worker manager
    - Handle interrupts gracefully
    - Track training state (fallback stats, capacity)

    Extracted from TrainingSession to achieve single responsibility.
    """

    def __init__(
        self,
        config: Config,
        checkpoint_manager: CheckpointManager,
        metrics_reporter: MetricsReporter,
        verbose: bool = True,
    ):
        """
        Initialize training orchestrator.

        Args:
            config: Training configuration
            checkpoint_manager: Checkpoint coordination
            metrics_reporter: Metrics and progress reporting
            verbose: Whether to print progress
        """
        self._config = config
        self._checkpoint_mgr = checkpoint_manager
        self._metrics_reporter = metrics_reporter
        self._verbose = verbose

        # State tracking
        self._fallback_stats: dict[str, float] | None = None
        self._last_capacity: int | None = None

    def run_training(
        self,
        worker_manager: "SharedArrayWorkerManager",
        num_iterations: int,
        num_workers: int,
        batch_size: int,
        start_iteration: int,
        training_start_time: float,
    ) -> dict[str, Any]:
        """
        Execute the main training loop.

        Args:
            worker_manager: Worker manager for parallel execution
            num_iterations: Number of iterations to run
            num_workers: Number of parallel workers
            batch_size: Batch size for parallel execution
            start_iteration: Starting iteration number (for resume)
            training_start_time: Training start time for elapsed calculation

        Returns:
            Dict with:
                - total_iterations: int
                - final_infosets: int
                - interrupted: bool
                - elapsed_time: float
                - last_capacity: int | None
                - fallback_stats: dict[str, float] | None
        """
        # Setup
        num_batches = (num_iterations + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            range(num_batches),
            desc="Training batches",
            unit="batch",
            disable=not self._verbose,
        )

        completed_iterations = 0
        total_infosets = 0
        interrupted = False

        try:
            for batch_idx in batch_iterator:
                # Wait for any pending checkpoint
                self._checkpoint_mgr.wait_for_checkpoint()

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
                    start_iteration=start_iteration + completed_iterations,
                    verbose=self._verbose,
                )

                # Update state
                batch_utilities = batch_result["utilities"]
                completed_iterations += len(batch_utilities)
                total_infosets = batch_result.get("num_infosets", 0)
                max_worker_capacity = batch_result.get("max_worker_capacity", 0.0)
                self._last_capacity = batch_result.get("capacity", self._last_capacity)

                if "fallback_stats" in batch_result:
                    self._fallback_stats = batch_result["fallback_stats"]

                if batch_result.get("interrupted"):
                    interrupted = True

                # Exchange IDs and apply updates between batches
                # (owner-to-requester, not global broadcast)
                if batch_idx < num_batches - 1:
                    worker_manager.exchange_ids(verbose=self._verbose)
                    worker_manager.apply_pending_updates(verbose=self._verbose)

                # Log metrics for each iteration in batch
                self._metrics_reporter.log_batch_utilities(
                    batch_utilities,
                    start_iteration + completed_iterations - len(batch_utilities),
                    total_infosets,
                )

                # Update progress bar
                self._metrics_reporter.update_progress_bar(
                    batch_iterator,
                    start_iteration + completed_iterations,
                    total_infosets,
                    max_worker_capacity,
                )

                # Checkpoint if needed
                if self._checkpoint_mgr.should_checkpoint(
                    start_iteration + completed_iterations, batch_size
                ):
                    self._checkpoint_mgr.async_checkpoint(
                        worker_manager=worker_manager,
                        iteration=start_iteration + completed_iterations,
                        total_infosets=total_infosets,
                        storage_capacity=self._last_capacity or 0,
                        training_start_time=training_start_time,
                    )

                if interrupted:
                    break

        except KeyboardInterrupt:
            interrupted = True

        # Save checkpoint on interrupt if we haven't just saved
        if interrupted and self._checkpoint_mgr.checkpoint_enabled and completed_iterations > 0:
            if self._verbose:
                print("[Master] Saving checkpoint...", flush=True)
            self._checkpoint_mgr.async_checkpoint(
                worker_manager=worker_manager,
                iteration=start_iteration + completed_iterations,
                total_infosets=total_infosets,
                storage_capacity=self._last_capacity or 0,
                training_start_time=training_start_time,
            )
            # Wait for checkpoint to complete before returning
            self._checkpoint_mgr.wait_for_checkpoint()

        elapsed_time = time.time() - training_start_time

        return {
            "total_iterations": completed_iterations,
            "final_infosets": total_infosets,
            "interrupted": interrupted,
            "elapsed_time": elapsed_time,
            "last_capacity": self._last_capacity,
            "fallback_stats": self._fallback_stats,
        }
