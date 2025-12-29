"""
Checkpoint management for training runs.

Handles async checkpoint coordination, background execution, and run metadata updates.
"""

import concurrent.futures
import time
from typing import TYPE_CHECKING, Optional

from src.training.run_tracker import RunTracker
from src.utils.config import Config

if TYPE_CHECKING:
    from src.training.parallel import SharedArrayWorkerManager


class CheckpointManager:
    """
    Manages async checkpointing for training runs.

    Responsibilities:
    - Coordinate async checkpoint saves
    - Manage background executor lifecycle
    - Update run metadata after checkpoints
    - Skip checkpoints if previous is still running

    Extracted from TrainingSession to achieve single responsibility.
    """

    def __init__(
        self,
        config: Config,
        run_tracker: RunTracker,
        verbose: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            config: Training configuration
            run_tracker: Run metadata tracker
            verbose: Whether to print checkpoint messages
        """
        self._config = config
        self._run_tracker = run_tracker
        self._verbose = verbose

        # Async executor (single background thread)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint"
        )
        self._pending_checkpoint: Optional[concurrent.futures.Future[float]] = None
        self._checkpoint_failed: Optional[Exception] = None

    @property
    def checkpoint_enabled(self) -> bool:
        """Check if checkpointing is enabled in config."""
        return self._config.storage.checkpoint_enabled

    @property
    def checkpoint_frequency(self) -> int:
        """Get checkpoint frequency from config."""
        return self._config.training.checkpoint_frequency

    def should_checkpoint(self, iteration: int, batch_size: int) -> bool:
        """
        Check if checkpoint should be saved at this iteration.

        Args:
            iteration: Current iteration number
            batch_size: Batch size (used to determine if we crossed a checkpoint boundary)

        Returns:
            True if checkpoint should be saved
        """
        if not self.checkpoint_enabled or iteration == 0:
            return False
        return iteration % self.checkpoint_frequency < batch_size

    def async_checkpoint(
        self,
        worker_manager: "SharedArrayWorkerManager",
        iteration: int,
        total_infosets: int,
        storage_capacity: int,
        training_start_time: float,
    ) -> None:
        """
        Submit async checkpoint save (non-blocking).

        Skips if previous checkpoint is still running to avoid backup.

        Args:
            worker_manager: Worker manager to collect keys from
            iteration: Current iteration number
            total_infosets: Total number of infosets discovered
            storage_capacity: Storage capacity to save (passed to avoid race condition)
            training_start_time: Training start time for elapsed calculation
        """
        if self._pending_checkpoint is not None and not self._pending_checkpoint.done():
            if self._verbose:
                print("[Master] Previous checkpoint still running; skipping", flush=True)
            return

        self._pending_checkpoint = self._executor.submit(
            self._checkpoint_with_timing,
            worker_manager,
            iteration,
            total_infosets,
            storage_capacity,
            training_start_time,
        )

    def wait_for_checkpoint(self) -> None:
        """
        Block until pending checkpoint completes.

        Raises:
            Exception: If checkpoint failed, the exception is re-raised
        """
        if self._pending_checkpoint is not None:
            if self._verbose:
                print(
                    "[Master] Waiting for background checkpoint to complete...",
                    flush=True,
                )
            try:
                elapsed = self._pending_checkpoint.result()
                if self._verbose:
                    print(
                        f"[Master] Background checkpoint completed ({elapsed:.2f}s)",
                        flush=True,
                    )
            except Exception as e:
                # Store exception and print warning
                self._checkpoint_failed = e
                print(f"[Master] ERROR: Background checkpoint failed: {e}", flush=True)
                # Re-raise to fail-fast
                raise
            finally:
                self._pending_checkpoint = None

    def shutdown(self) -> None:
        """Shutdown checkpoint executor (waits for pending operations)."""
        self.wait_for_checkpoint()
        self._executor.shutdown(wait=True)

    def _checkpoint_with_timing(
        self,
        worker_manager: "SharedArrayWorkerManager",
        iteration: int,
        total_infosets: int,
        storage_capacity: int,
        training_start_time: float,
    ) -> float:
        """
        Execute checkpoint with timing (runs in background thread).

        Args:
            worker_manager: Worker manager to checkpoint
            iteration: Current iteration number
            total_infosets: Total number of infosets
            storage_capacity: Storage capacity (passed to avoid race condition)
            training_start_time: Training start time

        Returns:
            Elapsed time for checkpoint operation
        """
        start = time.time()

        # Collect keys from workers and save checkpoint
        worker_manager.checkpoint(iteration)

        # Update run metadata
        elapsed_time = time.time() - training_start_time
        self._run_tracker.update(
            iterations=iteration,
            runtime_seconds=elapsed_time,
            num_infosets=total_infosets,
            storage_capacity=storage_capacity,
        )

        checkpoint_time = time.time() - start
        if self._verbose:
            print(
                f"[Master] Checkpoint saved at iter={iteration} in {checkpoint_time:.2f}s",
                flush=True,
            )
        return checkpoint_time
