"""Batch orchestration for partitioned training sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm

from . import checkpointing, reporting

if TYPE_CHECKING:
    from src.pipeline.training.parallel_manager import SharedArrayWorkerManager
    from src.pipeline.training.trainer.session import TrainingSession


@dataclass(slots=True)
class BatchLoopState:
    """Mutable state accumulated across training batches."""

    completed_iterations: int = 0
    total_infosets: int = 0
    interrupted: bool = False
    fallback_stats: dict[str, float] | None = None
    last_capacity: int | None = None


class TrainingBatchCoordinator:
    """Coordinates one training batch: run, sync, metrics, and checkpoint decisions."""

    def __init__(
        self,
        session: TrainingSession,
        worker_manager: SharedArrayWorkerManager,
        *,
        num_workers: int,
        num_iterations: int,
        batch_size: int,
        num_batches: int,
        start_iteration: int,
        training_start_time: float,
        verbose: bool,
    ):
        self.session = session
        self.worker_manager = worker_manager
        self.num_workers = num_workers
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.start_iteration = start_iteration
        self.training_start_time = training_start_time
        self.verbose = verbose

    def run_batch(self, batch_idx: int, batch_iterator: tqdm, state: BatchLoopState) -> None:
        """Run a single batch and update shared loop state."""
        checkpointing.wait_for_checkpoint(self.session)

        remaining = self.num_iterations - state.completed_iterations
        current_batch_size = min(self.batch_size, remaining)
        iterations_per_worker = self._get_iterations_per_worker(current_batch_size)

        batch_result = self.worker_manager.run_batch(
            iterations_per_worker=iterations_per_worker,
            batch_id=batch_idx,
            start_iteration=self.start_iteration + state.completed_iterations,
            verbose=self.verbose,
        )

        batch_utilities = batch_result["utilities"]
        completed_before_batch = state.completed_iterations
        state.completed_iterations += len(batch_utilities)
        state.total_infosets = batch_result.get("num_infosets", 0)
        max_worker_capacity = batch_result.get("max_worker_capacity", 0.0)
        state.last_capacity = batch_result.get("capacity", state.last_capacity)
        state.interrupted = bool(batch_result.get("interrupted"))

        if "fallback_stats" in batch_result:
            state.fallback_stats = batch_result["fallback_stats"]

        if batch_idx < self.num_batches - 1:
            inter_batch_timeout = max(60.0, batch_result["batch_time"] * 2.0)
            self.worker_manager.exchange_ids(
                timeout=inter_batch_timeout,
                verbose=self.verbose,
            )
            self.worker_manager.apply_pending_updates(
                timeout=inter_batch_timeout,
                verbose=self.verbose,
            )

        self._record_batch_metrics(
            batch_utilities,
            state.total_infosets,
            completed_before_batch,
        )
        reporting.update_progress_bar(
            self.session,
            batch_iterator,
            self.start_iteration + state.completed_iterations,
            state.total_infosets,
            max_worker_capacity,
        )

        if checkpointing.should_checkpoint(
            self.session,
            self.start_iteration + state.completed_iterations,
            self.batch_size,
        ):
            checkpointing.async_checkpoint(
                session=self.session,
                worker_manager=self.worker_manager,
                iteration=self.start_iteration + state.completed_iterations,
                total_infosets=state.total_infosets,
                storage_capacity=state.last_capacity or 0,
                training_start_time=self.training_start_time,
            )

    def _get_iterations_per_worker(self, current_batch_size: int) -> list[int]:
        base = current_batch_size // self.num_workers
        extra = current_batch_size % self.num_workers
        return [base + (1 if i < extra else 0) for i in range(self.num_workers)]

    def _record_batch_metrics(
        self,
        batch_utilities: list[float],
        total_infosets: int,
        completed_before_batch: int,
    ) -> None:
        for i, util in enumerate(batch_utilities):
            iter_num = self.start_iteration + completed_before_batch + i + 1
            self.session.metrics.log_iteration(
                iteration=iter_num,
                utility=util,
                num_infosets=total_infosets,
                infoset_sampler=None,
            )
