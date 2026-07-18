"""Batch orchestration for partitioned training sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from src.pipeline.training.metrics import compute_quality_from_arrays
from src.pipeline.training.metrics_history import MetricsHistoryWriter

from . import reporting

if TYPE_CHECKING:
    from src.pipeline.training.parallel_manager import SharedArrayWorkerManager
    from src.pipeline.training.trainer.session import TrainingSession


@dataclass(slots=True)
class BatchLoopState:
    """Mutable state accumulated across training batches."""

    completed_iterations: int = 0
    total_infosets: int = 0
    interrupted: bool = False
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

        # Durable convergence curve: one row per batch under the run dir. Seed the
        # sampler deterministically so metrics.jsonl is reproducible for a fixed
        # config seed; sampling only reads shared arrays, so it never perturbs training.
        self.history_writer = MetricsHistoryWriter(session.run_dir / "metrics.jsonl")
        seed = session.config.system.seed
        self._quality_rng = np.random.default_rng(seed if seed is not None else 0)

    def run_batch(self, batch_idx: int, batch_iterator: tqdm, state: BatchLoopState) -> None:
        """Run a single batch and update shared loop state."""
        self.session.checkpoints.wait()

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

        if batch_idx < self.num_batches - 1:
            inter_batch_timeout = max(60.0, batch_result["batch_time"] * 2.0)
            self.worker_manager.exchange_ids(
                timeout=inter_batch_timeout,
                verbose=self.verbose,
            )

        self._record_batch_metrics(
            batch_utilities,
            state.total_infosets,
            completed_before_batch,
        )
        self._record_history(
            self.start_iteration + state.completed_iterations, state.total_infosets
        )
        reporting.update_progress_bar(
            self.session,
            batch_iterator,
            self.start_iteration + state.completed_iterations,
            state.total_infosets,
            max_worker_capacity,
        )

        if self.session.checkpoints.should_checkpoint(
            self.start_iteration + state.completed_iterations
        ):
            self.session.checkpoints.async_checkpoint(
                worker_manager=self.worker_manager,
                iteration=self.start_iteration + state.completed_iterations,
                total_infosets=state.total_infosets,
                storage_capacity=state.last_capacity or 0,
                training_start_time=self.training_start_time,
            )

    def _sample_quality(self) -> dict[str, float] | None:
        """Sample solver-health metrics from the live shared arrays (coordinator view).

        Reads a random sample of allocated infoset rows straight from shared memory.
        The read races benign against Hogwild worker writes — torn values are fine for
        aggregate statistics — and is bounded to ``sample_size`` rows. Returns None if
        the arrays are not yet populated or on any error (metrics must never crash a run).
        """
        storage = getattr(self.worker_manager, "storage", None)
        if storage is None:
            return None
        try:
            action_counts = storage.shared_action_counts
            regrets = storage.shared_regrets
            allocated = np.flatnonzero(action_counts > 0)
            if allocated.size == 0:
                return None
            sample_size = self.session.metrics.sample_size
            if allocated.size > sample_size:
                sample_ids = self._quality_rng.choice(allocated, sample_size, replace=False)
            else:
                sample_ids = allocated
            return compute_quality_from_arrays(regrets, action_counts, sample_ids)
        except Exception as exc:  # pragma: no cover - defensive; metrics must not kill a run
            print(f"[metrics-history] quality sampling skipped: {exc}", flush=True)
            return None

    def _record_history(self, iteration: int, total_infosets: int) -> None:
        """Append one convergence-curve row (utility, speed, and solver-health) to disk."""
        metrics = self.session.metrics
        quality = self._sample_quality()
        if quality is not None:
            metrics.record_quality(quality)

        row = {
            "iteration": iteration,
            "elapsed_s": round(metrics.get_elapsed_time(), 3),
            "iter_per_sec": round(metrics.get_iterations_per_second(), 2),
            "num_infosets": int(total_infosets),
            "avg_utility": metrics.get_avg_utility(),
            "utility_std": metrics.get_utility_std(),
        }
        if quality is not None:
            row.update(quality)
        self.history_writer.append(row)

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
            )
