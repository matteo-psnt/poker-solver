"""Batch orchestration for partitioned training sessions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from tqdm import tqdm

from src.pipeline.training.metrics import (
    compute_quality_from_arrays,
    mean_policy_l1_delta,
    regret_matched_policies,
)
from src.pipeline.training.metrics_history import MetricsHistoryWriter

from . import reporting

logger = logging.getLogger(__name__)

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
    # Infoset count after the previous batch, for the new_infosets growth delta.
    # -1 sentinel => not yet set (first batch, incl. the first post-resume batch,
    # reports a null delta rather than a spurious jump off the loaded count).
    prev_infosets: int = -1


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

        # Fixed probe set for the policy-convergence delta: a constant sample of
        # infoset ids whose current policy is diffed batch-to-batch. Fixed once,
        # after a warmup, so the set spans more than the first-discovered
        # (early-street) infosets. Biased-but-consistent is all a delta needs.
        self._probe_ids: np.ndarray | None = None
        self._prev_probe_policies: dict[int, np.ndarray] | None = None
        self._probe_size = 1000
        self._probe_warmup_infosets = 10_000

    def run_batch(self, batch_idx: int, batch_iterator: tqdm, state: BatchLoopState) -> None:
        """Run a single batch and update shared loop state."""
        self.session.checkpoints.wait()

        remaining = self.num_iterations - state.completed_iterations
        current_batch_size = min(self.batch_size, remaining)
        iterations_per_worker = self._get_iterations_per_worker(current_batch_size)

        # Seed coordinate must be the batch's ABSOLUTE iteration index, not the
        # 0-based loop counter: ``batch_idx`` restarts at 0 on resume while
        # ``start_iteration`` carries the offset, so a resumed leg would otherwise
        # replay the deal stream from the beginning -- re-traversing already-trained
        # deals and double-counting them in the average strategy. Absolute iteration
        # is also batch-size invariant, so legs with different batch sizes cannot
        # collide. Kept distinct from ``batch_idx``, which still indexes metrics rows.
        batch_start_iteration = self.start_iteration + state.completed_iterations

        batch_result = self.worker_manager.run_batch(
            iterations_per_worker=iterations_per_worker,
            batch_id=batch_start_iteration,
            start_iteration=batch_start_iteration,
            verbose=self.verbose,
            auto_resize=False,
        )

        batch_utilities = batch_result["utilities"]
        completed_before_batch = state.completed_iterations
        state.completed_iterations += len(batch_utilities)
        state.total_infosets = batch_result.get("num_infosets", 0)
        max_worker_capacity = batch_result.get("max_worker_capacity", 0.0)
        state.last_capacity = batch_result.get("capacity", state.last_capacity)
        state.interrupted = bool(batch_result.get("interrupted"))

        if max_worker_capacity >= self.worker_manager.storage.CAPACITY_THRESHOLD:
            # A resize crash (e.g. /dev/shm exhaustion while old and new arrays
            # coexist) must never cost more than the current batch: block and
            # persist a checkpoint before attempting it.
            self.session.checkpoints.ensure_final_checkpoint(
                worker_manager=self.worker_manager,
                iteration=self.start_iteration + state.completed_iterations,
                total_infosets=state.total_infosets,
                storage_capacity=state.last_capacity or 0,
                training_start_time=self.training_start_time,
            )
            self.worker_manager.check_and_resize_if_needed(
                max_worker_capacity=max_worker_capacity,
                verbose=self.verbose,
            )
            state.last_capacity = self.worker_manager.capacity

        # Skip the exchange barrier only when EVERY worker explicitly reported
        # no ID-sync state in any direction (missing data => exchange).
        worker_results = batch_result.get("worker_results") or []
        all_sync_idle = bool(worker_results) and all(
            r.get("sync_idle") is True for r in worker_results
        )
        # Sync-health telemetry: meaningful only on batches where the exchange
        # actually ran. Left None on skipped/idle batches so a stale value is never
        # carried forward (which would read as sync cost that did not happen).
        exchange_s: float | None = None
        unresolved_frontier: int | None = None
        if batch_idx < self.num_batches - 1 and not all_sync_idle:
            inter_batch_timeout = max(60.0, batch_result["batch_time"] * 2.0)
            sync_t0 = time.perf_counter()
            exchange = self.worker_manager.exchange_ids(
                timeout=inter_batch_timeout,
                verbose=self.verbose,
            )
            exchange_s = round(time.perf_counter() - sync_t0, 3)
            acks = cast("list[dict[str, int]]", exchange.get("acks", []))
            unresolved_frontier = sum(a.get("requested_unresolved", 0) for a in acks)

        applied_updates = sum(
            cast("dict[str, int]", w).get("applied_updates", 0) for w in worker_results
        )
        # First batch (incl. first post-resume) has no prior count => null delta.
        new_infosets = (
            None if state.prev_infosets < 0 else max(0, state.total_infosets - state.prev_infosets)
        )
        state.prev_infosets = state.total_infosets

        self._record_batch_metrics(
            batch_utilities,
            state.total_infosets,
            completed_before_batch,
        )
        self._record_history(
            self.start_iteration + state.completed_iterations,
            state.total_infosets,
            dropped_unknown_id_updates=int(batch_result.get("dropped_unknown_id_updates", 0)),
            applied_updates=applied_updates,
            exchange_s=exchange_s,
            unresolved_frontier=unresolved_frontier,
            new_infosets=new_infosets,
            capacity_pct=round(100.0 * float(max_worker_capacity), 1),
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
            sample_size = self.session.metrics.sample_size
            capacity = action_counts.shape[0]

            # Rejection-sample allocated rows instead of materializing every
            # allocated index: np.flatnonzero over a 24-48M array plus a
            # multi-million index allocation per batch is pure telemetry
            # overhead. Uniform over allocated rows either way.
            sample_ids = None
            for oversample in (8, 64):
                probes = self._quality_rng.integers(0, capacity, size=sample_size * oversample)
                hits = np.unique(probes[action_counts[probes] > 0])
                if hits.size >= sample_size:
                    sample_ids = self._quality_rng.choice(hits, sample_size, replace=False)
                    break
            if sample_ids is None:
                # Sparse storage (early training): the full scan is cheap here.
                allocated = np.flatnonzero(action_counts > 0)
                if allocated.size == 0:
                    return None
                if allocated.size > sample_size:
                    sample_ids = self._quality_rng.choice(allocated, sample_size, replace=False)
                else:
                    sample_ids = allocated
            return compute_quality_from_arrays(regrets, action_counts, sample_ids)
        except Exception as exc:  # pragma: no cover - defensive; metrics must not kill a run
            logger.warning(f"[metrics-history] quality sampling skipped: {exc}")
            return None

    def _policy_delta(self, total_infosets: int) -> float | None:
        """Mean L1 change of the fixed probe set's current policy since last batch.

        ~0 => the policy has stopped moving (converged/plateaued); larger => still
        learning. None until the probe set is fixed (post-warmup) and one prior
        snapshot exists. Reads shared arrays only — never perturbs training.
        """
        storage = getattr(self.worker_manager, "storage", None)
        if storage is None or total_infosets < self._probe_warmup_infosets:
            return None
        try:
            action_counts = storage.shared_action_counts
            regrets = storage.shared_regrets
            if self._probe_ids is None:
                capacity = action_counts.shape[0]
                probes = self._quality_rng.integers(0, capacity, size=self._probe_size * 64)
                hits = np.unique(probes[action_counts[probes] > 0])
                if hits.size < self._probe_size:
                    return None
                self._probe_ids = self._quality_rng.choice(hits, self._probe_size, replace=False)
            current = regret_matched_policies(regrets, action_counts, self._probe_ids)
            delta = (
                None
                if self._prev_probe_policies is None
                else mean_policy_l1_delta(self._prev_probe_policies, current)
            )
            self._prev_probe_policies = current
            return delta
        except Exception as exc:  # pragma: no cover - defensive; metrics must not kill a run
            logger.warning(f"[metrics-history] policy delta skipped: {exc}")
            return None

    def _record_history(
        self,
        iteration: int,
        total_infosets: int,
        *,
        dropped_unknown_id_updates: int = 0,
        applied_updates: int = 0,
        exchange_s: float | None = None,
        unresolved_frontier: int | None = None,
        new_infosets: int | None = None,
        capacity_pct: float | None = None,
    ) -> None:
        """Append one convergence-curve row (utility, speed, and solver-health) to disk."""
        metrics = self.session.metrics
        quality = self._sample_quality()
        if quality is not None:
            metrics.record_quality(quality)

        total_updates = dropped_unknown_id_updates + applied_updates
        row = {
            "iteration": iteration,
            "elapsed_s": round(metrics.get_elapsed_time(), 3),
            "iter_per_sec": round(metrics.get_iterations_per_second(), 2),
            "num_infosets": int(total_infosets),
            # Tree-growth signal: new infosets allocated this batch. A tree that
            # keeps allocating rather than plateauing is spending iterations on
            # discovery, not refinement (see the 25M long-run postmortem).
            "new_infosets": new_infosets,
            "capacity_pct": capacity_pct,
            # Cross-worker updates skipped for not-yet-propagated infoset IDs this
            # batch, plus the applied count and the resulting per-visit drop RATE
            # — the sample-efficiency signal for hash-sharded multi-worker runs.
            "dropped_unknown_id_updates": int(dropped_unknown_id_updates),
            "applied_updates": int(applied_updates),
            "drop_rate": (
                round(dropped_unknown_id_updates / total_updates, 4) if total_updates else None
            ),
            # ID-exchange wall time and the still-unresolved cross-worker frontier
            # size; None on batches where the exchange barrier was skipped.
            "exchange_s": exchange_s,
            "unresolved_frontier": unresolved_frontier,
            # Mean L1 change of the fixed probe set's current policy since the
            # previous batch: ~0 = converged/plateaued, larger = still learning.
            "policy_delta_l1": self._policy_delta(total_infosets),
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
