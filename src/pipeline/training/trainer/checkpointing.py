"""Checkpoint lifecycle manager for TrainingSession."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from src.pipeline.training.parallel_manager import SharedArrayWorkerManager

if TYPE_CHECKING:
    from src.pipeline.training.trainer.session import TrainingSession


class CheckpointManager:
    """Owns the background checkpoint executor and its back-pressure state.

    Holds a back-reference to the session for config, run tracker, and verbosity;
    deliberately defines no ``__del__`` (the session/manager reference cycle would
    make finalization order fragile) — the session's ``__del__`` calls
    :meth:`shutdown`.
    """

    def __init__(self, session: TrainingSession):
        self.session = session
        self.executor: ThreadPoolExecutor | None = None
        if session.config.storage.checkpoint_enabled:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self.pending: Future[float] | None = None
        # Back-pressure state: iteration of the last submitted checkpoint, and the
        # wall-clock cost / finish time of the last completed one. Used to keep
        # checkpointing from dominating compute on large (slow-to-serialize) runs.
        self.last_iteration: int = 0
        self.last_seconds: float = 0.0
        self.last_end_time: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.session.config.storage.checkpoint_enabled

    @property
    def frequency(self) -> int:
        return self.session.config.training.checkpoint_frequency

    def anchor(self, iteration: int) -> None:
        """Anchor the checkpoint interval at ``iteration`` (so a resume does not
        immediately re-checkpoint the state it just loaded)."""
        self.last_iteration = iteration

    def should_checkpoint(self, iteration: int) -> bool:
        # Interval since the last checkpoint, NOT ``iteration % freq < batch_size`` — the
        # latter fires every batch once batch_size >= freq, which thrashed large runs.
        if not self.enabled or iteration == 0:
            return False
        return iteration - self.last_iteration >= self.frequency

    def async_checkpoint(
        self,
        worker_manager: SharedArrayWorkerManager,
        iteration: int,
        total_infosets: int,
        storage_capacity: int,
        training_start_time: float,
    ) -> None:
        if self.executor is None:
            return
        if self.pending is not None and not self.pending.done():
            if self.session.verbose:
                print("[Master] Previous checkpoint still running; skipping", flush=True)
            return
        # Back-pressure: cap checkpointing at ~`max_checkpoint_overhead` of wall-clock. If
        # the last checkpoint cost T seconds, require (1-f)/f * T seconds of training since
        # it finished before starting another (f=0.1 → wait 9*T → ~10% overhead). Self-adapts
        # to checkpoint cost at any scale.
        if self.last_seconds > 0.0:
            frac = self.session.config.storage.max_checkpoint_overhead
            min_gap = self.last_seconds * (1.0 - frac) / frac
            if time.time() - self.last_end_time < min_gap:
                if self.session.verbose:
                    print("[Master] Deferring checkpoint (back-pressure)", flush=True)
                return

        self._submit(
            worker_manager, iteration, total_infosets, storage_capacity, training_start_time
        )

    def ensure_final_checkpoint(
        self,
        worker_manager: SharedArrayWorkerManager,
        iteration: int,
        total_infosets: int,
        storage_capacity: int,
        training_start_time: float,
    ) -> None:
        """Guarantee the final state is checkpointed (blocking), on normal *and* interrupted
        exit — unless the exact iteration was already checkpointed."""
        if self.executor is None or iteration <= 0:
            return
        self.wait()
        if self.last_iteration >= iteration:
            return
        self._submit(
            worker_manager, iteration, total_infosets, storage_capacity, training_start_time
        )
        self.wait()

    def wait(self) -> None:
        """Block until any in-flight checkpoint finishes; re-raise its failure."""
        if self.pending is None:
            return
        if self.session.verbose:
            print("[Master] Waiting for background checkpoint to complete...", flush=True)
        try:
            self.pending.result()
        except Exception as exc:
            print(f"[Master] ERROR: Background checkpoint failed: {exc}", flush=True)
            raise
        finally:
            self.pending = None

    def shutdown(self) -> None:
        if self.executor is None:
            return
        self.wait()
        self.executor.shutdown(wait=True)

    def _submit(
        self,
        worker_manager: SharedArrayWorkerManager,
        iteration: int,
        total_infosets: int,
        storage_capacity: int,
        training_start_time: float,
    ) -> None:
        assert self.executor is not None
        self.last_iteration = iteration
        self.pending = self.executor.submit(
            self._checkpoint_with_timing,
            worker_manager,
            iteration,
            total_infosets,
            storage_capacity,
            training_start_time,
        )

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
        if self.session.run_tracker is not None:
            self.session.run_tracker.update(
                iterations=iteration,
                runtime_seconds=elapsed_time,
                num_infosets=total_infosets,
                storage_capacity=storage_capacity,
            )

        checkpoint_time = time.time() - start
        self.last_seconds = checkpoint_time
        self.last_end_time = time.time()
        if self.session.verbose:
            print(
                f"[Master] Checkpoint saved at iter={iteration} in {checkpoint_time:.2f}s",
                flush=True,
            )
        return checkpoint_time
