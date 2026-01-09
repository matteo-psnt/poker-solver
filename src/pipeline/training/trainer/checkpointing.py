"""Checkpoint lifecycle helpers for TrainingSession."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.pipeline.training.parallel_manager import SharedArrayWorkerManager

if TYPE_CHECKING:
    from src.pipeline.training.trainer.session import TrainingSession


def checkpoint_enabled(session: TrainingSession) -> bool:
    return session.config.storage.checkpoint_enabled


def checkpoint_frequency(session: TrainingSession) -> int:
    return session.config.training.checkpoint_frequency


def should_checkpoint(session: TrainingSession, iteration: int, batch_size: int) -> bool:
    # Interval since the last checkpoint, NOT ``iteration % freq < batch_size`` — the
    # latter fires every batch once batch_size >= freq, which thrashed large runs.
    if not checkpoint_enabled(session) or iteration == 0:
        return False
    return iteration - session.last_checkpoint_iteration >= checkpoint_frequency(session)


def async_checkpoint(
    session: TrainingSession,
    worker_manager: SharedArrayWorkerManager,
    iteration: int,
    total_infosets: int,
    storage_capacity: int,
    training_start_time: float,
) -> None:
    if not checkpoint_enabled(session) or session.checkpoint_executor is None:
        return
    if session.pending_checkpoint is not None and not session.pending_checkpoint.done():
        if session.verbose:
            print("[Master] Previous checkpoint still running; skipping", flush=True)
        return
    # Back-pressure: cap checkpointing at ~`max_checkpoint_overhead` of wall-clock. If
    # the last checkpoint cost T seconds, require (1-f)/f * T seconds of training since
    # it finished before starting another (f=0.1 → wait 9*T → ~10% overhead). Self-adapts
    # to checkpoint cost at any scale.
    if session.last_checkpoint_seconds > 0.0:
        frac = session.config.storage.max_checkpoint_overhead
        min_gap = session.last_checkpoint_seconds * (1.0 - frac) / frac
        if time.time() - session.last_checkpoint_end_time < min_gap:
            if session.verbose:
                print("[Master] Deferring checkpoint (back-pressure)", flush=True)
            return

    session.last_checkpoint_iteration = iteration
    session.pending_checkpoint = session.checkpoint_executor.submit(
        checkpoint_with_timing,
        session,
        worker_manager,
        iteration,
        total_infosets,
        storage_capacity,
        training_start_time,
    )


def ensure_final_checkpoint(
    session: TrainingSession,
    worker_manager: SharedArrayWorkerManager,
    iteration: int,
    total_infosets: int,
    storage_capacity: int,
    training_start_time: float,
) -> None:
    """Guarantee the final state is checkpointed (blocking), on normal *and* interrupted
    exit — unless the exact iteration was already checkpointed."""
    if not checkpoint_enabled(session) or session.checkpoint_executor is None or iteration <= 0:
        return
    wait_for_checkpoint(session)
    if session.last_checkpoint_iteration >= iteration:
        return
    session.last_checkpoint_iteration = iteration
    session.pending_checkpoint = session.checkpoint_executor.submit(
        checkpoint_with_timing,
        session,
        worker_manager,
        iteration,
        total_infosets,
        storage_capacity,
        training_start_time,
    )
    wait_for_checkpoint(session)


def wait_for_checkpoint(session: TrainingSession) -> None:
    if session.pending_checkpoint is None:
        return
    if session.verbose:
        print("[Master] Waiting for background checkpoint to complete...", flush=True)
    try:
        session.pending_checkpoint.result()
    except Exception as exc:
        print(f"[Master] ERROR: Background checkpoint failed: {exc}", flush=True)
        raise
    finally:
        session.pending_checkpoint = None


def shutdown_checkpoint_executor(session: TrainingSession) -> None:
    if session.checkpoint_executor is None:
        return
    wait_for_checkpoint(session)
    session.checkpoint_executor.shutdown(wait=True)


def checkpoint_with_timing(
    session: TrainingSession,
    worker_manager: SharedArrayWorkerManager,
    iteration: int,
    total_infosets: int,
    storage_capacity: int,
    training_start_time: float,
) -> float:
    start = time.time()
    worker_manager.checkpoint(iteration)

    elapsed_time = time.time() - training_start_time
    if session.run_tracker is not None:
        session.run_tracker.update(
            iterations=iteration,
            runtime_seconds=elapsed_time,
            num_infosets=total_infosets,
            storage_capacity=storage_capacity,
        )

    checkpoint_time = time.time() - start
    session.last_checkpoint_seconds = checkpoint_time
    session.last_checkpoint_end_time = time.time()
    if session.verbose:
        print(
            f"[Master] Checkpoint saved at iter={iteration} in {checkpoint_time:.2f}s",
            flush=True,
        )
    return checkpoint_time
