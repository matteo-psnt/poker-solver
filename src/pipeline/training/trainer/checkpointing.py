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
    if not checkpoint_enabled(session) or iteration == 0:
        return False
    return iteration % checkpoint_frequency(session) < batch_size


def async_checkpoint(
    session: TrainingSession,
    worker_manager: SharedArrayWorkerManager,
    iteration: int,
    total_infosets: int,
    storage_capacity: int,
    training_start_time: float,
) -> None:
    if not checkpoint_enabled(session) or session._checkpoint_executor is None:
        return
    if session._pending_checkpoint is not None and not session._pending_checkpoint.done():
        if session.verbose:
            print("[Master] Previous checkpoint still running; skipping", flush=True)
        return

    session._pending_checkpoint = session._checkpoint_executor.submit(
        checkpoint_with_timing,
        session,
        worker_manager,
        iteration,
        total_infosets,
        storage_capacity,
        training_start_time,
    )


def wait_for_checkpoint(session: TrainingSession) -> None:
    if session._pending_checkpoint is None:
        return
    if session.verbose:
        print("[Master] Waiting for background checkpoint to complete...", flush=True)
    try:
        session._pending_checkpoint.result()
    except Exception as exc:
        print(f"[Master] ERROR: Background checkpoint failed: {exc}", flush=True)
        raise
    finally:
        session._pending_checkpoint = None


def shutdown_checkpoint_executor(session: TrainingSession) -> None:
    if session._checkpoint_executor is None:
        return
    wait_for_checkpoint(session)
    session._checkpoint_executor.shutdown(wait=True)


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
    if session.verbose:
        print(
            f"[Master] Checkpoint saved at iter={iteration} in {checkpoint_time:.2f}s",
            flush=True,
        )
    return checkpoint_time
