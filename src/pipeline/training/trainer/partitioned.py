"""Partitioned training-loop helpers for TrainingSession."""

from __future__ import annotations

import logging
import pickle
import random
import signal
import time
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from src.pipeline.training.parallel_manager import SharedArrayWorkerManager

from . import reporting
from .batch_coordinator import BatchLoopState, TrainingBatchCoordinator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.pipeline.training.trainer.session import TrainingSession


def get_training_config(
    session: TrainingSession, num_workers: int, batch_size: int | None
) -> dict[str, Any]:
    """Parse training configuration with defaults."""
    assert session.run_tracker is not None
    if batch_size is None:
        batch_size = session.config.training.iterations_per_worker * num_workers

    initial_capacity = session.config.storage.initial_capacity
    stored_capacity = session.run_tracker.metadata.resolve_initial_capacity(initial_capacity)
    if stored_capacity != initial_capacity:
        logger.info(
            f"[Resume] Using stored capacity {stored_capacity:,} "
            f"(config initial_capacity={initial_capacity:,})"
        )
    initial_capacity = stored_capacity

    if session.capacity_override is not None:
        if session.capacity_override < initial_capacity:
            raise ValueError(
                f"capacity override {session.capacity_override:,} is smaller than the "
                f"checkpoint's capacity {initial_capacity:,}"
            )
        logger.info(f"[Resume] Pre-allocating capacity override: {session.capacity_override:,}")
        initial_capacity = session.capacity_override

    return {
        "batch_size": batch_size,
        "verbose": session.config.training.verbose,
        "initial_capacity": initial_capacity,
        "max_actions": session.config.storage.max_actions,
        "checkpoint_enabled": session.config.storage.checkpoint_enabled,
    }


def train_partitioned(
    session: TrainingSession,
    num_iterations: int,
    num_workers: int,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run parallel training using shared array storage."""
    assert session.run_tracker is not None

    # Route SIGTERM (Modal app stop / preemption grace, systemd, etc.) into the
    # KeyboardInterrupt path so the run attempts a final checkpoint instead of
    # dying with unsaved progress.
    def _sigterm_to_interrupt(_signum: int, _frame: object) -> None:
        logger.info("[Master] SIGTERM received — attempting final checkpoint...")
        raise KeyboardInterrupt

    try:
        prev_sigterm = signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
    except ValueError:
        prev_sigterm = None  # not the main thread (tests) — skip installation

    try:
        return _train_partitioned(session, num_iterations, num_workers, batch_size)
    finally:
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)


def _train_partitioned(
    session: TrainingSession,
    num_iterations: int,
    num_workers: int,
    batch_size: int | None = None,
) -> dict[str, Any]:
    assert session.run_tracker is not None
    config = get_training_config(session, num_workers, batch_size)
    batch_size_val = config["batch_size"]
    verbose = config["verbose"]
    initial_capacity = config["initial_capacity"]
    max_actions = config["max_actions"]
    checkpoint_enabled = config["checkpoint_enabled"]

    session.run_tracker.initialize()

    reporting.print_training_header(
        session, num_workers, num_iterations, batch_size_val, initial_capacity, max_actions
    )

    training_start_time = time.time()
    start_iteration = session.solver.iteration

    serialized_action_model = pickle.dumps(session.solver.action_model)
    serialized_card_abstraction = pickle.dumps(session.solver.card_abstraction)

    if verbose:
        abstraction_size = len(serialized_action_model) + len(serialized_card_abstraction)
        logger.info(f"   Serialized abstractions: {abstraction_size:,} bytes")

    # Hand shared memory over to the worker manager before it creates its own
    # coordinator storage. Both use session_id=run_dir.name, and creating the
    # second one calls cleanup_stale_shm(), which unlinks the segments this
    # bootstrap storage is still mapping. Left implicit, that orphans a full
    # capacity-sized allocation (GBs at production capacity) for the whole run --
    # nothing ever frees it -- on top of the checkpoint it loaded on resume.
    # Releasing here makes the ownership handoff explicit: from this point the
    # worker manager's storage is the only live view of the arrays.
    session.release_bootstrap_storage()

    pool_start_time = time.time()

    with SharedArrayWorkerManager(
        num_workers=num_workers,
        config=session.config,
        serialized_action_model=serialized_action_model,
        serialized_card_abstraction=serialized_card_abstraction,
        session_id=session.run_dir.name,
        base_seed=session.config.system.seed
        if session.config.system.seed is not None
        else random.randint(0, 2**31 - 1),
        initial_capacity=initial_capacity,
        max_actions=max_actions,
        checkpoint_dir=str(session.run_dir) if checkpoint_enabled else None,
    ) as worker_manager:
        pool_init_time = time.time() - pool_start_time

        if verbose:
            logger.info(f"   Worker pool ready ({pool_init_time:.2f}s)\n")

        num_batches = (num_iterations + batch_size_val - 1) // batch_size_val
        batch_iterator = tqdm(
            range(num_batches),
            desc="Training batches",
            unit="batch",
            disable=not verbose,
        )

        completed_iterations = 0
        total_infosets = 0
        interrupted = False
        last_capacity: int | None = None
        loop_state = BatchLoopState()
        batch_coordinator = TrainingBatchCoordinator(
            session,
            worker_manager,
            num_workers=num_workers,
            num_iterations=num_iterations,
            batch_size=batch_size_val,
            num_batches=num_batches,
            start_iteration=start_iteration,
            training_start_time=training_start_time,
            verbose=verbose,
        )

        session.checkpoints.anchor(start_iteration)

        try:
            for batch_idx in batch_iterator:
                batch_coordinator.run_batch(batch_idx, batch_iterator, loop_state)
                if loop_state.interrupted:
                    break

        except KeyboardInterrupt:
            loop_state.interrupted = True

        completed_iterations = loop_state.completed_iterations
        total_infosets = loop_state.total_infosets
        interrupted = loop_state.interrupted
        last_capacity = loop_state.last_capacity

        elapsed_time = time.time() - training_start_time
        # Let any in-flight checkpoint finish, then guarantee the final state is saved
        # (on normal completion, not just interrupts) — deduped if already checkpointed.
        session.checkpoints.wait()
        if completed_iterations > 0:
            if verbose:
                logger.info("[Master] Saving final checkpoint...")
            session.checkpoints.ensure_final_checkpoint(
                worker_manager=worker_manager,
                iteration=start_iteration + completed_iterations,
                total_infosets=total_infosets,
                storage_capacity=last_capacity or 0,
                training_start_time=training_start_time,
            )

        session.solver.iteration = start_iteration + completed_iterations

        storage_capacity = last_capacity or initial_capacity
        if interrupted:
            session.run_tracker.mark_interrupted()
        else:
            session.run_tracker.mark_completed()

        session.run_tracker.update(
            iterations=session.solver.iteration,
            runtime_seconds=elapsed_time,
            num_infosets=total_infosets,
            storage_capacity=storage_capacity,
        )

        reporting.print_final_summary(
            session,
            session.solver.iteration,
            total_infosets,
            elapsed_time,
            interrupted,
        )

    return {
        "total_iterations": completed_iterations,
        "final_infosets": total_infosets,
        "avg_utility": session.metrics.get_avg_utility(),
        "elapsed_time": elapsed_time,
        "interrupted": interrupted,
    }
