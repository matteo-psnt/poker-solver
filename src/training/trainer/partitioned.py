"""Partitioned training-loop helpers for TrainingSession."""

from __future__ import annotations

import pickle
import random
import time
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from src.training.parallel_manager import SharedArrayWorkerManager

from . import checkpointing, reporting
from .batch_coordinator import BatchLoopState, TrainingBatchCoordinator

if TYPE_CHECKING:
    from src.training.trainer.session import TrainingSession


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
        print(
            f"[Resume] Using stored capacity {stored_capacity:,} "
            f"(config initial_capacity={initial_capacity:,})"
        )
    initial_capacity = stored_capacity

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

    serialized_action_abstraction = pickle.dumps(session.solver.action_abstraction)
    serialized_card_abstraction = pickle.dumps(session.solver.card_abstraction)

    if verbose:
        abstraction_size = len(serialized_action_abstraction) + len(serialized_card_abstraction)
        print(f"   Serialized abstractions: {abstraction_size:,} bytes")

    pool_start_time = time.time()

    with SharedArrayWorkerManager(
        num_workers=num_workers,
        config=session.config,
        serialized_action_abstraction=serialized_action_abstraction,
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
            print(f"   Worker pool ready ({pool_init_time:.2f}s)\n")

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
        fallback_stats: dict[str, float] | None = None
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
        fallback_stats = loop_state.fallback_stats
        last_capacity = loop_state.last_capacity

        if interrupted and checkpointing.checkpoint_enabled(session) and completed_iterations > 0:
            if verbose:
                print("[Master] Saving checkpoint...", flush=True)
            checkpointing.async_checkpoint(
                session=session,
                worker_manager=worker_manager,
                iteration=start_iteration + completed_iterations,
                total_infosets=total_infosets,
                storage_capacity=last_capacity or 0,
                training_start_time=training_start_time,
            )
            checkpointing.wait_for_checkpoint(session)

        elapsed_time = time.time() - training_start_time
        checkpointing.wait_for_checkpoint(session)

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
            fallback_stats,
        )

    return {
        "total_iterations": completed_iterations,
        "final_infosets": total_infosets,
        "avg_utility": session.metrics.get_avg_utility(),
        "elapsed_time": elapsed_time,
        "interrupted": interrupted,
    }
