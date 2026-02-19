"""Partitioned training-loop helpers for TrainingSession."""

from __future__ import annotations

import pickle
import random
import time
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from src.training.parallel_manager import SharedArrayWorkerManager

from . import checkpointing, reporting

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

        try:
            for batch_idx in batch_iterator:
                checkpointing.wait_for_checkpoint(session)

                remaining = num_iterations - completed_iterations
                current_batch_size = min(batch_size_val, remaining)

                iters_per_worker_base = current_batch_size // num_workers
                extra_iters = current_batch_size % num_workers
                iterations_per_worker = [
                    iters_per_worker_base + (1 if i < extra_iters else 0)
                    for i in range(num_workers)
                ]

                batch_result = worker_manager.run_batch(
                    iterations_per_worker=iterations_per_worker,
                    batch_id=batch_idx,
                    start_iteration=start_iteration + completed_iterations,
                    verbose=verbose,
                )

                batch_utilities = batch_result["utilities"]
                completed_iterations += len(batch_utilities)
                total_infosets = batch_result.get("num_infosets", 0)
                max_worker_capacity = batch_result.get("max_worker_capacity", 0.0)
                last_capacity = batch_result.get("capacity", last_capacity)

                if "fallback_stats" in batch_result:
                    fallback_stats = batch_result["fallback_stats"]

                if batch_result.get("interrupted"):
                    interrupted = True

                if batch_idx < num_batches - 1:
                    inter_batch_timeout = max(60.0, batch_result["batch_time"] * 2.0)
                    worker_manager.exchange_ids(
                        timeout=inter_batch_timeout,
                        verbose=verbose,
                    )
                    worker_manager.apply_pending_updates(
                        timeout=inter_batch_timeout,
                        verbose=verbose,
                    )

                for i, util in enumerate(batch_utilities):
                    iter_num = start_iteration + completed_iterations - len(batch_utilities) + i + 1
                    session.metrics.log_iteration(
                        iteration=iter_num,
                        utility=util,
                        num_infosets=total_infosets,
                        infoset_sampler=None,
                    )

                reporting.update_progress_bar(
                    session,
                    batch_iterator,
                    start_iteration + completed_iterations,
                    total_infosets,
                    max_worker_capacity,
                )

                if checkpointing.should_checkpoint(
                    session, start_iteration + completed_iterations, batch_size_val
                ):
                    checkpointing.async_checkpoint(
                        session=session,
                        worker_manager=worker_manager,
                        iteration=start_iteration + completed_iterations,
                        total_infosets=total_infosets,
                        storage_capacity=last_capacity or 0,
                        training_start_time=training_start_time,
                    )

                if interrupted:
                    break

        except KeyboardInterrupt:
            interrupted = True

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
