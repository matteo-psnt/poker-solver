"""Multi-process training over static, tree-indexed storage.

Static enumeration answers "which row is this infoset?" from config alone,
identically in every process. So this module has no queues but the job queue:
no ownership, no id exchange, no resize. Answering it at runtime instead
previously took ~1,300 lines and still dropped a measured 39-74% of update
samples. A worker here can always write.

What crosses a process boundary: the config, a seed, and an iteration count.
Notably NOT the tree or the abstraction. Each worker rebuilds the tree from
config (a pure function, ~1s) and attaches to the shared arrays by name, so no
index information is ever shipped or reconciled. Shipping the abstraction would
mean pickling ~773 MB per worker.

Concurrency is Hogwild, unchanged: workers write shared memory lock-free and
races are tolerated. That is the same convergence argument as before; only the
addressing changed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint, save_checkpoint
from src.pipeline.training.abstraction_resolver import ComboAbstractionResolver
from src.shared import run_events
from src.shared.config import Config
from src.shared.log import configure_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaticTrainingResult:
    iterations: int
    num_rows: int
    touched_rows: int
    coverage: float
    mean_visits_per_touched: float
    elapsed_s: float
    iterations_per_second: float
    dropped_updates: int


def worker_seed(base_seed: int, worker_id: int, batch_id: int = 0) -> int:
    """Deterministic, decorrelated seed per (worker, batch).

    Mixed through SeedSequence rather than added: pairs sharing a seed would
    silently draw correlated MCCFR samples, which costs effective sample size
    without showing up anywhere.
    """
    return int(np.random.SeedSequence([base_seed, worker_id, batch_id]).generate_state(1)[0])


def worker_iteration_indices(
    worker_id: int, num_workers: int, num_iterations: int, start: int = 0
) -> range:
    """Global iteration indices for one worker: ``worker_id, +N, +2N, ...``

    INTERLEAVED, not contiguous blocks, and the difference is not cosmetic.
    ``solver.iteration`` is not a progress counter — it is read as *t* by the
    DCFR discount ``t^a/(t^a+1)``, by the strategy weight ``t^gamma``, and by the
    traversing-player and button alternation. Contiguous blocks would give the
    first worker only small *t* and the last only large *t*, so each would apply
    a different slice of the discount schedule to the shared arrays.

    Interleaving gives every worker the full range of *t*, and the union across
    workers is exactly ``range(num_iterations)`` — the same absolute numbering a
    single-worker run would see.

    ``start`` is the absolute iteration this chunk begins at, so a resumed or
    mid-run chunk keeps numbering from where the checkpoint left off rather than
    replaying *t* from zero -- which would re-apply the early, barely-discounted
    part of the DCFR schedule to an already-trained table.
    """
    return range(start + worker_id, num_iterations, num_workers)


def _build_local(config: Config, abstraction: BucketingStrategy | None = None):
    """Rebuild the tree (and abstraction) inside a worker, from config alone.

    ``abstraction`` is an injection point for tests; production passes None so
    each worker resolves it from disk rather than receiving ~773 MB by pickle.
    The resolution is pinned by config, so every process loads the same one.
    """
    action_model = ActionModel(config)
    if abstraction is None:
        abstraction = ComboAbstractionResolver().load(
            abstraction_config=config.card_abstraction.config
        )
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, abstraction, starting_stack=config.game.starting_stack
    )
    return action_model, abstraction, tree


def _worker_entry(
    config: Config,
    worker_id: int,
    session_id: str,
    indices: range,
    base_seed: int,
    result_queue: mp.Queue,
    abstraction: BucketingStrategy | None = None,
    chunk_id: int = 0,
) -> None:
    """Train ``iterations`` on the shared arrays, then report.

    There is no message loop: nothing needs saying between workers. The whole
    body is attach, seed, train, report.
    """
    # Spawned: no logging config is inherited, so without this everything
    # _build_local logs falls below lastResort's WARNING floor.
    configure_logging(config.system.log_level)
    storage = None
    try:
        action_model, abstraction, tree = _build_local(config, abstraction)
        storage = StaticArrayStorage(tree, session_id=session_id, attach=True)
        solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)

        # chunk_id mixed in: without it every chunk re-seeds identically and
        # replays the same RNG stream, so extra chunks would add correlated
        # samples instead of new ones.
        seed = worker_seed(base_seed, worker_id, chunk_id)
        random.seed(seed)
        np.random.seed(seed)

        started = time.time()
        # Assign the ABSOLUTE iteration before each step: train_iteration reads
        # self.iteration as t and then increments, so leaving it to count locally
        # would give this worker t = 0..share instead of its true global indices.
        count = 0
        for global_iteration in indices:
            solver.iteration = global_iteration
            solver.train_iteration()
            count += 1
        result_queue.put(
            {
                "worker_id": worker_id,
                "iterations": count,
                "elapsed_s": time.time() - started,
                "dropped": solver.dropped_unknown_id_updates,
                "error": None,
            }
        )
    except Exception as exc:  # surfaced by the coordinator, never swallowed
        logger.exception(f"[static worker {worker_id}] failed")
        result_queue.put({"worker_id": worker_id, "error": repr(exc)})
    finally:
        if storage is not None:
            storage.close()


def train_static_parallel(
    config: Config,
    *,
    num_iterations: int,
    num_workers: int,
    session_id: str,
    checkpoint_dir: Path | None = None,
    base_seed: int = 42,
    checkpoint_retain_every: int = 0,
    abstraction: BucketingStrategy | None = None,
    checkpoint_every: int = 0,
    start_iteration: int = 0,
    resume: bool = False,
) -> StaticTrainingResult:
    """Train on static storage across ``num_workers`` processes.

    The coordinator owns the shared arrays and does not train: it allocates,
    fans out, waits, then checkpoints. Iterations are split evenly with the
    remainder going to the first workers, so the total is exact rather than
    approximately right.
    """
    _, abstraction, tree = _build_local(config, abstraction)
    storage = StaticArrayStorage(tree, session_id=session_id)
    logger.info(
        f"[static] {len(tree):,} nodes, {tree.num_rows:,} rows, "
        f"{storage.nbytes() / 1e6:.0f} MB shared across {num_workers} workers"
    )

    try:
        # Resume BEFORE any training: the arrays must hold the checkpoint's state
        # before a worker touches them, and `start` decides the absolute `t` the
        # DCFR schedule continues from.
        start = start_iteration
        if resume and checkpoint_dir is not None and start == 0:
            try:
                start = load_checkpoint(storage, checkpoint_dir)
                logger.info(f"[static] resumed from iteration {start:,}")
            except FileNotFoundError:
                start = 0  # nothing banked yet; a fresh run
        if start >= num_iterations:
            # Absolute target already met, so this call is a no-op:
            # a retried leg past its target is a no-op, not a repeat.
            logger.info(
                f"[static] already at {start:,} >= target {num_iterations:,}; nothing to do"
            )
            touched = storage.num_touched_infosets()
            return StaticTrainingResult(
                iterations=start,
                num_rows=tree.num_rows,
                touched_rows=touched,
                coverage=storage.coverage(),
                mean_visits_per_touched=storage.mean_visits_per_touched_infoset(),
                elapsed_s=0.0,
                iterations_per_second=0.0,
                dropped_updates=0,
            )

        # Chunking is what makes a LONG run survivable: a process that dies loses
        # at most one chunk instead of everything since the start. 0 means one
        # chunk, i.e. the historical checkpoint-only-at-the-end behaviour.
        step = checkpoint_every if checkpoint_every > 0 else (num_iterations - start)
        started = time.time()
        ctx = mp.get_context("spawn")
        dropped = 0
        chunk_id = 0
        done = start

        while done < num_iterations:
            chunk_end = min(done + step, num_iterations)
            assignments = [
                worker_iteration_indices(w, num_workers, chunk_end, start=done)
                for w in range(num_workers)
            ]
            result_queue: mp.Queue = ctx.Queue()
            processes = [
                ctx.Process(
                    target=_worker_entry,
                    args=(
                        config,
                        worker_id,
                        session_id,
                        assignments[worker_id],
                        base_seed,
                        result_queue,
                        abstraction,
                        chunk_id,
                    ),
                    daemon=False,
                )
                for worker_id in range(num_workers)
                if len(assignments[worker_id]) > 0
            ]
            for process in processes:
                process.start()

            results = [result_queue.get() for _ in processes]
            for process in processes:
                process.join()

            failures = [r for r in results if r.get("error")]
            if failures:
                raise RuntimeError(
                    f"{len(failures)} static worker(s) failed: {failures[0]['error']}"
                )

            completed = sum(r["iterations"] for r in results)
            if completed != chunk_end - done:
                raise AssertionError(
                    f"workers ran {completed} iterations against a chunk of "
                    f"{chunk_end - done}; the interleaved assignment does not tile it."
                )
            chunk_dropped = sum(r["dropped"] for r in results)
            if chunk_dropped:
                # Structurally impossible here; a nonzero value means something
                # reintroduced dynamic allocation and must not pass silently.
                raise AssertionError(
                    f"{chunk_dropped} updates were dropped on static storage, which has no "
                    "code path for it — dynamic allocation has been reintroduced."
                )
            dropped += chunk_dropped
            done = chunk_end
            chunk_id += 1

            # Checkpoint per chunk, so the bound on loss is the chunk, not the run.
            if checkpoint_dir is not None:
                write_started = time.time()
                save_checkpoint(storage, checkpoint_dir, done, retain_every=checkpoint_retain_every)
                checkpoint_seconds = time.time() - write_started
                coverage = storage.coverage()
                logger.info(
                    f"[static] {done:,}/{num_iterations:,} ({coverage:.1%} coverage) checkpointed"
                )
                # The one place a run can record what it looked like mid-flight.
                # After save_checkpoint, so a row never describes state the
                # arrays did not reach.
                leg_elapsed = time.time() - started
                run_events.append(
                    checkpoint_dir,
                    run_events.CHECKPOINT,
                    ts=datetime.now(UTC).isoformat(),
                    iteration=done,
                    # Scoped to the LEG: a resumed leg restarts its clock while
                    # `iteration` keeps counting, so an absolute iteration over a
                    # leg's elapsed time reports a rate the run never achieved.
                    leg_iterations=done - start,
                    leg_elapsed_s=round(leg_elapsed, 3),
                    iters_per_sec=(
                        round((done - start) / leg_elapsed, 1) if leg_elapsed > 0 else 0.0
                    ),
                    touched_rows=storage.num_touched_infosets(),
                    num_rows=tree.num_rows,
                    coverage=round(coverage, 6),
                    mean_visits_per_touched=round(storage.mean_visits_per_touched_infoset(), 3),
                    dropped_updates=dropped,
                    checkpoint_seconds=round(checkpoint_seconds, 3),
                )

        elapsed = time.time() - started
        num_iterations = done

        touched = storage.num_touched_infosets()
        return StaticTrainingResult(
            iterations=num_iterations,
            num_rows=tree.num_rows,
            touched_rows=touched,
            coverage=storage.coverage(),
            mean_visits_per_touched=storage.mean_visits_per_touched_infoset(),
            elapsed_s=elapsed,
            iterations_per_second=num_iterations / elapsed if elapsed else 0.0,
            dropped_updates=dropped,
        )
    finally:
        storage.close()


__all__ = (
    "StaticTrainingResult",
    "train_static_parallel",
    "worker_iteration_indices",
    "worker_seed",
)
