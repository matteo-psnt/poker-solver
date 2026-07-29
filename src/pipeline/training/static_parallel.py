"""Multi-process training over static, tree-indexed storage.

The dynamic backend needed ~1,300 lines of worker machinery — per-worker ID
request/response queues, an ownership hash, a frontier re-send with a throttle,
a resize protocol, and job re-targeting so one worker could not consume
another's message. Every line of it existed to answer one question at runtime:
*which row is this infoset?*

Static enumeration answers that from config alone, identically in every
process, before anything starts. So this module has no queues but the job queue,
no ownership, no exchange, no resize, and no notion of a worker "not yet
knowing" a row — which is what made the measured 39-74% dropped-update rate
possible in the first place. A worker here can always write.

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
from pathlib import Path

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline.training.abstraction_resolver import ComboAbstractionResolver
from src.shared.config import Config

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
    iterations: int,
    base_seed: int,
    result_queue: mp.Queue,
    abstraction: BucketingStrategy | None = None,
) -> None:
    """Train ``iterations`` on the shared arrays, then report.

    There is no message loop: nothing needs saying between workers. The whole
    body is attach, seed, train, report.
    """
    storage = None
    try:
        action_model, abstraction, tree = _build_local(config, abstraction)
        storage = StaticArrayStorage(tree, session_id=session_id, attach=True)
        solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)

        seed = worker_seed(base_seed, worker_id)
        random.seed(seed)
        np.random.seed(seed)

        started = time.time()
        for _ in range(iterations):
            solver.train_iteration()
        result_queue.put(
            {
                "worker_id": worker_id,
                "iterations": iterations,
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
        per_worker = [num_iterations // num_workers] * num_workers
        for i in range(num_iterations % num_workers):
            per_worker[i] += 1

        started = time.time()
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_worker_entry,
                args=(
                    config,
                    worker_id,
                    session_id,
                    per_worker[worker_id],
                    base_seed,
                    result_queue,
                    abstraction,
                ),
                daemon=False,
            )
            for worker_id in range(num_workers)
            if per_worker[worker_id] > 0
        ]
        for process in processes:
            process.start()

        results = [result_queue.get() for _ in processes]
        for process in processes:
            process.join()

        failures = [r for r in results if r.get("error")]
        if failures:
            raise RuntimeError(f"{len(failures)} static worker(s) failed: {failures[0]['error']}")

        elapsed = time.time() - started
        dropped = sum(r["dropped"] for r in results)
        if dropped:
            # Structurally impossible here; a nonzero value means something
            # reintroduced dynamic allocation and must not pass silently.
            raise AssertionError(
                f"{dropped} updates were dropped on static storage, which has no "
                "code path for it — dynamic allocation has been reintroduced."
            )

        if checkpoint_dir is not None:
            from src.engine.solver.storage.static_checkpoint import save_checkpoint

            save_checkpoint(
                storage, checkpoint_dir, num_iterations, retain_every=checkpoint_retain_every
            )

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


__all__ = ("StaticTrainingResult", "train_static_parallel", "worker_seed")
