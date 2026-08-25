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
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint, save_checkpoint
from src.pipeline.abstraction.resolver import ComboAbstractionResolver
from src.shared import run_events
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.engine.solver.protocols import BucketingStrategy
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


def worker_seed(base_seed: int, worker_id: int, chunk_start: int = 0) -> int:
    """Deterministic, decorrelated seed per (worker, chunk).

    ``chunk_start`` is the chunk's ABSOLUTE start iteration, so a resumed leg
    seeds differently from the leg it continues. Mixed through SeedSequence
    rather than added: pairs sharing a seed would silently draw correlated
    MCCFR samples, which costs effective sample size without showing up.
    """
    return int(np.random.SeedSequence([base_seed, worker_id, chunk_start]).generate_state(1)[0])


def worker_iteration_indices(
    worker_id: int, num_workers: int, num_iterations: int, start: int = 0
) -> range:
    """Global iteration indices for one worker: ``worker_id, +N, +2N, ...``

    INTERLEAVED, not contiguous blocks. ``solver.iteration`` is not a progress
    counter -- it is read as *t* by the DCFR discount ``t^a/(t^a+1)``, by the
    strategy weight ``t^gamma``, and by the traversing-player and button
    alternation. Contiguous blocks would give the first worker only small *t* and
    the last only large *t*, each applying a different slice of the discount
    schedule to the shared arrays. Interleaved, every worker sees the full range and
    the union is exactly ``range(num_iterations)``.

    ``start`` is the absolute iteration this chunk begins at, so a resumed chunk
    keeps numbering from the checkpoint rather than replaying *t* from zero -- which
    would re-apply the barely-discounted early schedule to a trained table.
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
    chunk_start: int = 0,
    counters: Any = None,
    merge_lock: Any = None,
) -> None:
    """Train ``iterations`` on the shared arrays, then report.

    There is no message loop: nothing needs saying between workers. The whole
    body is attach, seed, train, report -- plus a slot in ``counters``, which is
    a write to shared memory once per full tree walk and needs no coordination
    because no other process reads or writes this worker's slot.
    """
    # Spawned: no logging config is inherited, so without this everything
    # _build_local logs falls below lastResort's WARNING floor.
    configure_logging(config.system.log_level)
    storage = None
    try:
        action_model, abstraction, tree = _build_local(config, abstraction)
        storage = StaticArrayStorage(tree, session_id=session_id, attach=True)
        # Reach/utility are tallies nothing reads until after the join, but as
        # shared arrays every traverser row RMWs two more cache lines that all
        # workers ping-pong -- and racing non-atomic increments silently LOSE
        # counts. Tally privately; merge once per chunk under the lock (int64
        # merge is exact, so the counts stop undercounting).
        shared_reach = storage.reach_counts
        shared_utility = storage.cumulative_utility
        storage.reach_counts = np.zeros_like(shared_reach)
        storage.cumulative_utility = np.zeros_like(shared_utility)
        solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)

        # The chunk's ABSOLUTE start iteration, not a per-call counter: a
        # counter restarts at 0 on every leg, so a resumed run would replay
        # leg one's RNG streams and add correlated samples instead of new ones
        # (the 25M-postmortem defect).
        seed = worker_seed(base_seed, worker_id, chunk_start)
        random.seed(seed)
        np.random.seed(seed)

        started = time.time()
        # CUMULATIVE over the task, not the chunk: this worker's slot is written
        # by this worker alone and chunks run in sequence, so carrying the banked
        # value forward makes the sum monotone and leaves the coordinator nothing
        # to reset between chunks -- and so no window where it reads a base from
        # one chunk against counts from another.
        banked = int(counters[worker_id]) if counters is not None else 0
        # ABSOLUTE indices, never a local count: the solver reads each as t.
        count = 0
        for offset in range(0, len(indices), COUNTER_STRIDE):
            stride = indices[offset : offset + COUNTER_STRIDE]
            solver.train_iterations(stride)
            count += len(stride)
            if counters is not None:
                counters[worker_id] = banked + count
        if merge_lock is not None:
            with merge_lock:
                shared_reach += storage.reach_counts
                shared_utility += storage.cumulative_utility
        else:
            shared_reach += storage.reach_counts
            shared_utility += storage.cumulative_utility
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


# Iterations per kernel call, and so per counter write. One crossing costs the
# random-state hand-off (~280 us); at a worker's ~3,000 it/s this keeps the
# counter under half a second stale while amortising that to nothing.
COUNTER_STRIDE = 1000


# How often the workers' counters are totalled and written. Deliberately well
# under the cadence anything READS it at -- the write is a small local JSON,
# and the two intervals otherwise compound into staleness neither names.
HEARTBEAT_SECONDS = 5


class _ProgressReporter:
    """Totals the workers' counters on a timer and publishes the absolute
    iteration.

    A thread rather than a call at the chunk boundary because the coordinator
    spends the whole chunk blocked on the result queue, and the chunk is the
    interval that is too coarse in the first place.
    """

    def __init__(
        self,
        publish: Callable[[int, int], None] | None,
        counters: Any,
        start: int,
        target: int,
    ) -> None:
        self._publish = publish
        self._counters = counters
        self._start = start
        self._target = target
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="train-progress", daemon=True)

    def start(self) -> None:
        if self._publish is not None:
            self._thread.start()

    def stop(self) -> None:
        """Idempotent, and safe on a reporter that was never started -- the
        caller stops it from a `finally` that also covers the paths where
        nothing ran."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=HEARTBEAT_SECONDS)
        self._report()

    def _loop(self) -> None:
        self._report()
        while not self._stop.wait(HEARTBEAT_SECONDS):
            self._report()

    def _report(self) -> None:
        if self._publish is None:
            return
        # Clamped: the counters are read without a lock while workers write
        # them, and a bar drawn past its end reads as a rendering bug.
        done = min(self._start + sum(self._counters), self._target)
        # NEVER FATAL, exactly like the writer it is handed: this describes the
        # training, it is not the training.
        try:
            self._publish(done, self._target)
        except Exception:  # noqa: BLE001
            logger.warning("Could not publish training progress; training continues.")


def _append_checkpoint_event(checkpoint_dir: Path, **fields: Any) -> None:
    """Record the mid-flight row, but never at the cost of the task.

    `records.append_log` propagates on purpose, so its callers can choose. Here
    the choice is clear: this runs immediately AFTER `save_checkpoint` succeeded,
    so a full disk or an IO error on `run.jsonl` would throw away a good rung and
    mark the run failed over telemetry.
    """
    try:
        run_events.append(checkpoint_dir, run_events.CHECKPOINT, **fields)
    except Exception:  # telemetry must not fail a checkpoint already written
        logger.warning("Could not record the checkpoint event; training continues.", exc_info=True)


def train_static_parallel(
    config: Config,
    *,
    num_iterations: int,
    num_workers: int,
    session_id: str,
    checkpoint_dir: Path | None = None,
    base_seed: int = 42,
    checkpoint_retain_every: int = 0,
    abstraction_id: str | None = None,
    abstraction: BucketingStrategy | None = None,
    checkpoint_every: int = 0,
    start_iteration: int = 0,
    resume: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    worker: Callable[..., None] = _worker_entry,
    worker_args: tuple[Any, ...] = (),
    before_checkpoint: Callable[[StaticArrayStorage], None] | None = None,
) -> StaticTrainingResult:
    """Train on static storage across ``num_workers`` processes.

    The coordinator owns the shared arrays and does not train: it allocates,
    fans out, waits, then checkpoints. Iterations are split evenly with the
    remainder going to the first workers, so the total is exact rather than
    approximately right.

    ``worker`` is what each process runs -- the scalar kernel by default, and
    the public-chance-sampling one through the same coordinator, since the
    chunking, the absolute-``t`` assignment, the counters and the checkpoint
    per chunk are properties of the shared table rather than of either kernel.
    ``before_checkpoint`` runs on the coordinator's storage before each write.

    ``on_progress`` is called with (absolute iteration, target) on a timer. It
    has to be a timer and it has to come from the workers: the coordinator is
    blocked on the result queue for a whole chunk at a time, and a chunk is a
    million iterations by default.
    """
    _, abstraction, tree = _build_local(config, abstraction)
    storage = StaticArrayStorage(tree, session_id=session_id)
    # Named before the try so the `finally` can stop it on the paths that return
    # before it exists -- an already-satisfied target is one of them.
    reporter = _ProgressReporter(None, (), 0, num_iterations)
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
            # a retried task past its target is a no-op, not a repeat.
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
        done = start

        # No lock: each slot has exactly one writer, and a sum that is a tick
        # behind is a progress bar, not a result.
        counters = ctx.Array("q", num_workers, lock=False)
        # Serializes only the once-per-chunk tally merge, never a training write.
        merge_lock = ctx.Lock()
        reporter = _ProgressReporter(on_progress, counters, start, num_iterations)
        reporter.start()

        while done < num_iterations:
            chunk_end = min(done + step, num_iterations)
            assignments = [
                worker_iteration_indices(w, num_workers, chunk_end, start=done)
                for w in range(num_workers)
            ]
            result_queue: mp.Queue = ctx.Queue()
            processes = [
                ctx.Process(
                    target=worker,
                    args=(
                        config,
                        worker_id,
                        session_id,
                        assignments[worker_id],
                        base_seed,
                        result_queue,
                        abstraction,
                        done,
                        counters,
                        merge_lock,
                        *worker_args,
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
            done = chunk_end

            # Checkpoint per chunk, so the bound on loss is the chunk, not the run.
            if checkpoint_dir is not None:
                write_started = time.time()
                if before_checkpoint is not None:
                    before_checkpoint(storage)
                save_checkpoint(
                    storage,
                    checkpoint_dir,
                    done,
                    retain_every=checkpoint_retain_every,
                    abstraction_id=abstraction_id,
                )
                checkpoint_seconds = time.time() - write_started
                coverage = storage.coverage()
                logger.info(
                    f"[static] {done:,}/{num_iterations:,} ({coverage:.1%} coverage) checkpointed"
                )
                # The one place a run can record what it looked like mid-flight.
                # After save_checkpoint, so a row never describes state the
                # arrays did not reach.
                task_elapsed = time.time() - started
                _append_checkpoint_event(
                    checkpoint_dir,
                    ts=datetime.now(UTC).isoformat(),
                    iteration=done,
                    # Scoped to the LEG: a resumed task restarts its clock while
                    # `iteration` keeps counting, so an absolute iteration over a
                    # task's elapsed time reports a rate the run never achieved.
                    task_iterations=done - start,
                    task_elapsed_s=round(task_elapsed, 3),
                    iters_per_sec=(
                        round((done - start) / task_elapsed, 1) if task_elapsed > 0 else 0.0
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
        reporter.stop()
        storage.close()


__all__ = (
    "StaticTrainingResult",
    "train_static_parallel",
    "worker_iteration_indices",
    "worker_seed",
)
