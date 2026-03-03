"""A resumed leg must not replay the fresh run's deal stream.

Regression test for the resume-seeding bug: the per-batch RNG coordinate was the
0-based loop counter ``batch_idx``, which restarts at 0 on resume while
``start_iteration`` carries the offset. A resumed leg therefore re-derived
``_compute_seed(base_seed, worker_id, 0..n)`` -- the exact streams the fresh leg
had already traversed -- so it re-sampled already-trained deals and double-counted
them in the average strategy.

The invariant: the seed coordinates a leg emits must be a function of the batch's
absolute iteration position, so legs starting at different iterations produce
disjoint deal streams.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from tqdm import tqdm

from src.pipeline.training.parallel_manager import SharedArrayWorkerManager
from src.pipeline.training.parallel_worker import _compute_seed
from src.pipeline.training.trainer.batch_coordinator import (
    BatchLoopState,
    TrainingBatchCoordinator,
)
from src.pipeline.training.trainer.session import TrainingSession

BATCH_SIZE = 100_000
NUM_BATCHES = 4
NUM_WORKERS = 8


class _RecordingWorkerManager:
    """Captures the batch_id the coordinator emits; does no real work."""

    def __init__(self) -> None:
        self.batch_ids: list[int] = []
        self.storage = SimpleNamespace(CAPACITY_THRESHOLD=0.9)
        self.capacity = 1_000_000

    def run_batch(self, *, iterations_per_worker, batch_id, start_iteration, **_kwargs):
        self.batch_ids.append(batch_id)
        return {
            "utilities": [0.0] * sum(iterations_per_worker),
            "num_infosets": 1,
            "max_worker_capacity": 0.0,
            "capacity": self.capacity,
            "batch_time": 0.01,
            "interrupted": False,
        }

    def exchange_ids(self, **_kwargs) -> dict:
        # Match the real return shape: sync-health telemetry reads acks off it.
        return {"acks": []}


def _emitted_batch_ids(start_iteration: int, tmp_path, monkeypatch) -> list[int]:
    """Run the batch loop with stubbed workers and return the seed coordinates."""
    worker_manager = _RecordingWorkerManager()
    session = SimpleNamespace(
        run_dir=tmp_path,
        config=SimpleNamespace(system=SimpleNamespace(seed=42)),
        checkpoints=SimpleNamespace(
            wait=lambda: None,
            should_checkpoint=lambda _iteration: False,
        ),
    )

    coordinator = TrainingBatchCoordinator(
        session=cast(TrainingSession, session),
        worker_manager=cast(SharedArrayWorkerManager, worker_manager),
        num_workers=NUM_WORKERS,
        num_iterations=BATCH_SIZE * NUM_BATCHES,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        start_iteration=start_iteration,
        training_start_time=0.0,
        verbose=False,
    )
    # Metrics/progress reporting is incidental to the seeding invariant.
    monkeypatch.setattr(coordinator, "_record_batch_metrics", lambda *_a, **_k: None)
    monkeypatch.setattr(coordinator, "_record_history", lambda *_a, **_k: None)

    state = BatchLoopState()
    for batch_idx in range(NUM_BATCHES):
        coordinator.run_batch(batch_idx, cast(tqdm, _NullBar()), state)
    return worker_manager.batch_ids


class _NullBar:
    """Stand-in for the tqdm iterator the coordinator updates."""

    def set_postfix_str(self, *_a, **_k) -> None:
        return None

    def update(self, *_a, **_k) -> None:
        return None


@pytest.fixture(autouse=True)
def _silence_progress(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.training.trainer.batch_coordinator.reporting.update_progress_bar",
        lambda *_a, **_k: None,
    )


def test_batch_id_is_absolute_iteration(tmp_path, monkeypatch):
    """A fresh leg emits the absolute iteration of each batch, not 0,1,2,..."""
    assert _emitted_batch_ids(0, tmp_path, monkeypatch) == [0, 100_000, 200_000, 300_000]


def test_resumed_leg_does_not_replay_fresh_stream(tmp_path, monkeypatch):
    """The bug: a leg resumed at 10M re-emitted the fresh leg's coordinates."""
    fresh = _emitted_batch_ids(0, tmp_path, monkeypatch)
    resumed = _emitted_batch_ids(10_000_000, tmp_path, monkeypatch)

    assert not set(fresh) & set(resumed), (
        f"resumed leg replays fresh batch coordinates: {sorted(set(fresh) & set(resumed))}"
    )
    assert resumed == [10_000_000, 10_100_000, 10_200_000, 10_300_000]


def test_resumed_leg_draws_distinct_rng_streams(tmp_path, monkeypatch):
    """The coordinates must actually yield different seeds for every worker."""
    fresh = _emitted_batch_ids(0, tmp_path, monkeypatch)
    resumed = _emitted_batch_ids(10_000_000, tmp_path, monkeypatch)

    fresh_seeds = {
        _compute_seed(42, worker_id, batch_id)
        for batch_id in fresh
        for worker_id in range(NUM_WORKERS)
    }
    resumed_seeds = {
        _compute_seed(42, worker_id, batch_id)
        for batch_id in resumed
        for worker_id in range(NUM_WORKERS)
    }

    assert len(fresh_seeds) == NUM_BATCHES * NUM_WORKERS, "seeds collide within a leg"
    assert not fresh_seeds & resumed_seeds, "resumed leg reuses fresh-run RNG streams"


def test_batch_size_change_across_legs_does_not_collide(tmp_path, monkeypatch):
    """Absolute-iteration coordinates are batch-size invariant.

    A leg that resumes with a different batch size (e.g. fewer workers) must still
    not land on a previous leg's coordinates except where batches genuinely align.
    """
    fresh = _emitted_batch_ids(0, tmp_path, monkeypatch)
    resumed = _emitted_batch_ids(BATCH_SIZE * NUM_BATCHES, tmp_path, monkeypatch)
    assert not set(fresh) & set(resumed)
