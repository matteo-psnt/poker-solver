"""Unit tests for checkpoint scheduling, back-pressure, and the final checkpoint.

These guard the fix for the thrashing failure where a large run spent most of its
compute serializing checkpoints (checkpoint cost grew with infosets while the
scheduler fired every batch).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from src.pipeline.training.trainer.checkpointing import CheckpointManager

if TYPE_CHECKING:
    from src.pipeline.training.parallel_manager import SharedArrayWorkerManager
    from src.pipeline.training.trainer.session import TrainingSession

# The scheduling helpers never touch the worker manager on the paths under test.
_NO_WM = cast("SharedArrayWorkerManager", None)


class _DoneFuture:
    def done(self) -> bool:
        return True

    def result(self) -> float:
        return 0.0


class _RecordingExecutor:
    def __init__(self) -> None:
        self.submits = 0

    def submit(self, *args, **kwargs) -> _DoneFuture:
        self.submits += 1
        return _DoneFuture()


def _manager(
    *,
    freq=100_000,
    last=0,
    enabled=True,
    ckpt_seconds=0.0,
    ckpt_end=0.0,
    recording=True,
    overhead=0.1,
) -> CheckpointManager:
    session = cast(
        "TrainingSession",
        SimpleNamespace(
            config=SimpleNamespace(
                # Disabled during construction so no real ThreadPoolExecutor is
                # created; the flag is flipped afterwards for the `enabled` path.
                storage=SimpleNamespace(checkpoint_enabled=False, max_checkpoint_overhead=overhead),
                training=SimpleNamespace(checkpoint_frequency=freq),
            ),
            run_tracker=None,
            verbose=False,
        ),
    )
    manager = CheckpointManager(session)
    session.config.storage.checkpoint_enabled = enabled
    if recording:
        manager.executor = cast(ThreadPoolExecutor, _RecordingExecutor())
    manager.last_iteration = last
    manager.last_seconds = ckpt_seconds
    manager.last_end_time = ckpt_end
    return manager


def _submits(manager: CheckpointManager) -> int:
    return cast(_RecordingExecutor, manager.executor).submits


def test_should_checkpoint_is_interval_based():
    manager = _manager(freq=100_000, last=0)
    assert not manager.should_checkpoint(50_000)
    assert manager.should_checkpoint(100_000)


def test_should_checkpoint_disabled_or_iteration_zero():
    assert not _manager(enabled=False, recording=False).should_checkpoint(100_000)
    assert not _manager().should_checkpoint(0)


def test_back_pressure_defers_within_the_overhead_window():
    # 10% overhead → must wait 9x the last checkpoint's cost (900s) before the next.
    manager = _manager(ckpt_seconds=100.0, ckpt_end=time.time() - 200.0, overhead=0.1)
    manager.async_checkpoint(_NO_WM, 200_000, 0, 0, 0.0)
    assert _submits(manager) == 0  # only 200s elapsed, need 900s


def test_back_pressure_allows_after_the_overhead_window():
    manager = _manager(ckpt_seconds=100.0, ckpt_end=time.time() - 1000.0, overhead=0.1)
    manager.async_checkpoint(_NO_WM, 200_000, 0, 0, 0.0)
    assert _submits(manager) == 1  # 1000s > 900s
    assert manager.last_iteration == 200_000


def test_higher_overhead_allowance_checkpoints_sooner():
    # At 50% overhead the wait is only 1x cost, so the same 200s gap is enough.
    manager = _manager(ckpt_seconds=100.0, ckpt_end=time.time() - 200.0, overhead=0.5)
    manager.async_checkpoint(_NO_WM, 200_000, 0, 0, 0.0)
    assert _submits(manager) == 1


def test_first_checkpoint_is_not_back_pressured():
    manager = _manager(ckpt_seconds=0.0)
    manager.async_checkpoint(_NO_WM, 100_000, 0, 0, 0.0)
    assert _submits(manager) == 1


def test_final_checkpoint_dedups_when_already_saved():
    manager = _manager(last=500_000)
    manager.ensure_final_checkpoint(_NO_WM, 500_000, 0, 0, 0.0)
    assert _submits(manager) == 0


def test_final_checkpoint_saves_unsaved_final_state():
    manager = _manager(last=400_000)
    manager.ensure_final_checkpoint(_NO_WM, 500_000, 0, 0, 0.0)
    assert _submits(manager) == 1
    assert manager.last_iteration == 500_000
