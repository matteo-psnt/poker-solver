"""Unit tests for checkpoint scheduling, back-pressure, and the final checkpoint.

These guard the fix for the thrashing failure where a large run spent most of its
compute serializing checkpoints (checkpoint cost grew with infosets while the
scheduler fired every batch).
"""

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from src.pipeline.training.trainer import checkpointing

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


def _session(
    *, freq=100_000, last=0, enabled=True, ckpt_seconds=0.0, ckpt_end=0.0, executor=None
) -> "TrainingSession":
    return cast(
        "TrainingSession",
        SimpleNamespace(
            config=SimpleNamespace(
                storage=SimpleNamespace(checkpoint_enabled=enabled),
                training=SimpleNamespace(checkpoint_frequency=freq),
            ),
            last_checkpoint_iteration=last,
            last_checkpoint_seconds=ckpt_seconds,
            last_checkpoint_end_time=ckpt_end,
            checkpoint_executor=executor,
            pending_checkpoint=None,
            verbose=False,
        ),
    )


def test_should_checkpoint_is_interval_based():
    session = _session(freq=100_000, last=0)
    assert not checkpointing.should_checkpoint(session, 50_000, batch_size=10_000)
    assert checkpointing.should_checkpoint(session, 100_000, batch_size=10_000)


def test_should_checkpoint_disabled_or_iteration_zero():
    assert not checkpointing.should_checkpoint(_session(enabled=False), 100_000, 10_000)
    assert not checkpointing.should_checkpoint(_session(), 0, 10_000)


def test_back_pressure_defers_right_after_an_expensive_checkpoint():
    executor = _RecordingExecutor()
    session = _session(ckpt_seconds=100.0, ckpt_end=time.time(), executor=executor)
    checkpointing.async_checkpoint(session, _NO_WM, 200_000, 0, 0, 0.0)
    assert executor.submits == 0  # a 100s checkpoint just finished → defer


def test_back_pressure_allows_after_enough_training_elapsed():
    executor = _RecordingExecutor()
    session = _session(ckpt_seconds=100.0, ckpt_end=time.time() - 200.0, executor=executor)
    checkpointing.async_checkpoint(session, _NO_WM, 200_000, 0, 0, 0.0)
    assert executor.submits == 1
    assert session.last_checkpoint_iteration == 200_000


def test_first_checkpoint_is_not_back_pressured():
    executor = _RecordingExecutor()
    session = _session(ckpt_seconds=0.0, executor=executor)
    checkpointing.async_checkpoint(session, _NO_WM, 100_000, 0, 0, 0.0)
    assert executor.submits == 1


def test_final_checkpoint_dedups_when_already_saved():
    executor = _RecordingExecutor()
    session = _session(last=500_000, executor=executor)
    checkpointing.ensure_final_checkpoint(session, _NO_WM, 500_000, 0, 0, 0.0)
    assert executor.submits == 0


def test_final_checkpoint_saves_unsaved_final_state():
    executor = _RecordingExecutor()
    session = _session(last=400_000, executor=executor)
    checkpointing.ensure_final_checkpoint(session, _NO_WM, 500_000, 0, 0, 0.0)
    assert executor.submits == 1
    assert session.last_checkpoint_iteration == 500_000
