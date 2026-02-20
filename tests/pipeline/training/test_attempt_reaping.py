"""A dead attempt must not stay "running" forever.

An attempt killed mid-flight (guillotine, OOM, SIGKILL) never runs any ``mark_*``,
so its status stayed ``running`` and its ``ended_at`` null indefinitely. Run
``c2ef8c`` accumulated 15 such attempts -- four hours of wall clock that committed
nothing -- and none was distinguishable from a live one, so the run read as still
in flight and the failures were invisible to any reconstruction based on
``metrics.jsonl`` (a dead attempt writes no metrics rows).

Resuming proves the previous attempt is gone, which is the moment to reap it.
"""

import pytest

from src.pipeline.training.run_tracker import RunTracker
from src.shared.config import Config


@pytest.fixture
def tracker(tmp_path) -> RunTracker:
    return RunTracker(
        run_dir=tmp_path / "run-x",
        config_name="test",
        config=Config.default(),
        action_config_hash="hash",
    )


def test_resuming_reaps_the_dead_attempt(tracker):
    metadata = tracker.metadata
    tracker.update(iterations=10_000_000, runtime_seconds=100.0, num_infosets=5, storage_capacity=9)

    tracker.mark_resumed()

    previous = metadata.attempts[0]
    assert previous.status == "died"
    assert previous.ended_at is not None


def test_attempt_that_never_checkpointed_records_committing_nothing(tracker):
    """The 13-consecutive-failures case: died during checkpoint load."""
    metadata = tracker.metadata
    tracker.update(iterations=10_000_000, runtime_seconds=1.0, num_infosets=5, storage_capacity=9)
    tracker.mark_resumed()  # attempt 1 opens at 10_000_000
    tracker.mark_resumed()  # attempt 1 died before any checkpoint; attempt 2 opens

    dead = metadata.attempts[1]
    assert dead.status == "died"
    assert dead.start_iter == 10_000_000
    assert dead.end_iter == 10_000_000, "an attempt that committed nothing must say so"


def test_reaping_does_not_rewrite_a_properly_closed_attempt(tracker):
    metadata = tracker.metadata
    tracker.update(iterations=500, runtime_seconds=10.0, num_infosets=5, storage_capacity=9)
    tracker.mark_interrupted()

    tracker.mark_resumed()

    assert metadata.attempts[0].status == "interrupted"


def test_live_attempt_is_still_reported_running(tracker):
    """Guards against reaping becoming unconditional."""
    metadata = tracker.metadata
    assert metadata.attempts[-1].status == "running"
