"""The sink's contract is about FAILURE, so that is what these test.

`emit` must not block and must not raise, `opened` must do both, and a drop must
be reported rather than swallowed. Every one of those is a promise the trainer
relies on, and none of them shows up in a happy-path test.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from src.adapters.postgres.sink import PostgresSink, _folded


class _Engine:
    """An engine whose writes can be made to hang or explode on demand."""

    def __init__(
        self,
        *,
        fail: bool = False,
        block: threading.Event | None = None,
        delay: float = 0.0,
    ):
        self.fail = fail
        self.block = block
        self.delay = delay
        self.batches: list[Any] = []

    def begin(self):
        engine = self

        class _Tx:
            def __enter__(self):
                if engine.delay:
                    time.sleep(engine.delay)
                if engine.block is not None:
                    engine.block.wait(timeout=5)
                if engine.fail:
                    raise RuntimeError("the database is unreachable")
                return self

            def __exit__(self, *_exc):
                return False

            def execute(self, statement):
                engine.batches.append(statement)
                return

        return _Tx()


class TestEmitNeverRaisesAndPushesBack:
    """One that raises turns a transport fault into a dead task, so `emit` still
    never raises. What changed with the share's copy going away is that it no
    longer DISCARDS: a full queue means the database is unreachable, and the
    right answer is to slow down rather than to forget.
    """

    def test_a_full_queue_waits_rather_than_discarding(self, monkeypatch):
        """`put_nowait` threw an event away the instant the queue filled, which
        was fine while the same event sat in a file on the share."""
        monkeypatch.setattr("src.adapters.postgres.sink._BACKPRESSURE_SECONDS", 0.2)
        gate = threading.Event()
        sink = PostgresSink(_Engine(block=gate), queue_depth=2)
        started = time.perf_counter()
        for i in range(8):
            sink.emit("run-a", "progress", {"iteration": i})
        waited = time.perf_counter() - started
        gate.set()
        assert waited > 0.2, "emit discarded instead of pushing back"
        sink.close(timeout=2)

    def test_it_still_never_raises_on_the_caller(self, monkeypatch):
        """The ceiling exists so an outage cannot wedge the training loop
        forever -- reaching it is a lost event, not an exception."""
        monkeypatch.setattr("src.adapters.postgres.sink._BACKPRESSURE_SECONDS", 0.05)
        gate = threading.Event()
        sink = PostgresSink(_Engine(block=gate), queue_depth=1)
        for i in range(5):
            sink.emit("run-a", "progress", {"iteration": i})  # must not raise
        gate.set()
        assert sink.flush(1.0) is False, "and it must SAY it dropped"
        sink.close(timeout=2)

    def test_a_failing_write_does_not_reach_the_caller(self):
        sink = PostgresSink(_Engine(fail=True))
        for i in range(5):
            sink.emit("run-a", "progress", {"iteration": i})  # must not raise
        assert sink.close(timeout=2) is False, "a lost batch is reported, not hidden"

    def test_the_worker_outlives_a_bad_batch(self):
        """One failure must not end the thread: the next batch still has to be
        tried, or a blip becomes a permanent outage."""
        engine = _Engine(fail=True)
        sink = PostgresSink(engine)
        sink.emit("run-a", "progress", {"iteration": 1})
        sink.flush(1.0)
        engine.fail = False
        sink.emit("run-a", "progress", {"iteration": 2})
        sink.flush(1.0)
        sink.close(timeout=2)
        assert engine.batches, "the worker stopped after the first failure"


class TestOpenedIsTheOppositePolicy:
    """It carries the resolved config, and a resume reads it to decide it may
    continue. A run that cannot record its own existence must fail here."""

    def test_it_raises_rather_than_dropping(self):
        sink = PostgresSink(_Engine(fail=True))
        with pytest.raises(RuntimeError):
            sink.opened("run-a", {"config_name": "quick_test", "config": {"a": 1}})
        sink.close(timeout=2)

    def test_it_is_not_subject_to_the_queue_limit(self):
        """`emit` drops when full; `opened` waits, because dropping it is the
        unrecoverable case."""
        sink = PostgresSink(_Engine(), queue_depth=1)
        sink.opened("run-a", {"config_name": "quick_test", "config": {}})
        assert sink.flush(2.0) is True
        sink.close(timeout=2)


class TestClaimMayBlock:
    def test_a_failed_claim_reaches_the_caller(self):
        """Losing a claim strands bytes, which is recoverable -- but silently
        succeeding when the row was never written is not."""
        sink = PostgresSink(_Engine(fail=True))
        with pytest.raises(RuntimeError):
            sink.claim("run-a", 1000, "rungs/run-a/1000")
        sink.close(timeout=2)


class TestBatching:
    """Batching ADAPTS, and testing it against an instant writer tests nothing.

    A first version of this asserted <100 round trips for 500 events against a
    fake that returned immediately, and measured 161 -- because when the writer
    keeps up, small batches are the CORRECT behaviour and lower latency. The
    property only exists when the writer is the slow side, which is the case it
    was built for: a real round trip to Azure.
    """

    def test_a_slow_writer_makes_the_batches_grow(self):
        engine = _Engine(delay=0.02)
        sink = PostgresSink(engine)
        for i in range(500):
            sink.emit("run-a", "progress", {"iteration": i})
        sink.flush(10.0)
        sink.close(timeout=5)
        assert len(engine.batches) < 20, (
            f"{len(engine.batches)} round trips for 500 events against a 20ms writer"
        )


class TestTheRunRowIsFoldedNotJustLogged:
    """A status event in `run_events` is not enough: every listing reads the
    folded `runs.status`, and `prune-checkpoints` decides from it whether a
    ladder may be deleted. A finished run whose row still says `running` is the
    zombie this migration exists partly to stop creating -- and the first
    version of this sink produced exactly that."""

    def test_closed_writes_before_it_queues(self):
        engine = _Engine()
        sink = PostgresSink(engine)
        sink.closed("run-a", "completed", {"completed_at": "2026-01-01"})
        assert engine.batches, "the run row was never updated"
        sink.close(timeout=2)

    def test_a_failed_close_reaches_the_caller(self):
        """It takes the blocking path for the same reason `opened` does."""
        sink = PostgresSink(_Engine(fail=True))
        with pytest.raises(RuntimeError):
            sink.closed("run-a", "completed", {})
        sink.close(timeout=2)


class TestTheCountersAreFoldedToo:
    """`runs` is a cache of the fold, and a cache nothing maintains is a stale
    number. A completed 200,000-iteration run read as 0 iterations in 0 seconds
    on a real node while its own events said otherwise."""

    def test_a_progress_batch_moves_the_run_row(self):
        rows = [
            {"run_id": "run-a", "body": {"iterations": 1000, "attempt_runtime_seconds": 10.0}},
            {"run_id": "run-a", "body": {"iterations": 5000, "attempt_runtime_seconds": 50.0}},
        ]
        assert _folded(rows) == {"run-a": {"iterations": 5000, "runtime_seconds": 50.0}}

    def test_it_takes_the_max_so_a_late_batch_cannot_walk_it_back(self):
        """Batches are not ordered against each other; last-wins would let a
        straggler report a finished run as less far along than it is."""
        rows = [
            {"run_id": "run-a", "body": {"iterations": 9000}},
            {"run_id": "run-a", "body": {"iterations": 200}},
        ]
        assert _folded(rows)["run-a"]["iterations"] == 9000

    def test_events_with_no_counter_move_nothing(self):
        assert _folded([{"run_id": "run-a", "body": {"kind": "fresh"}}]) == {}

    def test_two_runs_in_one_batch_stay_apart(self):
        rows = [
            {"run_id": "run-a", "body": {"iterations": 10}},
            {"run_id": "run-b", "body": {"iterations": 20}},
        ]
        assert _folded(rows) == {
            "run-a": {"iterations": 10},
            "run-b": {"iterations": 20},
        }
