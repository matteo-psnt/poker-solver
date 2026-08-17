"""The sink is drained before the process that owns it exits.

`emit` queues and a background thread writes, and that thread is a DAEMON -- so
whatever is still queued at exit dies with it. Nothing called `flush`, and the
events lost were the last ones: measured on three node runs, every one reached
the database without its `checkpoint` or its `status`.

A missing terminal status is a finished run that goes on advertising itself as
training, which is the zombie this whole migration exists to stop creating.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from src.adapters.postgres import connect


class _Sink:
    def __init__(self, *, drained: bool = True) -> None:
        self.flushed: list[float] = []
        self._drained = drained

    def flush(self, timeout: float) -> bool:
        self.flushed.append(timeout)
        return self._drained


def _using(monkeypatch, sink: Any) -> None:
    monkeypatch.setattr(connect, "sink_from_environment", lambda: sink)


def test_leaving_the_block_drains(monkeypatch):
    sink = _Sink()
    _using(monkeypatch, sink)
    with connect.record_sink():
        pass
    assert sink.flushed == [connect.FLUSH_TIMEOUT_SECONDS]


def test_it_drains_even_when_the_run_raises(monkeypatch):
    """A run that died still wrote a terminal status, and that is the event
    whose loss is worst."""
    sink = _Sink()
    _using(monkeypatch, sink)
    try:
        with connect.record_sink():
            raise RuntimeError("training died")
    except RuntimeError:
        pass
    assert sink.flushed


def test_a_lossy_drain_is_logged_not_raised(monkeypatch, caplog):
    """The training has already succeeded and the share has the whole record.
    The database being behind is something to say, not something to fail on."""
    _using(monkeypatch, _Sink(drained=False))
    with connect.record_sink():
        pass
    assert "behind the share" in caplog.text


def test_no_dsn_is_no_sink_and_no_drain(monkeypatch):
    monkeypatch.setattr(connect, "sink_from_environment", lambda: None)
    with connect.record_sink() as sink:
        assert sink is None


def test_every_trainer_goes_through_it():
    """Three composition roots build a sink, and one that forgets to drain is a
    run whose last events are silently lost."""
    from pathlib import Path

    from src.shared import repo

    for name in ("train_static", "train_pcs", "train_vector"):
        source = Path(repo.SRC / "interfaces" / "commands" / f"{name}.py").read_text()
        assert "connect.record_sink()" in source, f"{name} builds a sink it never drains"
        assert "connect.sink_from_environment()" not in source


class TestTheFlushCeilingIsReal:
    """`Queue.join()` takes no timeout. The version this replaces passed its
    `timeout` to a log message and nowhere else, so the ceiling every caller
    believed in did not exist -- and a sink that could not reach the database
    would have hung the run at exit, after the work had succeeded.
    """

    def test_a_sink_that_never_drains_still_returns(self):
        from src.adapters.postgres.sink import PostgresSink

        class _Stalled:
            def begin(self):
                raise AssertionError("the writer must not be reached in this test")

        sink = PostgresSink.__new__(PostgresSink)
        sink._queue = _neverending()
        sink._dropped = 0
        sink._lock = threading.Lock()

        started = time.monotonic()
        drained = sink.flush(0.2)
        assert time.monotonic() - started < 5, "flush ignored its ceiling"
        assert drained is False, "a queue that did not drain means the database is behind"


def _neverending():
    """A queue that always reports work outstanding."""
    made = queue.Queue()
    made.put(object())
    return made
