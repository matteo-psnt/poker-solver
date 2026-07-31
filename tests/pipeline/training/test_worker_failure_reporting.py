"""A dead worker must be reported as dead, and a wedged one must be killed.

Both behaviours were absent on a real Azure leg: 16 workers went silent, the
master waited its full 600s timeout and reported only "0/16 received", then the
shutdown sent SIGTERM to a worker that ignored it and never escalated -- so the
process could not exit and the node billed until the task's 24h wall clock.
"""

from __future__ import annotations

import itertools
import queue
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.pipeline.training.parallel_manager import gather


class _FakeProcess:
    """Stands in for mp.Process: only pid/exitcode/is_alive/terminate/kill/join."""

    def __init__(self, pid: int, exitcode: int | None = None, *, ignores: tuple[str, ...] = ()):
        self.pid = pid
        self.exitcode = exitcode
        self.ignores = ignores
        self.signals: list[str] = []

    def is_alive(self) -> bool:
        return self.exitcode is None

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.signals.append("SIGTERM")
        if "SIGTERM" not in self.ignores:
            self.exitcode = -15

    def kill(self) -> None:
        self.signals.append("SIGKILL")
        if "SIGKILL" not in self.ignores:
            self.exitcode = -9


def _manager(processes: list[_FakeProcess], results: list | None = None):
    q: queue.Queue = queue.Queue()
    for item in results or []:
        q.put(item)
    return SimpleNamespace(result_queue=q, processes=processes, num_workers=len(processes))


class TestDeathIsReportedAsDeath:
    def test_sigkill_names_the_oom_killer(self):
        mgr = _manager([_FakeProcess(101, exitcode=-9), _FakeProcess(102)])
        with pytest.raises(RuntimeError) as exc:
            gather.gather_worker_results(
                mgr, accept=lambda r: True, expected=2, timeout=30.0, description="batch results"
            )
        message = str(exc.value)
        assert "died" in message
        assert "pid=101" in message
        assert "SIGKILL" in message
        assert "OOM" in message, "SIGKILL without naming OOM is the unhelpful version"

    def test_it_does_not_wait_out_the_full_timeout(self):
        """The whole point: a death is known immediately, not after 600 seconds."""
        mgr = _manager([_FakeProcess(101, exitcode=-9)])
        start = time.monotonic()
        with pytest.raises(RuntimeError):
            gather.gather_worker_results(
                mgr, accept=lambda r: True, expected=1, timeout=600.0, description="batch results"
            )
        assert time.monotonic() - start < 10.0

    def test_a_clean_nonzero_exit_is_reported_with_its_status(self):
        mgr = _manager([_FakeProcess(7, exitcode=3)])
        with pytest.raises(RuntimeError, match="status 3"):
            gather.gather_worker_results(
                mgr, accept=lambda r: True, expected=1, timeout=5.0, description="batch results"
            )


class TestTimeoutStillReportsState:
    def test_timeout_names_how_many_are_alive(self):
        mgr = _manager([_FakeProcess(1), _FakeProcess(2)])
        with pytest.raises(RuntimeError) as exc:
            gather.gather_worker_results(
                mgr, accept=lambda r: True, expected=2, timeout=0.1, description="batch results"
            )
        message = str(exc.value)
        assert "2/2 workers still alive" in message
        assert "0/2 received" in message

    def test_results_that_do_arrive_are_returned(self):
        mgr = _manager([_FakeProcess(1)], results=[{"worker_id": 0}])
        results, interrupted = gather.gather_worker_results(
            mgr, accept=lambda r: True, expected=1, timeout=5.0, description="batch results"
        )
        assert results == [{"worker_id": 0}] and not interrupted

    def test_timeout_is_between_messages_not_for_the_whole_gather(self):
        """Two messages, each arriving just under the per-message timeout, must pass.

        Making the timeout absolute would silently start failing large batches.
        """
        mgr = _manager([_FakeProcess(1), _FakeProcess(2)])
        calls = itertools.count(1)

        def _delayed_get(timeout: float | None = None) -> dict[str, int]:
            time.sleep(0.05)
            n = next(calls)
            if n <= 2:
                return {"worker_id": n}
            raise queue.Empty

        mgr.result_queue = SimpleNamespace(get=_delayed_get)
        results, _ = gather.gather_worker_results(
            mgr, accept=lambda r: True, expected=2, timeout=0.2, description="batch results"
        )
        assert len(results) == 2


class TestShutdownEscalates:
    def _shutdown(self, processes):
        from src.pipeline.training.parallel_manager import lifecycle

        mgr = SimpleNamespace(
            num_workers=len(processes),
            processes=processes,
            job_queue=SimpleNamespace(put=lambda _msg: None),
            storage=SimpleNamespace(cleanup=lambda: None),
        )
        lifecycle.shutdown(cast(Any, mgr))
        return processes

    def test_a_worker_ignoring_sigterm_gets_sigkill(self):
        stubborn = _FakeProcess(3872, ignores=("SIGTERM",))
        self._shutdown([stubborn])
        assert stubborn.signals == ["SIGTERM", "SIGKILL"], (
            "SIGTERM alone is a request; the leg that hung ignored it"
        )
        assert not stubborn.is_alive()

    def test_a_cooperative_worker_is_never_signalled(self):
        polite = _FakeProcess(1, exitcode=0)
        self._shutdown([polite])
        assert polite.signals == []

    def test_a_worker_surviving_sigkill_is_reported_not_hidden(self, monkeypatch):
        # Asserted on the logger call rather than via caplog: another test in the
        # suite calls configure_logging(), which reconfigures propagation, so a
        # caplog assertion here passes or fails depending on test ORDER.
        from src.pipeline.training.parallel_manager import lifecycle

        logged: list[str] = []
        monkeypatch.setattr(lifecycle.logger, "error", lambda msg, *a: logged.append(str(msg)))
        unkillable = _FakeProcess(999, ignores=("SIGTERM", "SIGKILL"))
        self._shutdown([unkillable])
        assert any("survived SIGKILL" in line for line in logged), logged
