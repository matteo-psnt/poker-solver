"""The guard and the tee: a child that will not exit, and output nobody kept.

Every case here stands in for something the shell got wrong once. The most
expensive was a deadline that returned 124 with the trainer's workers still
running, holding the /dev/shm segments that then killed the NEXT task.
"""

from __future__ import annotations

import pytest

from src.shared.cloudtask.node import process
from tests.shared.cloudtask.node.conftest import python


class TestRunGuarded:
    def test_a_clean_exit_reports_zero(self, paths, log):
        assert process.run_guarded(python("pass"), cwd=paths.work.parent, timeout=5, log=log) == 0

    def test_a_failing_command_reports_its_code(self, paths, log):
        argv = python("import sys", "sys.exit(3)")
        assert process.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log) == 3

    @pytest.mark.timeout(15)
    def test_the_deadline_reports_124_not_the_signal_it_sent(self, paths, log):
        """`timeout`'s convention. The guard sends TERM, so the child reports
        143 -- but 143 means "cancelled" to the task record, and this is a hang."""
        argv = python("import time", "time.sleep(30)")
        assert process.run_guarded(argv, cwd=paths.work.parent, timeout=1, log=log) == (
            process.EXIT_TIMEOUT
        )

    @pytest.mark.timeout(15)
    def test_a_child_killed_from_outside_reports_137(self, paths, log):
        """137 is the OOM killer on a training node. Reporting it as 124 would
        call an OOM a hang AND, being terminal, stop `tasks` asking Batch."""
        argv = python("import os, signal", "os.kill(os.getpid(), signal.SIGKILL)")
        assert process.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log) == 137

    def test_a_command_that_cannot_start_is_not_an_exception(self, paths, log):
        code = process.run_guarded(
            ["/nonexistent/binary"], cwd=paths.work.parent, timeout=5, log=log
        )
        assert code == 1
        assert "could not start" in log.path.read_text()

    def test_output_is_tee_d_to_the_task_log(self, paths, log):
        argv = python("print('hello from the trainer')")
        process.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log)
        assert "hello from the trainer" in log.path.read_text()

    def test_stderr_is_captured_too(self, paths, log):
        argv = python("import sys", "sys.stderr.write('boom\\n')")
        process.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log)
        assert "boom" in log.path.read_text()

    def test_a_carriage_return_stream_still_reaches_the_log(self, paths, log):
        """tqdm emits `\\r`, not `\\n`. Reading by line would block for minutes
        and publish an empty log through exactly the window worth watching."""
        argv = python(
            "import sys",
            "sys.stdout.write('50%\\r'); sys.stdout.flush()",
            "import time; time.sleep(0.2)",
        )
        process.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log)
        assert "50%" in log.path.read_text()

    def test_json_stdout_is_captured_apart_from_the_log(self, paths, log):
        """precompute's stdout is a payload, not a log; its stderr still tees."""
        payload = paths.work / "out.json"
        argv = python(
            "import sys",
            'sys.stdout.write(\'{"output_dir": "/x"}\')',
            "sys.stderr.write('progress\\n')",
        )
        process.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log, stdout_to=payload)
        assert payload.read_text() == '{"output_dir": "/x"}'
        assert "progress" in log.path.read_text()
        assert "output_dir" not in log.path.read_text()


class TestTaskLogger:
    @staticmethod
    def _logger(tmp_path, monkeypatch):
        """A logger writing to a fake diagnostics container."""
        sent: dict[str, bytes] = {}
        monkeypatch.setattr(
            process.blobstore,
            "put_bytes",
            lambda _s, name, body: sent.__setitem__(name, body) or len(body),
        )
        return process.TaskLogger(tmp_path / "task.log", "https://a/diagnostics?sig=x"), sent

    def test_publishing_lands_the_log_in_the_container(self, tmp_path, monkeypatch):
        log, sent = self._logger(tmp_path, monkeypatch)
        log("something worth reading later")
        log.publish()
        assert b"something worth reading later" in sent["task.log"]

    def test_only_the_tail_is_published(self, tmp_path, monkeypatch):
        """A multi-hour tqdm stream is mostly progress-bar repaints that cost
        more to send than they inform."""
        monkeypatch.setattr(process, "PUBLISHED_LOG_BYTES", 64)
        log, sent = self._logger(tmp_path, monkeypatch)
        log("x" * 500)
        log("the end")
        log.publish()
        assert len(sent["task.log"]) == 64
        assert b"the end" in sent["task.log"]

    def test_a_store_that_refuses_does_not_kill_the_task(self, tmp_path, monkeypatch):
        """The observer must never fail the work it is observing."""

        def refuse(*args, **kwargs):
            raise OSError("the container went away")

        monkeypatch.setattr(process.blobstore, "put_bytes", refuse)
        log = process.TaskLogger(tmp_path / "task.log", "https://a/diagnostics?sig=x")
        log("something")
        log.publish()

    def test_without_a_credential_it_publishes_nothing(self, tmp_path, monkeypatch):
        """A task sealed with no diagnostics SAS keeps its log node-local; there
        is no second store to fall back to."""
        monkeypatch.setattr(
            process.blobstore, "put_bytes", lambda *_a: pytest.fail("published with no SAS")
        )
        log = process.TaskLogger(tmp_path / "task.log")
        log("something")
        log.publish()


class TestTheGuardReachesTheWholeTree:
    """`terminate()` reaches only `uv`; the trainer and its workers are
    grandchildren. A deadline that returned 124 while the workers kept running
    is how stale /dev/shm segments outlived a killed task and took down the
    NEXT one for that run."""

    @pytest.mark.timeout(30)
    def test_the_deadline_kills_the_grandchild_too(self, paths, log, tmp_path, monkeypatch):
        # The real grace is 120s, so the trainer can flush. This child ignores
        # TERM on purpose -- the case that motivated the guard -- so the test
        # would otherwise sit through the whole window.
        monkeypatch.setattr(process, "GRACE_SECONDS", 2)
        marker = tmp_path / "grandchild-alive"
        # A child that spawns a grandchild and then ignores TERM itself --
        # exactly the shape `uv run python -m trainer` has.
        child = python(
            "import subprocess, sys, signal, time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            f"g = subprocess.Popen([sys.executable, '-c', "
            f'"import time, pathlib; p = pathlib.Path({str(marker)!r});\\n"'
            f'"[ (p.write_text(str(i)), time.sleep(0.1)) for i in range(200) ]"])',
            "time.sleep(60)",
        )
        assert process.run_guarded(child, cwd=tmp_path, timeout=1, log=log) == process.EXIT_TIMEOUT

        import time as _time

        _time.sleep(0.5)
        before = marker.read_text() if marker.exists() else ""
        _time.sleep(0.5)
        after = marker.read_text() if marker.exists() else ""
        assert before == after, "the grandchild outlived the guard's deadline"
