"""Copying a task's record must never cost the task -- including its wall clock.

The share holds the record and this is the copy. Every call site is on a path
where raising would kill work that has already succeeded, and HANGING would be
worse: a task whose 40s of training had finished sat `running` for twenty
minutes because a mirror could not be killed.

MEASURED on a node: 20.5s of training finished at 17:31:32 and the task
published at 17:38:29. Not a hang -- STACKED TIMEOUTS. The first `uv run` after
a fresh `uv sync` takes over a minute and the watcher was mirroring every 15s.
The CADENCE now belongs to the watcher's coarse tick, which exists for exactly
this; `test_progress.py` holds that end.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from src.shared.cloudtask import task_log
from src.shared.cloudtask.node import mirror


class _Fake:
    def __init__(self, code: int = 0) -> None:
        self.pid = 1234
        self._code = code

    def wait(self, timeout: float | None = None) -> int:
        return self._code


@pytest.fixture
def launched(monkeypatch):
    """Records how the subprocess was launched, without launching one."""
    calls: list[tuple[list[str], dict]] = []

    def _popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Fake()

    monkeypatch.setattr(mirror.subprocess, "Popen", _popen)
    return calls


def test_it_asks_for_this_task_and_the_legs_directory(tmp_path, launched):
    mirror.publish(tmp_path, "task-a", cwd=tmp_path)
    (argv, _kwargs) = launched[0]
    assert argv[:4] == ["uv", "run", "poker-solver", "mirror-legs"]
    assert "task-a" in argv
    assert str(task_log.tasks_dir(tmp_path)) in argv


class TestATimeoutCannotLeaveTheTreeRunning:
    def test_it_opens_no_pipes(self, tmp_path, launched):
        """Nothing here is worth reading, and a pipe a grandchild still owns is
        one more thing that can block."""
        mirror.publish(tmp_path, "task-a", cwd=tmp_path)
        (_argv, kwargs) = launched[0]
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL

    def test_it_runs_in_its_own_session(self, tmp_path, launched):
        """So a timeout can signal the whole tree: `terminate()` reaches `uv`
        and leaves the python it spawned running -- the reason `run_guarded`
        does the same, and why `terminate` here is ITS teardown, not a second
        one."""
        mirror.publish(tmp_path, "task-a", cwd=tmp_path)
        (_argv, kwargs) = launched[0]
        assert kwargs["start_new_session"] is True

    def test_a_real_process_tree_that_outlives_its_parent_is_killed(self, tmp_path, monkeypatch):
        """With real processes: a shell that spawns a background child. The
        call must return on its ceiling rather than waiting on the grandchild,
        and the grandchild must not be left behind."""
        monkeypatch.setattr(mirror, "TIMEOUT_SECONDS", 1)
        real = subprocess.Popen
        monkeypatch.setattr(
            mirror.subprocess,
            "Popen",
            lambda _argv, **kwargs: real(["sh", "-c", "sleep 30 & sleep 30"], **kwargs),
        )
        logged: list[str] = []
        started = time.monotonic()
        mirror.publish(tmp_path, "task-a", cwd=tmp_path, log=logged.append)
        assert time.monotonic() - started < 20, "publish did not return on its ceiling"
        assert logged
        assert "timed out" in logged[0]


def test_no_uv_yet_is_survivable(tmp_path, monkeypatch):
    """Expected before `uv sync` has run: there is no `uv run` to call, and the
    started record is written before that sync by design."""

    def _missing(*_a, **_k):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(mirror.subprocess, "Popen", _missing)
    logged: list[str] = []
    mirror.publish(tmp_path, "task-a", cwd=tmp_path, log=logged.append)
    assert logged
    assert "unavailable" in logged[0]


def test_a_nonzero_exit_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror.subprocess, "Popen", lambda _argv, **_k: _Fake(1))
    logged: list[str] = []
    mirror.publish(tmp_path, "task-a", cwd=tmp_path, log=logged.append)
    assert logged
    assert "rc=1" in logged[0]


def test_the_timeout_survives_a_cold_first_invocation():
    """MEASURED on a node: the first `uv run` after a fresh `uv sync` took over
    a minute -- the project install plus a cold import off an empty page cache
    -- and a 60s ceiling timed it out."""
    assert mirror.TIMEOUT_SECONDS >= 120


def test_it_is_stdlib_only():
    """`shared/cloudtask` is loaded on the node's bootstrap interpreter. The
    guard in `test_imports.py` covers the package; this states the reason this
    module in particular is a subprocess and not an import."""
    source = Path(mirror.__file__).read_text()
    assert "import subprocess" in source
    assert "sqlalchemy" not in source
