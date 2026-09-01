"""Copying a task's record must never cost the task.

The share holds the record and this is the copy. Every call site is on a path
where raising would kill work that has already succeeded -- the exit path most
of all, where the task is finished and only the account of it is in flight.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.shared.cloudtask import task_log
from src.shared.cloudtask.node import mirror


def test_it_asks_for_this_task_and_the_legs_directory(tmp_path, monkeypatch):
    seen: list[list[str]] = []

    def _run(argv, **_kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mirror.subprocess, "run", _run)
    mirror.publish(tmp_path, "task-a", cwd=tmp_path)
    (argv,) = seen
    assert argv[:4] == ["uv", "run", "poker-solver", "mirror-legs"]
    assert "task-a" in argv
    assert str(task_log.tasks_dir(tmp_path)) in argv


def test_no_uv_yet_is_survivable(tmp_path, monkeypatch):
    """Expected before `uv sync` has run: there is no `uv run` to call, and the
    started record is written before that sync by design."""

    def _missing(*_a, **_k):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(mirror.subprocess, "run", _missing)
    logged: list[str] = []
    mirror.publish(tmp_path, "task-a", cwd=tmp_path, log=logged.append)
    assert logged
    assert "unavailable" in logged[0]


def test_a_hung_mirror_is_survivable(tmp_path, monkeypatch):
    """Bounded below the 120s between progress ticks, so two cannot overlap."""

    def _hang(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, mirror.TIMEOUT_SECONDS)

    monkeypatch.setattr(mirror.subprocess, "run", _hang)
    mirror.publish(tmp_path, "task-a", cwd=tmp_path)


def test_a_nonzero_exit_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mirror.subprocess,
        "run",
        lambda argv, **_k: subprocess.CompletedProcess(argv, 1, "", "boom"),
    )
    logged: list[str] = []
    mirror.publish(tmp_path, "task-a", cwd=tmp_path, log=logged.append)
    assert logged
    assert "rc=1" in logged[0]


def test_the_timeout_survives_a_cold_first_invocation():
    """MEASURED on a node: the first `uv run` after a fresh `uv sync` took over
    a minute -- the project install plus a cold import off an empty page cache
    -- and a 60s ceiling timed it out. Nothing waits on this, and the watcher
    publishes from ONE thread so two mirrors cannot overlap, which is what makes
    a generous ceiling safe rather than merely tolerable."""
    assert mirror.TIMEOUT_SECONDS >= 120


def test_it_is_stdlib_only(tmp_path):
    """`shared/cloudtask` is loaded on the node's bootstrap interpreter. The
    guard in `test_imports.py` covers the package; this states the reason this
    module in particular is a subprocess and not an import."""
    source = Path(mirror.__file__).read_text()
    assert "import subprocess" in source
    assert "sqlalchemy" not in source
