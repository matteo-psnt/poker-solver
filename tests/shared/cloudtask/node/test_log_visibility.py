"""A running task's log must be readable BEFORE it ends.

Two defects, both measured against a live console on 08-15, and both invisible
in every existing test because each one only bites over wall-clock:

* Training never called `TaskLogger.publish`. The eval handler did, between
  rungs; training ran `run_guarded` for the whole task and published only in
  `lifecycle.main`'s `finally`. So `logs --task` -- and the console, which has
  no other source -- answered "no published log yet" for the entire length of a
  multi-hour run, then produced the whole thing at once when it ended.
* The child's stdout was block-buffered. Python buffers at 8 KB when stdout is
  a pipe rather than a terminal, and every one of these is a pipe, so a `print`
  sat in the child until 8 KB of them accumulated. The tee was always prompt;
  it had nothing to forward.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.shared.cloudtask.node import progress
from src.shared.cloudtask.node.process import TaskLogger, run_guarded
from tests.shared.cloudtask.node.conftest import eventually


class TestTheLogReachesTheShareWhileRunning:
    def test_the_watcher_publishes_without_the_task_ending(self, paths, tmp_path, monkeypatch):
        """The defect: only `lifecycle.main`'s `finally` ever copied the log."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "task-live")
        paths.runs.mkdir(parents=True)
        logger = TaskLogger(tmp_path / "task.log", paths.share)
        logger("first thing the wrapper says")

        watcher = progress.ProgressWatcher(
            paths, logger, interval=9999, publish_log=logger.publish, log_interval=0.01
        )
        watcher.start()
        try:
            published = paths.share / "logs" / "task-live.log"
            eventually(published.is_file)
        finally:
            watcher.stop()

        assert "first thing the wrapper says" in published.read_text()

    def test_later_output_reaches_the_share_too(self, paths, tmp_path, monkeypatch):
        """One publish at startup would still leave a run's own output stranded."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "task-live")
        paths.runs.mkdir(parents=True)
        logger = TaskLogger(tmp_path / "task.log", paths.share)

        watcher = progress.ProgressWatcher(
            paths, logger, interval=9999, publish_log=logger.publish, log_interval=0.01
        )
        watcher.start()
        try:
            logger("iteration 200000")
            published = paths.share / "logs" / "task-live.log"
            eventually(lambda: published.is_file() and "iteration 200000" in published.read_text())
        finally:
            watcher.stop()

    def test_every_op_publishes_its_log_not_only_training(self):
        """On `ProgressWatcher`, not on the `LadderWatcher` subclass.

        The subclass is training-only -- an evaluation FETCHES rungs onto the
        node and would spend its coarse tick pushing somebody else's checkpoints
        back. But an evaluation's log is exactly as unreadable while it runs,
        and it is the one a failed rung has to be explained from.
        """
        assert "_send_log" in vars(progress.ProgressWatcher)
        assert "_send_log" not in vars(progress.LadderWatcher)

    def test_a_watcher_without_a_publisher_still_runs(self, paths, log):
        """The parameter is optional, and the existing callers that pass a bare
        write-a-line callable must not start raising in a daemon thread."""
        paths.runs.mkdir(parents=True)
        watcher = progress.ProgressWatcher(paths, log, interval=0.01, log_interval=0.01)
        watcher.start()
        watcher.stop()
        assert not watcher._thread.is_alive()


class TestPublishingIsAffordableOnATimer:
    def test_an_unchanged_log_is_not_copied_again(self, paths, tmp_path, monkeypatch):
        """This rewrites the whole 2 MB tail, so a quiet task on a 60s timer
        would otherwise resend it unchanged all day."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "task-quiet")
        logger = TaskLogger(tmp_path / "task.log", paths.share)
        logger("something")
        published = paths.share / "logs" / "task-quiet.log"

        logger.publish()
        first = published.stat().st_mtime_ns
        logger.publish()

        assert published.stat().st_mtime_ns == first, "an unchanged log was copied twice"

    def test_growth_after_a_skip_is_still_published(self, paths, tmp_path, monkeypatch):
        """The guard must not latch: a task that goes quiet and then speaks
        again is the normal shape of a training run."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "task-quiet")
        logger = TaskLogger(tmp_path / "task.log", paths.share)
        logger("before")
        logger.publish()
        logger.publish()
        logger("after")
        logger.publish()

        assert "after" in (paths.share / "logs" / "task-quiet.log").read_text()


class TestTheChildIsNotBlockBuffered:
    def test_a_single_print_arrives_before_the_child_exits(self, tmp_path):
        """Well under Python's 8 KB pipe buffer, and the child stays alive after
        printing -- so anything the tee sees was flushed rather than drained at
        exit. Without `PYTHONUNBUFFERED` this line does not arrive."""
        seen: list[bytes] = []
        logger = TaskLogger(tmp_path / "task.log", tmp_path / "share")
        original = logger.write

        def _record(chunk: bytes) -> None:
            seen.append(chunk)
            original(chunk)

        logger.write = _record  # type: ignore[method-assign]

        code = run_guarded(
            [
                sys.executable,
                "-c",
                "import time; print('hello from the child'); time.sleep(0.5)",
            ],
            cwd=Path.cwd(),
            timeout=30,
            log=logger,
        )

        assert code == 0
        assert b"hello from the child" in b"".join(seen)

    def test_the_child_environment_carries_the_flag(self, tmp_path):
        """Pinned directly, so the reason survives someone rewriting the tee."""
        out = tmp_path / "env.txt"
        logger = TaskLogger(tmp_path / "task.log", tmp_path / "share")

        run_guarded(
            [
                sys.executable,
                "-c",
                f"import os; open({str(out)!r}, 'w').write(os.environ.get('PYTHONUNBUFFERED', ''))",
            ],
            cwd=Path.cwd(),
            timeout=30,
            log=logger,
        )

        assert out.read_text() == "1"

    def test_the_rest_of_the_environment_survives(self, tmp_path, monkeypatch):
        """`env=` REPLACES the environment rather than adding to it, and the
        node's own settings -- `POKER_SOLVER_CACHE`, the task id, the share
        mount -- all arrive that way."""
        monkeypatch.setenv("POKER_SOLVER_MARKER", "kept")
        out = tmp_path / "env.txt"
        logger = TaskLogger(tmp_path / "task.log", tmp_path / "share")

        run_guarded(
            [
                sys.executable,
                "-c",
                f"import os; open({str(out)!r}, 'w').write(os.environ.get('POKER_SOLVER_MARKER', ''))",
            ],
            cwd=Path.cwd(),
            timeout=30,
            log=logger,
        )

        assert out.read_text() == "kept"
        assert os.environ.get("PYTHONUNBUFFERED") is None, "the parent must not be mutated"
