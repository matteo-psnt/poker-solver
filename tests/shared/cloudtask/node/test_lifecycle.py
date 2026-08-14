"""One task start to finish, and the account of how it ended.

The most expensive thing the shell got wrong lives here: the exit trap read
`$?` as zero on a signal death, so `cancel` recorded clean completions that were
never reconciled against Batch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.shared import cache, task_history
from src.shared.cloudtask import task_log
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import lifecycle
from src.shared.cloudtask.node.paths import NodePaths
from src.shared.cloudtask.node.process import Killed, TaskLogger
from tests.shared.cloudtask.node.conftest import python


class TestExitAccounting:
    @pytest.mark.parametrize(
        ("code", "cause"),
        [
            (0, task_log.CAUSE_COMPLETED),
            (lifecycle.EXIT_TIMEOUT, task_log.CAUSE_TIMEOUT),
            (143, task_log.CAUSE_CANCELLED),
            (130, task_log.CAUSE_CANCELLED),
            (137, task_log.CAUSE_KILLED),
            (1, task_log.CAUSE_FAILED),
            (2, task_log.CAUSE_FAILED),
        ],
    )
    def test_each_death_maps_to_its_own_cause(self, code, cause):
        """A wrong terminal cause is permanent: it suppresses reconciliation,
        so the observer half of the join is lost for good."""
        assert lifecycle._cause(code, None) == cause

    def test_an_explicit_outcome_wins_over_the_code(self):
        """An evaluate task exits 0 for Batch's retry economics, which is not a
        claim that all 30 rungs scored."""
        assert lifecycle._cause(0, task_log.CAUSE_PARTIAL) == task_log.CAUSE_PARTIAL


class TestMain:
    """The wiring `run_task.sh`'s traps used to carry."""

    @pytest.fixture(autouse=True)
    def _node(self, paths, monkeypatch):
        monkeypatch.setattr(NodePaths, "from_environment", classmethod(lambda cls: paths))
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "task-1")
        for key, value in {
            "RUN_OP": "train",
            "RUN_CONFIG": "quick_test",
            "RUN_TO": "1000",
            "RUN_ID": "",
            "RUN_SETS_JSON": "[]",
        }.items():
            monkeypatch.setenv(key, value)

    def test_a_signalled_task_records_cancelled_not_completed(self, paths, monkeypatch):
        """THE defect this port fixes. Bash's EXIT trap reads `$?` as zero when
        killed while blocked on a child -- measured: SIGTERM ran the trap with
        `$? = 0` and exited 143, so a cancelled task was recorded as clean and
        never reconciled against Batch."""
        monkeypatch.setattr(lifecycle, "_stage", lambda paths, log: 0)
        monkeypatch.setitem(lifecycle.HANDLERS, TaskName.TRAIN, _signalled)

        assert lifecycle.main() == 143
        (row,) = task_history.read_tasks(paths.share)
        assert row.cause == task_log.CAUSE_CANCELLED
        assert row.exit_code == 143

    def test_a_task_that_dies_before_the_sync_still_leaves_a_record(self, paths, monkeypatch):
        """The whole reason the started record is written first: a task dying
        during dependency install must not be indistinguishable from one that
        never ran."""
        monkeypatch.setattr(lifecycle, "_stage", lambda paths, log: 1)

        assert lifecycle.main() == 1
        (row,) = task_history.read_tasks(paths.share)
        assert row.cause == task_log.CAUSE_FAILED

    def test_a_bad_environment_is_a_message_not_a_traceback(self, paths, monkeypatch):
        monkeypatch.setenv("RUN_TO", "0")
        assert lifecycle.main() == 1
        (row,) = task_history.read_tasks(paths.share)
        assert row.cause == task_log.CAUSE_FAILED
        assert "ABSOLUTE" in (paths.share / "logs" / "task-1.log").read_text()

    def test_progress_is_published_even_on_a_failure(self, paths, monkeypatch):
        """An operator-cancelled task still leaves its progress on the share."""
        run_dir = paths.runs / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / ".run.json").write_text("{}")
        monkeypatch.setattr(lifecycle, "_stage", lambda paths, log: 0)
        monkeypatch.setitem(lifecycle.HANDLERS, TaskName.TRAIN, lambda *a: (1, None))

        lifecycle.main()
        assert (paths.archive / "run-a" / ".run.json").exists()


def _signalled(plan, paths, log):
    raise Killed(15)


class TestStage:
    def test_the_data_symlink_points_at_the_node_disk(self, paths, monkeypatch):
        """`precompute` writes to <base>/data/, and <base> is the throwaway code
        tree; the symlink is what lands it on the data disk instead."""
        paths.code.mkdir(parents=True)
        monkeypatch.setattr(lifecycle, "run_guarded", lambda *a, **k: 0)
        logger = TaskLogger(paths.work / "task.log", paths.share)
        try:
            assert lifecycle._stage(paths, logger) == 0
        finally:
            logger.close()
        assert (paths.code / "data").resolve() == paths.data.resolve()

    def test_a_stale_symlink_is_replaced(self, paths, monkeypatch):
        """A Batch retry reuses the extracted tree."""
        paths.code.mkdir(parents=True)
        (paths.code / "data").symlink_to(paths.work / "somewhere-else")
        monkeypatch.setattr(lifecycle, "run_guarded", lambda *a, **k: 0)
        logger = TaskLogger(paths.work / "task.log", paths.share)
        try:
            lifecycle._stage(paths, logger)
        finally:
            logger.close()
        assert (paths.code / "data").resolve() == paths.data.resolve()


class TestTheCacheSurvivesBetweenTasks:
    """Sharing the board cache across tasks on a node is the whole reason it
    lives on the data disk rather than in the task's HOME, which is wiped."""

    def _stage(self, paths, monkeypatch):
        paths.code.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(lifecycle, "run_guarded", lambda *a, **k: 0)
        logger = TaskLogger(paths.work / "task.log", paths.share)
        try:
            assert lifecycle._stage(paths, logger) == 0
        finally:
            logger.close()

    def test_the_cache_points_at_the_data_disk_not_the_task_home(self, paths, monkeypatch):
        """A Batch task's HOME is its own working directory, wiped with the
        task, so the ~/.cache default would rebuild the river's 2.6M-board
        cache on every single task."""
        monkeypatch.delenv(cache.ENV_OVERRIDE, raising=False)
        self._stage(paths, monkeypatch)
        assert os.environ[cache.ENV_OVERRIDE] == str(paths.work / "cache")

    def test_the_child_process_inherits_it(self, paths, monkeypatch):
        """`run_guarded` passes no `env=`, so the training subprocess -- and its
        16 workers -- see what the wrapper set. Checked against the REAL
        run_guarded, not the stub the other cases use."""
        monkeypatch.setenv(cache.ENV_OVERRIDE, "/mnt/work/cache")
        paths.work.mkdir(parents=True, exist_ok=True)
        logger = TaskLogger(paths.work / "child.log", paths.share)
        try:
            lifecycle.run_guarded(
                python("import os", f"print(os.environ['{cache.ENV_OVERRIDE}'])"),
                cwd=paths.work,
                timeout=10,
                log=logger,
            )
            assert "/mnt/work/cache" in logger.path.read_text()
        finally:
            logger.close()

    def test_it_is_writable_by_a_later_task(self, paths, monkeypatch):
        """`submit_task` sets no `user_identity`, so tasks run as Batch's
        default auto-user. A directory left with the first task's ownership and
        umask is one the SECOND task cannot write into -- which would silently
        undo the sharing this exists for."""
        monkeypatch.delenv(cache.ENV_OVERRIDE, raising=False)
        self._stage(paths, monkeypatch)
        mode = (paths.work / "cache").stat().st_mode & 0o777
        assert mode == 0o777, f"cache dir is {oct(mode)}, not shareable across task users"

    def test_a_cache_that_cannot_be_prepared_does_not_kill_the_task(self, paths, monkeypatch):
        monkeypatch.delenv(cache.ENV_OVERRIDE, raising=False)

        real = Path.chmod

        def refuse(self, *a, **k):
            if self.name == "cache":
                raise OSError("read-only")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "chmod", refuse)
        self._stage(paths, monkeypatch)  # must still return 0
