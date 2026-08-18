"""The node's own account: what it writes, and the floor it must clear to write it.

The JOIN over these records -- whose view wins, attempt numbering, what may be
compacted -- is `tests/shared/test_task_history.py`, because that is where the
reading lives. What is left here is the half that runs on a node.
"""

from __future__ import annotations

import subprocess
import sys

from src.shared import repo
from src.shared.cloudtask import task_log
from src.shared.cloudtask.kinds import Progress

REPO_ROOT = repo.ROOT


def _node(task_id, event, cause=None, attempt=1, **kw):
    return task_log.node_record(task_id=task_id, attempt=attempt, event=event, cause=cause, **kw)


class TestTheRecordIsBuiltNotWritten:
    """It used to be one file per event per attempt, and the LAYOUT was the
    safety: SMB has no atomic rename and no atomic append, so a torn write must
    not destroy a record that already landed.

    Nothing is written now, so that hazard is gone and a different one replaces
    it -- the row is the only account, and the attempt it is keyed by is claimed
    rather than counted. See `node/test_legmirror.py::TestClaimingAnAttempt`.
    """

    def test_it_writes_no_file(self, tmp_path):
        _node("task-a", task_log.EVENT_STARTED)
        _node("task-a", task_log.EVENT_FINISHED, cause=task_log.CAUSE_COMPLETED)
        assert list(tmp_path.rglob("*.json")) == []

    def test_start_and_exit_stay_separate_records(self, tmp_path):
        """Still two records, because they say different things and the exit one
        is written by a trap that may know nothing the entry point computed."""
        started = _node("task-a", task_log.EVENT_STARTED)
        finished = _node("task-a", task_log.EVENT_FINISHED, cause=task_log.CAUSE_COMPLETED)
        assert started["event"] == task_log.EVENT_STARTED
        assert finished["event"] == task_log.EVENT_FINISHED
        assert finished["cause"] == task_log.CAUSE_COMPLETED

    def test_the_attempt_is_carried_not_counted(self, tmp_path):
        """A retry's records must belong to ITS attempt: the failed one holds
        the OOM that caused the retry, and overwriting it destroys the only
        account of why this task ran twice."""
        assert _node("task-1", task_log.EVENT_STARTED, attempt=2)["attempt"] == 2

    def test_progress_writes_no_file_at_all(self, tmp_path):
        """It used to overwrite one file per task, to keep the thing that makes
        every read of this directory slow -- the file COUNT -- from growing. It
        now writes none: the sample goes to the database, where it is read from,
        and a share directory of 14,000 files is not on a running task's path."""
        for done in (1, 2, 3):
            record = task_log.progress_record(
                task_id="t", progress=Progress(done=done, total=10, unit="rungs")
            )
            assert record["progress"]["done"] == done
            # TASK-scoped, matching the row's own key: one live sample per task,
            # replaced in place, never one per attempt.
            assert record["attempt"] == task_log.TASK_SCOPED
        assert list(tmp_path.rglob("*.json")) == []


class TestNodeSideConstraints:
    """The interpreter and dependency floor this module must clear is checked
    for the whole node package in tests/shared/cloudtask/node/test_node_interpreter.py
    -- including a real 3.10 run, which this file could only approximate by
    grepping for names.

    It is also why the READING half is not in this package: a fail-closed guard
    walks every file here, so a laptop-only join would be held to a floor it has
    no reason to meet."""

    def test_importable_without_the_project_environment(self):
        """Proves the node-side contract on a bare interpreter, not just in-suite.

        Still the floor that matters most: this module is imported BEFORE
        `uv sync`, so a third-party import reaching it kills the task at
        bootstrap -- before it can record the thing that would explain it.
        """
        script = (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
            "from src.shared.cloudtask.task_log import node_record;"
            "r = node_record(task_id='t', attempt=1, event='started');"
            "print(r['task_id'], r['attempt'])"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "t 1" in result.stdout
