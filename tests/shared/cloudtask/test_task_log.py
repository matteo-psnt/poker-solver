"""The node's own account: what it writes, and the floor it must clear to write it.

The JOIN over these records -- whose view wins, attempt numbering, what may be
compacted -- is `tests/shared/test_task_history.py`, because that is where the
reading lives. What is left here is the half that runs on a node.
"""

from __future__ import annotations

import json
import subprocess
import sys

from src.shared import records, repo
from src.shared.cloudtask import task_log
from src.shared.cloudtask.kinds import Progress

REPO_ROOT = repo.ROOT


def _node(share, task_id, event, cause=None, **kw):
    return task_log.write_node_record(share, task_id=task_id, event=event, cause=cause, **kw)


class TestOneFilePerEventPerAttempt:
    """The layout is the safety: SMB has no atomic rename and no atomic append,
    so a torn write must not be able to destroy a record that already landed."""

    def test_start_and_exit_are_separate_files(self, tmp_path):
        """`write_text` truncates. One file for both would mean a kill mid-write
        made the task vanish from the listing entirely -- in exactly the SIGKILL
        window this record exists to explain."""
        _node(tmp_path, "task-a", task_log.EVENT_STARTED)
        _node(tmp_path, "task-a", task_log.EVENT_FINISHED, cause=task_log.CAUSE_COMPLETED)

        directory = task_log.tasks_dir(tmp_path)
        assert (directory / f"task-a.1{task_log.START_SUFFIX}").exists()
        assert (directory / f"task-a.1{task_log.EXIT_SUFFIX}").exists()

    def test_a_retry_writes_its_own_attempt_rather_than_overwriting(self, tmp_path):
        """Batch reuses the task id, and the failed attempt is the one worth
        keeping -- it holds the OOM that caused the retry."""
        _node(tmp_path, "task-1", task_log.EVENT_STARTED)
        _node(tmp_path, "task-1", task_log.EVENT_FINISHED, cause=task_log.CAUSE_KILLED)
        _node(tmp_path, "task-1", task_log.EVENT_STARTED)

        directory = task_log.tasks_dir(tmp_path)
        assert (directory / f"task-1.1{task_log.EXIT_SUFFIX}").exists()
        assert (directory / f"task-1.2{task_log.START_SUFFIX}").exists()

    def test_progress_is_overwritten_rather_than_accumulated(self, tmp_path):
        """Current state, not history: one file per tick per task would grow the
        thing that makes every read of this directory slow -- the file COUNT."""
        for done in (1, 2, 3):
            task_log.write_progress_record(
                tmp_path, task_id="t", progress=Progress(done=done, total=10, unit="rungs")
            )
        directory = task_log.tasks_dir(tmp_path)
        assert len(list(directory.glob(f"*{task_log.PROGRESS_SUFFIX}"))) == 1


class TestAttemptNumbering:
    def test_it_counts_bundled_starts_too(self, tmp_path):
        """The hazard that would corrupt the record silently: this number NAMES
        the file the next attempt writes, so missing a bundled start makes a
        retry overwrite the failure that caused it.

        The writer therefore needs `read_documents`, which is why that one
        reading primitive lives here rather than with the rest of the reading.
        """
        directory = task_log.tasks_dir(tmp_path)
        _node(tmp_path, "t", task_log.EVENT_STARTED)
        _node(tmp_path, "t", task_log.EVENT_FINISHED, cause=task_log.CAUSE_KILLED, exit_code=137)

        documents = task_log.read_documents(directory)
        records.write_snapshot(
            directory / f"test{task_log.BUNDLE_SUFFIX}",
            {"records": dict(documents)},
            records.REGISTRY[f"legs/*{task_log.BUNDLE_SUFFIX}"],
        )
        for name in documents:
            (directory / name).unlink()

        assert task_log._next_attempt(directory, "t") == 2


class TestNodeSideConstraints:
    """The interpreter and dependency floor this module must clear is checked
    for the whole node package in tests/shared/cloudtask/node/test_node_interpreter.py
    -- including a real 3.10 run, which this file could only approximate by
    grepping for names.

    It is also why the READING half is not in this package: a fail-closed guard
    walks every file here, so a laptop-only join would be held to a floor it has
    no reason to meet."""

    def test_importable_without_the_project_environment(self, tmp_path):
        """Proves the node-side contract on a bare interpreter, not just in-suite."""
        script = (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
            "from src.shared.cloudtask.task_log import write_node_record;"
            f"write_node_record({str(tmp_path)!r}, task_id='t', event='started');"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
        assert (task_log.tasks_dir(tmp_path) / "t.1.start.json").exists()


class TestOneTasksDocumentsAreReadWithoutTheDirectory:
    """`read_documents` parses every file in the directory. Two callers did that
    to answer a question about ONE task, and the second -- `_next_attempt`,
    reached from every progress sample -- cost every task on the pool FIVE
    MINUTES: the watcher thread and the final sample raced for the memo, so both
    paid it, 120s of join giving up plus 213s of the second read.
    """

    def test_it_counts_an_attempt_sealed_into_a_bundle(self, tmp_path):
        """Load-bearing, not tidy: if compaction swept an earlier attempt's
        start record into a bundle and this counted only loose files, a retry
        would reuse an attempt number and overwrite the record of the failure
        that caused it."""
        (tmp_path / "sealed.bundle.json").write_text(
            json.dumps({"records": {"task-a.1.start.json": {"task_id": "task-a"}}})
        )
        (tmp_path / "task-a.2.start.json").write_text(json.dumps({"task_id": "task-a"}))
        assert task_log._next_attempt(tmp_path, "task-a") == 3

    def test_a_loose_file_wins_over_the_same_name_in_a_bundle(self, tmp_path):
        (tmp_path / "sealed.bundle.json").write_text(
            json.dumps({"records": {"task-a.1.start.json": {"task_id": "task-a", "v": "old"}}})
        )
        (tmp_path / "task-a.1.start.json").write_text(json.dumps({"task_id": "task-a", "v": "new"}))
        found = task_log.read_task_documents(tmp_path, "task-a")
        assert found["task-a.1.start.json"]["v"] == "new"

    def test_another_tasks_files_are_not_read(self, tmp_path):
        for i in range(20):
            (tmp_path / f"other-{i}.1.start.json").write_text(json.dumps({"task_id": f"o{i}"}))
        (tmp_path / "task-a.1.start.json").write_text(json.dumps({"task_id": "task-a"}))
        assert set(task_log.read_task_documents(tmp_path, "task-a")) == {"task-a.1.start.json"}
