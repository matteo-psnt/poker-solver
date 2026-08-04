"""Per-leg outcome records: the node's account, Batch's, and their join."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from src.shared import leg_log

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _node(share, task_id, event, cause=None, **kw):
    return leg_log.write_node_record(share, task_id=task_id, event=event, cause=cause, **kw)


class TestNodeRecord:
    def test_started_record_is_not_terminal(self, tmp_path):
        _node(tmp_path, "task-a", "started")
        assert leg_log.read_legs(tmp_path)[0]["cause"] == "unresolved"

    def test_terminal_record_supersedes_started(self, tmp_path):
        _node(tmp_path, "task-a", "started")
        _node(tmp_path, "task-a", "finished", cause="completed", exit_code=0)

        rows = leg_log.read_legs(tmp_path)
        assert len(rows) == 1, "start + exit are one attempt, not two rows"
        assert rows[0]["cause"] == "completed"
        assert rows[0]["exit_code"] == 0
        # Both from the node's own two records.
        assert rows[0]["started_at"]
        assert rows[0]["ended_at"]

    def test_carries_the_run_identity(self, tmp_path):
        _node(
            tmp_path, "task-a", "started", run_id="run-xyz", op="train-static", config="production"
        )
        row = leg_log.read_legs(tmp_path)[0]
        assert (row["run_id"], row["op"], row["config"]) == (
            "run-xyz",
            "train-static",
            "production",
        )

    def test_timeout_stays_distinct_from_failure(self, tmp_path):
        """The RUN_TIMEOUT guard is a hang; Batch reports it as plain failure."""
        _node(tmp_path, "hung", "finished", cause="timeout", exit_code=124)
        _node(tmp_path, "crashed", "finished", cause="failed", exit_code=1)

        causes = {r["task_id"]: r["cause"] for r in leg_log.read_legs(tmp_path)}
        assert causes == {"hung": "timeout", "crashed": "failed"}


class TestJoin:
    """The case the module exists for: a leg killed before its trap could run."""

    def test_observer_explains_a_leg_the_node_never_finished(self, tmp_path):
        _node(tmp_path, "task-oom", "started", run_id="run-xyz")
        leg_log.write_observed_record(
            tmp_path,
            task_id="task-oom",
            job_id="poker-20260801",
            state="completed",
            result="failure",
            exit_code=137,
            failure={"code": "TaskEnded", "message": "node lost"},
            end_time="2026-08-01T10:00:00Z",
        )

        row = leg_log.read_legs(tmp_path)[0]
        assert row["cause"] == "failed"
        assert row["cause_source"] == "batch"
        assert row["run_id"] == "run-xyz", "the run identity comes from the node half"
        assert row["failure"]["code"] == "TaskEnded"
        assert row["exit_code"] == 137, (
            "the node's record carries a null exit_code, which must not shadow "
            "the only code that exists for a leg killed before its trap ran"
        )

    def test_node_account_wins_when_it_reached_a_terminal_event(self, tmp_path):
        """Batch calls a timed-out leg 'failure'; the node knows it was a hang."""
        _node(tmp_path, "task-hang", "finished", cause="timeout", exit_code=124)
        leg_log.write_observed_record(
            tmp_path, task_id="task-hang", job_id="j", state="completed", result="failure"
        )

        row = leg_log.read_legs(tmp_path)[0]
        assert row["cause"] == "timeout"
        assert row["cause_source"] == "node"

    def test_observer_only_leg_still_appears(self, tmp_path):
        """A task killed before the node wrote anything must not vanish."""
        leg_log.write_observed_record(
            tmp_path, task_id="task-ghost", job_id="j", state="completed", result="failure"
        )
        assert leg_log.read_legs(tmp_path)[0]["task_id"] == "task-ghost"

    def test_running_task_is_not_called_dead(self, tmp_path):
        leg_log.write_observed_record(tmp_path, task_id="t", job_id="j", state="running")
        assert leg_log.read_legs(tmp_path)[0]["cause"] == "running"


class TestBatchRetry:
    """A retry reuses the task id; the failed attempt must survive it."""

    def test_a_retry_does_not_erase_the_failed_attempt(self, tmp_path):
        _node(tmp_path, "leg-1", "started")
        _node(tmp_path, "leg-1", "finished", cause="killed", exit_code=137)
        _node(tmp_path, "leg-1", "started")  # Batch retries with the SAME id
        _node(tmp_path, "leg-1", "finished", cause="completed", exit_code=0)

        rows = sorted(leg_log.read_legs(tmp_path), key=lambda r: r["attempt"])
        assert [r["attempt"] for r in rows] == [1, 2]
        assert [r["cause"] for r in rows] == ["killed", "completed"], (
            "the OOM that caused the retry is the whole point of the record"
        )

    def test_the_observer_explains_only_the_latest_attempt(self, tmp_path):
        """Batch's executionInfo describes no earlier attempt, so it must not
        be attached to one -- that would explain the wrong death."""
        _node(tmp_path, "leg-1", "started")
        _node(tmp_path, "leg-1", "finished", cause="killed", exit_code=137)
        _node(tmp_path, "leg-1", "started")
        leg_log.write_observed_record(tmp_path, task_id="leg-1", job_id="j", state="running")

        rows = {r["attempt"]: r for r in leg_log.read_legs(tmp_path)}
        assert rows[1]["cause"] == "killed"
        assert rows[2]["cause"] == "running"

    def test_unresolved_reports_each_task_once(self, tmp_path):
        _node(tmp_path, "leg-1", "started")
        _node(tmp_path, "leg-1", "finished", cause="failed", exit_code=1)
        _node(tmp_path, "leg-1", "started")

        assert leg_log.unresolved_task_ids(tmp_path) == ["leg-1"]


class TestTornTerminalWrite:
    def test_a_torn_exit_record_leaves_the_leg_unresolved_not_absent(self, tmp_path):
        """write_text truncates, so the SIGKILL window can tear the exit file.

        The leg must still appear -- and as unresolved, so reconciliation asks
        Batch. Vanishing would be worse than never having written anything.
        """
        _node(tmp_path, "leg-torn", "started")
        (leg_log.legs_dir(tmp_path) / "leg-torn.1.exit.json").write_text('{"task_id": "leg')

        rows = leg_log.read_legs(tmp_path)
        assert [r["task_id"] for r in rows] == ["leg-torn"]
        assert rows[0]["cause"] == "unresolved"
        assert leg_log.unresolved_task_ids(tmp_path) == ["leg-torn"]


class TestCauseVocabulary:
    """A wrong terminal cause is worse than none: it suppresses reconciliation."""

    @pytest.mark.parametrize(
        "cause",
        [
            leg_log.CAUSE_COMPLETED,
            leg_log.CAUSE_FAILED,
            leg_log.CAUSE_TIMEOUT,
            leg_log.CAUSE_KILLED,
            leg_log.CAUSE_CANCELLED,
            leg_log.CAUSE_PARTIAL,
        ],
    )
    def test_every_node_cause_is_terminal(self, tmp_path, cause):
        _node(tmp_path, "t", "started")
        _node(tmp_path, "t", "finished", cause=cause)
        assert leg_log.read_legs(tmp_path)[0]["cause"] == cause
        assert leg_log.unresolved_task_ids(tmp_path) == []

    def test_an_oom_is_not_recorded_as_a_hang(self, tmp_path):
        """137 is SIGKILL from outside; `timeout` returns 124 even after its
        own --kill-after fires, so 137 never means the guard."""
        _node(tmp_path, "oom", "finished", cause=leg_log.CAUSE_KILLED, exit_code=137)
        _node(tmp_path, "hang", "finished", cause=leg_log.CAUSE_TIMEOUT, exit_code=124)

        causes = {r["task_id"]: r["cause"] for r in leg_log.read_legs(tmp_path)}
        assert causes == {"oom": "killed", "hang": "timeout"}

    def test_a_cancelled_leg_is_not_a_clean_completion(self, tmp_path):
        _node(tmp_path, "c", "finished", cause=leg_log.CAUSE_CANCELLED, exit_code=143)
        assert leg_log.read_legs(tmp_path)[0]["cause"] == "cancelled"


class TestReconcile:
    def test_only_unresolved_legs_are_written(self, tmp_path):
        _node(tmp_path, "done", "finished", cause="completed", exit_code=0)
        _node(tmp_path, "vanished", "started")

        explained = leg_log.reconcile(
            tmp_path,
            [
                {"task": "done", "state": "completed", "result": "success"},
                {"task": "vanished", "state": "completed", "result": "failure"},
            ],
        )

        assert explained == ["vanished"]
        assert not (leg_log.legs_dir(tmp_path) / "done.observed.json").exists(), (
            "a leg that reported its own exit needs no external explanation"
        )

    def test_unknown_tasks_are_ignored(self, tmp_path):
        _node(tmp_path, "mine", "started")
        assert leg_log.reconcile(tmp_path, [{"task": "someone-elses", "state": "completed"}]) == []

    def test_an_explained_leg_reads_back_as_an_outcome_not_a_state_string(self, tmp_path):
        """The whole join is worthless if the cause column says
        `batchtaskstate.completed`, so the shape reconcile consumes is pinned to
        the shape `batch.list_jobs_with_tasks` produces."""
        _node(tmp_path, "vanished", "started")
        leg_log.reconcile(
            tmp_path,
            [{"task": "vanished", "job": "poker-1", "state": "completed", "result": "failure"}],
        )
        row = next(r for r in leg_log.read_legs(tmp_path) if r["task_id"] == "vanished")
        assert row["cause"] == leg_log.CAUSE_FAILED


class TestRobustness:
    def test_a_half_written_record_does_not_break_the_listing(self, tmp_path):
        """Truncated files are the expected residue of the kills this explains."""
        _node(tmp_path, "good", "finished", cause="completed", exit_code=0)
        (leg_log.legs_dir(tmp_path) / "torn.1.exit.json").write_text('{"task_id": "torn"')

        rows = leg_log.read_legs(tmp_path)
        assert [r["task_id"] for r in rows] == ["good"]

    def test_missing_directory_reads_as_empty(self, tmp_path):
        assert leg_log.read_legs(tmp_path / "nothing-here") == []

    def test_format_table_handles_no_legs(self, tmp_path):
        assert "no leg records" in leg_log.format_table([])


class TestNodeSideConstraints:
    """The interpreter and dependency floor this module must clear is checked
    for the whole node package in tests/shared/node/test_node_interpreter.py --
    including a real 3.10 run, which this file could only approximate by
    grepping for names."""

    def test_importable_without_the_project_environment(self, tmp_path):
        """Proves the node-side contract on a bare interpreter, not just in-suite."""
        script = (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
            "from src.shared.leg_log import write_node_record;"
            f"write_node_record({str(tmp_path)!r}, task_id='t', event='started');"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
        assert (leg_log.legs_dir(tmp_path) / "t.1.start.json").exists()
