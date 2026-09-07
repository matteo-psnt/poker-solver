"""`tasks` answers from either store, through ONE join.

Compared across the whole record before this flip: 5,281 rows both ways, none
present on one side only, and the only differences were live tasks dispatched
before the node could mirror its legs -- their progress moves on the share and
nowhere else. That window closes as they drain; every task dispatched since
writes both.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.commands import tasks
from src.shared import task_history
from src.shared.cloudtask import task_log


def _row(task_id: str, attempt: int, cause: str) -> task_history.TaskRow:
    return task_history.TaskRow(task_id=task_id, attempt=attempt, cause=cause, cause_source="node")


class TestOnlyAskBatchWhatBatchCanAnswer:
    """Which rows Batch could still explain, decided over rows already in hand.
    Batch describes only a task's CURRENT attempt, so an earlier one is
    unresolved by construction: 1,326 non-terminal rows, 1,290 superseded."""

    def test_a_superseded_attempt_is_not_asked_about(self):
        rows = [_row("t", 1, "unresolved"), _row("t", 2, "completed")]
        assert tasks._still_open(rows) == []

    def test_the_latest_attempt_is(self):
        rows = [_row("t", 1, "failed"), _row("t", 2, "unresolved")]
        assert [r.attempt for r in tasks._still_open(rows)] == [2]

    def test_a_finished_task_is_not(self):
        assert tasks._still_open([_row("t", 1, "completed")]) == []


class TestAnObservationIsOnlyNewWhenItSaysSomethingNew:
    def _seen(self, **over: Any) -> list[dict[str, Any]]:
        return [{"task": "t", "job": "j", "state": "completed", "result": "success", **over}]

    def test_an_unknown_task_is_new(self):
        fresh = tasks._new_observations(self._seen(), [_row("t", 1, "unresolved")], {})
        assert set(fresh) == {"t"}

    def test_one_the_record_already_says_is_not(self):
        """`observed_at` is stamped on every read, so two observations of one
        finished task differ in a field that means nothing. Re-publishing those
        cost 14.1s per poll restating what the share already said."""
        stored = task_history.observed_record(
            task_id="t", job_id="j", state="completed", result="success"
        )
        fresh = tasks._new_observations(self._seen(), [_row("t", 1, "unresolved")], {"t": stored})
        assert fresh == {}

    def test_a_task_nobody_asked_about_is_ignored(self):
        """Batch may answer about more than was asked; a row that is not an open
        question must not be written back."""
        assert tasks._new_observations(self._seen(), [], {}) == {}


def test_the_row_written_back_is_the_one_everything_else_builds():
    """The node, the importer and this all build a leg row through
    `task_log.leg_row`. A second builder is a divergence `--verify` reports
    forever."""
    document = task_history.observed_record(task_id="t", job_id="j", state="completed")
    from src.adapters.postgres import observations

    assert observations._LEG == "observed"
    assert task_log.leg_row("t", task_log.TASK_SCOPED, "observed", document)["attempt"] == (
        task_log.TASK_SCOPED
    )


def test_a_local_directory_still_bypasses_both(tmp_path, monkeypatch):
    """`--tasks-dir` reads a fetched copy and must not reach for a database."""
    monkeypatch.setattr(tasks.connect, "engine_from_environment", lambda: object())
    payload = tasks.run(argparse.Namespace(tasks_dir=str(tmp_path), skip_reconcile=True, limit=0))
    assert payload.rows == []
