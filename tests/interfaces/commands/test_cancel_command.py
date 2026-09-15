"""`cancel` learns a task's job from the record, so `--job` is no longer typed.

Batch addresses a task by (job, task); the caller knows only the task. Demanding
the job cost eleven failed calls across seven sessions, every one followed by a
`jobs` call to look it up by eye.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from src.interfaces.commands import cancel, tasks
from src.interfaces.errors import CommandError
from src.shared.task_history import TaskRow


def _rows(monkeypatch, rows: list[TaskRow]) -> dict[str, Any]:
    asked: dict[str, Any] = {}

    def _invoke_as(model, **over):
        asked.update(over)
        return tasks.TasksPayload(rows=rows)

    monkeypatch.setattr(tasks, "COMMAND", SimpleNamespace(invoke_as=_invoke_as))
    return asked


def _row(job_id: str, attempt: int = 1) -> TaskRow:
    return TaskRow(
        task_id="t-1", attempt=attempt, job_id=job_id, cause="running", cause_source="node"
    )


def test_the_job_comes_from_the_tasks_record_without_reconcile(monkeypatch):
    """Skipping reconcile is deliberate: the job is on the node's own start leg,
    and asking Batch about it would be a round trip to learn what the record
    already says."""
    asked = _rows(monkeypatch, [_row("poker-20260901")])
    assert cancel.job_of("t-1") == "poker-20260901"
    assert asked == {"task": "t-1", "skip_reconcile": True}


def test_an_unknown_task_is_refused_naming_the_flag(monkeypatch):
    _rows(monkeypatch, [])
    with pytest.raises(CommandError, match="--job"):
        cancel.job_of("t-1")


def test_a_row_recorded_before_the_field_is_refused_not_guessed(monkeypatch):
    _rows(monkeypatch, [_row("")])
    with pytest.raises(CommandError, match="--job"):
        cancel.job_of("t-1")


def test_an_explicit_job_skips_the_lookup(monkeypatch):
    terminated: list[tuple[str, str]] = []
    monkeypatch.setattr(cancel.CloudConfig, "load", staticmethod(lambda: object()))
    monkeypatch.setattr(cancel.batch, "client", lambda _config: object())
    monkeypatch.setattr(
        cancel.batch, "cancel_task", lambda _client, job, task: terminated.append((job, task))
    )
    monkeypatch.setattr(cancel, "job_of", lambda _task: pytest.fail("looked up anyway"))
    payload = cancel.run(argparse.Namespace(task="t-1", job="poker-j"))
    assert terminated == [("poker-j", "t-1")]
    assert (payload.job_id, payload.task_id) == ("poker-j", "t-1")
