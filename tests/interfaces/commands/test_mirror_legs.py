"""The node's own copy of its records, and what happens when it cannot be made.

This exists because the wrapper that writes leg documents CANNOT write them to
the database: `infra/run_task.py` runs on the pool's bootstrap interpreter
rather than the venv `uv sync` builds, so the driver is not on its path -- and
the started record is written before that sync happens at all.
"""

from __future__ import annotations

import argparse
import json

import pytest

from src.interfaces.commands import mirror_legs
from src.interfaces.errors import CommandError
from src.shared.cloudtask import task_log


def _legs(tmp_path, task="task-a"):
    directory = task_log.tasks_dir(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{task}.1.start.json").write_text(json.dumps({"task_id": task, "attempt": 1}))
    (directory / f"{task}.progress.json").write_text(json.dumps({"task_id": task, "done": 10.0}))
    (directory / "other-task.1.start.json").write_text(json.dumps({"task_id": "other-task"}))
    return directory


def _args(directory, task="task-a"):
    return argparse.Namespace(task=task, legs_dir=str(directory))


def test_no_dsn_is_a_clean_no_op(tmp_path, monkeypatch):
    """Dual-write is opt-in and an unset DSN is the pre-migration behaviour --
    the rollout and the rollback both. It must not read as a failure."""
    monkeypatch.setattr(mirror_legs.connect, "engine_from_environment", lambda **_: None)
    payload = mirror_legs.run(_args(_legs(tmp_path)))
    assert payload.enabled is False
    assert payload.written == 0
    assert payload.documents == 2, "it still says what it WOULD have written"


def test_only_this_tasks_documents_are_mirrored(tmp_path, monkeypatch):
    """A node knows which task it is. One that swept the directory would race
    every other node doing the same."""
    seen: list = []
    monkeypatch.setattr(mirror_legs.connect, "engine_from_environment", lambda **_: object())
    monkeypatch.setattr(
        mirror_legs.legs, "record_legs", lambda _e, rows: seen.extend(rows) or len(rows)
    )
    mirror_legs.run(_args(_legs(tmp_path)))
    assert {row[0] for row in seen} == {"task-a"}


def test_both_filename_shapes_survive(tmp_path, monkeypatch):
    """`<task>.<attempt>.start.json` is per ATTEMPT and `<task>.progress.json`
    is per TASK. Requiring the first dropped a third of the record once."""
    seen: list = []
    monkeypatch.setattr(mirror_legs.connect, "engine_from_environment", lambda **_: object())
    monkeypatch.setattr(
        mirror_legs.legs, "record_legs", lambda _e, rows: seen.extend(rows) or len(rows)
    )
    mirror_legs.run(_args(_legs(tmp_path)))
    assert {(row[1], row[2]) for row in seen} == {(1, "start"), (-1, "progress")}


def test_a_missing_directory_refuses(tmp_path):
    with pytest.raises(CommandError, match="No such legs directory"):
        mirror_legs.run(_args(tmp_path / "nope"))
