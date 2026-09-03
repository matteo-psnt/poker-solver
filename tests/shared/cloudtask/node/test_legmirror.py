"""The wrapper stores its own records, and cannot be hurt by failing to.

This replaces a subprocess. The wrapper runs on the interpreter the pool's start
task installs, which is why it can import a driver at all -- and why the import
is deferred and caught: the node loads this package BEFORE `uv sync`, and a
module-level third-party import that is missing kills the task at bootstrap,
before it can write the record that would explain it.
"""

from __future__ import annotations

import json
from typing import Any

from src.shared.cloudtask import task_log
from src.shared.cloudtask.node import legmirror

DSN = "postgresql://u:p@h:5432/db"


def _legs(tmp_path, task: str = "task-a"):
    directory = task_log.tasks_dir(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{task}.1.start.json").write_text(
        json.dumps({"task_id": task, "attempt": 1, "ts": "2026-09-01T00:00:00+00:00"})
    )
    (directory / f"{task}.progress.json").write_text(
        json.dumps({"task_id": task, "done": 10.0, "ts": "2026-09-01T00:01:00+00:00"})
    )
    (directory / "other.1.start.json").write_text(json.dumps({"task_id": "other"}))
    return directory


def _stored(monkeypatch) -> list[list[dict[str, Any]]]:
    calls: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: calls.append(rows))
    return calls


class TestNothingItDoesCanCostTheTask:
    def test_an_unreachable_database_is_survivable(self, tmp_path, monkeypatch):
        def _explode(_dsn, _rows):
            raise RuntimeError("database is gone")

        monkeypatch.setattr(legmirror, "_store", _explode)
        _legs(tmp_path)
        logged: list[str] = []
        legmirror.publish(tmp_path, "task-a", dsn=DSN, log=logged.append)
        assert logged
        assert "not mirrored" in logged[0]

    def test_a_missing_driver_is_survivable(self, tmp_path, monkeypatch):
        """A node whose start task did not install it mirrors nothing. That is
        the same as having no DSN, which is already the policy."""

        def _missing(_dsn, _rows):
            raise ImportError("no module named psycopg")

        monkeypatch.setattr(legmirror, "_store", _missing)
        _legs(tmp_path)
        legmirror.publish(tmp_path, "task-a", dsn=DSN)

    def test_a_missing_legs_directory_is_survivable(self, tmp_path, monkeypatch):
        _stored(monkeypatch)
        legmirror.publish(tmp_path, "task-a", dsn=DSN)


class TestWhatItStores:
    def test_no_dsn_writes_nothing(self, tmp_path, monkeypatch):
        """The pre-migration behaviour, which is the rollout and the rollback."""
        calls = _stored(monkeypatch)
        _legs(tmp_path)
        legmirror.publish(tmp_path, "task-a", dsn="")
        assert calls == []

    def test_only_this_tasks_records(self, tmp_path, monkeypatch):
        calls = _stored(monkeypatch)
        _legs(tmp_path)
        legmirror.publish(tmp_path, "task-a", dsn=DSN)
        assert {row["task_id"] for row in calls[0]} == {"task-a"}

    def test_both_filename_shapes(self, tmp_path, monkeypatch):
        """`<task>.<attempt>.start.json` is per ATTEMPT; `<task>.progress.json`
        is per TASK. Requiring the first dropped a third of the record once."""
        calls = _stored(monkeypatch)
        _legs(tmp_path)
        legmirror.publish(tmp_path, "task-a", dsn=DSN)
        assert {(row["attempt"], row["leg"]) for row in calls[0]} == {
            (1, "start"),
            (task_log.TASK_SCOPED, "progress"),
        }

    def test_the_row_is_task_logs_and_not_a_second_one(self, tmp_path, monkeypatch):
        """The importer builds the same row from the same function. Two ways of
        building it is two answers about one record, which `--verify` would
        report as a divergence forever."""
        calls = _stored(monkeypatch)
        _legs(tmp_path)
        legmirror.publish(tmp_path, "task-a", dsn=DSN)
        start = next(row for row in calls[0] if row["leg"] == "start")
        assert start == task_log.leg_row(
            "task-a",
            1,
            "start",
            {"task_id": "task-a", "attempt": 1, "ts": "2026-09-01T00:00:00+00:00"},
        )


class TestTheRowIsWrittenWhereTheRecordIsMade:
    """The direct half. `publish` re-reads this task's files and upserts what it
    finds, which is self-healing -- but it can only heal a record that HAS a
    file, and progress is about to stop having one.
    """

    def test_one_record_becomes_one_row(self, tmp_path, monkeypatch):
        stored: list = []
        monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: stored.extend(rows))
        legmirror.record("task-a", 1, "start", {"task_id": "task-a", "ts": "t"}, dsn=DSN)
        assert [(r["task_id"], r["attempt"], r["leg"]) for r in stored] == [("task-a", 1, "start")]

    def test_it_builds_the_row_with_task_logs_builder(self, tmp_path, monkeypatch):
        """The node, the importer and the reader's write-back all go through
        `task_log.leg_row`. A second builder is a divergence `--verify` reports
        forever."""
        stored: list = []
        monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: stored.extend(rows))
        doc = {"task_id": "task-a", "done": 3.0, "ts": "t"}
        legmirror.record("task-a", task_log.TASK_SCOPED, "progress", doc, dsn=DSN)
        assert stored == [task_log.leg_row("task-a", task_log.TASK_SCOPED, "progress", doc)]

    def test_no_dsn_writes_nothing(self, tmp_path, monkeypatch):
        stored: list = []
        monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: stored.extend(rows))
        legmirror.record("task-a", 1, "start", {"task_id": "task-a"}, dsn="")
        assert stored == []

    def test_a_failure_cannot_cost_the_task(self, tmp_path, monkeypatch):
        def _explode(_dsn, _rows):
            raise RuntimeError("database is gone")

        monkeypatch.setattr(legmirror, "_store", _explode)
        logged: list[str] = []
        legmirror.record("task-a", 1, "exit", {"task_id": "task-a"}, dsn=DSN, log=logged.append)
        assert logged
        assert "not recorded" in logged[0]
