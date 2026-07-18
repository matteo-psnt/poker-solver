"""Tests for the client-side orchestration log."""

from __future__ import annotations

import json
from pathlib import Path

from src.shared import orchestration_log as ol


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_classify_status_maps_known_and_unknown() -> None:
    assert ol.classify_status("SUCCESS") == "completed"
    assert ol.classify_status("TERMINATED") == "cancelled"  # guillotine
    assert ol.classify_status("TIMEOUT") == "timeout"
    assert ol.classify_status("FAILURE") == "failed"  # includes OOM
    assert ol.classify_status("INIT_FAILURE") == "init_failed"
    assert ol.classify_status("PENDING") == "running"
    assert ol.classify_status("SOMETHING_NEW") == "unknown"


def test_record_spawn_appends_correlation(tmp_path: Path) -> None:
    log = tmp_path / "orchestration.jsonl"
    ol.record_spawn(
        run_id="run-abc",
        function="resume",
        object_id="fc-123",
        resources={"cpu": 64, "memory": 65536},
        extra={"additional": 4560000},
        path=log,
    )
    (record,) = _read_lines(log)
    assert record["event"] == "spawn"
    assert record["run_id"] == "run-abc"
    assert record["object_id"] == "fc-123"
    assert record["resources"] == {"cpu": 64, "memory": 65536}
    assert record["extra"] == {"additional": 4560000}
    assert record["ts"]  # timestamp present


def test_record_spawn_is_append_only(tmp_path: Path) -> None:
    log = tmp_path / "orchestration.jsonl"
    ol.record_spawn(run_id="r1", function="resume", object_id="a", path=log)
    ol.record_spawn(run_id="r2", function="evaluate", object_id="b", path=log)
    records = _read_lines(log)
    assert [r["object_id"] for r in records] == ["a", "b"]


def test_snapshot_call_records_status_from_live_call(tmp_path: Path) -> None:
    log = tmp_path / "orchestration.jsonl"

    class _Status:
        name = "TERMINATED"

    class _Node:
        status = _Status()

    class _FakeCall:
        def get_call_graph(self):
            return [_Node()]

    record = ol.snapshot_call(
        run_id="run-abc",
        function="resume",
        object_id="fc-123",
        call=_FakeCall(),
        path=log,
    )
    assert record["modal_status"] == "TERMINATED"
    assert record["exit_cause"] == "cancelled"
    assert "query_error" not in record
    assert _read_lines(log) == [record]


def test_snapshot_call_degrades_when_query_fails(tmp_path: Path) -> None:
    log = tmp_path / "orchestration.jsonl"

    class _BrokenCall:
        def get_call_graph(self):
            raise RuntimeError("modal unreachable")

    record = ol.snapshot_call(
        run_id="run-abc",
        function="resume",
        object_id="fc-123",
        call=_BrokenCall(),
        path=log,
    )
    assert record["modal_status"] is None
    assert record["exit_cause"] == "unknown"
    assert "modal unreachable" in record["query_error"]
    # Still durably recorded, so a failed query is visible rather than silent.
    assert _read_lines(log) == [record]
