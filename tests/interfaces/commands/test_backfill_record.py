"""Reading `legs/` wrong loses a third of it silently, which is what happened.

Two shapes and two clocks, because two different programs write into that
directory. Both defects below were found by arithmetic -- 13,120 files produced
8,849 rows and the numbers did not reconcile -- not by anything failing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.adapters.postgres import legs
from src.interfaces.commands import backfill_record
from src.shared import task_history


class _Leg:
    """Stands in for the model: records what the importer decided."""

    def __init__(self, **fields: Any):
        self.__dict__.update(fields)


class _Models:
    Leg = _Leg


def _rows(monkeypatch, documents: dict[str, dict[str, Any]], tmp_path):
    monkeypatch.setattr(
        backfill_record,
        "_leg_rows",
        backfill_record._leg_rows,  # keep the real one; only its input is faked
    )
    import src.shared.cloudtask.task_log as task_log

    monkeypatch.setattr(task_log, "read_documents", lambda _d: documents)
    return backfill_record._leg_rows(tmp_path, _Models)


class TestBothNameShapesSurvive:
    """`progress` and `observed` carry NO attempt, because they are written per
    TASK. Requiring one dropped 4,591 of 13,440 documents -- including every
    `observed` leg, which is the only account of a death the node did not
    survive."""

    def test_an_attempt_scoped_leg_keeps_its_attempt(self, monkeypatch, tmp_path):
        rows = _rows(
            monkeypatch, {"task-a.2.exit.json": {"ts": "2026-01-01", "cause": "ok"}}, tmp_path
        )
        assert [(r.task_id, r.attempt, r.leg) for r in rows] == [("task-a", 2, "exit")]

    @pytest.mark.parametrize("leg", ["progress", "observed"])
    def test_a_task_scoped_leg_is_kept_under_the_sentinel(self, monkeypatch, tmp_path, leg):
        rows = _rows(monkeypatch, {f"task-a.{leg}.json": {"ts": "2026-01-01"}}, tmp_path)
        assert [(r.task_id, r.attempt, r.leg) for r in rows] == [
            ("task-a", task_history.TASK_SCOPED, leg)
        ]

    def test_the_sentinel_does_not_collide_with_a_real_first_attempt(self, monkeypatch, tmp_path):
        """Flattening task-scoped legs onto attempt 0 would overwrite the record
        of attempt 0 itself."""
        rows = _rows(
            monkeypatch,
            {
                "task-a.0.exit.json": {"ts": "2026-01-01", "cause": "killed"},
                "task-a.observed.json": {"observed_at": "2026-01-02", "state": "completed"},
            },
            tmp_path,
        )
        assert len(rows) == 2
        assert {r.attempt for r in rows} == {0, task_history.TASK_SCOPED}

    def test_a_task_id_containing_dots_still_parses(self, monkeypatch, tmp_path):
        rows = _rows(monkeypatch, {"a.b.c.3.start.json": {"ts": "2026-01-01"}}, tmp_path)
        assert [(r.task_id, r.attempt, r.leg) for r in rows] == [("a.b.c", 3, "start")]


class TestEveryWriterHasItsOwnClock:
    """The node stamps `ts`; the READER stamps `observed_at`, because it is
    saying when IT looked. Reading only `ts` left 1,823 observed legs with no
    time, against a NOT NULL column."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("ts", "2026-01-01"),
            ("observed_at", "2026-01-02"),
            ("end_time", "2026-01-03"),
            ("start_time", "2026-01-04"),
        ],
    )
    def test_each_writers_field_is_found(self, field, value):
        assert legs.leg_values("t", 1, "start", {field: value})["at"] == value

    def test_ts_wins_when_several_are_present(self):
        """The node's own account first: `observed_at` is when a reader looked,
        which is a different fact."""
        assert (
            legs.leg_values(
                "t", 1, "start", {"ts": "node", "observed_at": "reader", "end_time": "batch"}
            )["at"]
            == "node"
        )

    def test_a_document_with_no_clock_at_all_reports_none(self):
        assert legs.leg_values("t", 1, "start", {"state": "running"})["at"] is None


class TestLegacyEvalDocumentsAreSkipped:
    """`eval-*` and `record-*` are the pre-substrate shapes, and a legacy record
    points at the OLD filename -- so reading both enters one evaluation twice.
    Measured historically at 63 rows becoming 110."""

    @pytest.mark.parametrize("prefix", ["eval-", "record-"])
    def test_a_legacy_document_is_not_imported(self, tmp_path, prefix):
        evals = tmp_path / "evals"
        evals.mkdir()
        body = {
            "run_id": "run-a",
            "method": "exact_br",
            "knobs": {"base_seed": 7},
            "timestamp": "2026-01-01T00:00:00+00:00",
            "results": {"exploitability_mbb": 900.0},
        }
        (evals / f"{prefix}20260101_000000-abc.json").write_text(json.dumps(body))
        (evals / "20260101_000000-abc.json").write_text(json.dumps(body))
        rows = backfill_record._eval_rows(tmp_path, _EvalModels)
        assert len(rows) == 1, "the legacy twin must not enter the index a second time"

    @pytest.mark.parametrize("missing", ["knobs", "timestamp"])
    def test_an_untierable_document_is_withheld(self, tmp_path, missing):
        """With no knobs or no timestamp it hashes into `(method, None, ...)`
        and sorts to year 1 AD -- and since tiers rank by coverage, a pile of
        them becomes the DEFAULT curve."""
        evals = tmp_path / "evals"
        evals.mkdir()
        body = {
            "run_id": "run-a",
            "method": "exact_br",
            "knobs": {"base_seed": 7},
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        del body[missing]
        (evals / "20260101_000000-abc.json").write_text(json.dumps(body))
        assert backfill_record._eval_rows(tmp_path, _EvalModels) == []


class _Eval:
    def __init__(self, **fields: Any):
        self.__dict__.update(fields)


class _EvalModels:
    Eval = _Eval
