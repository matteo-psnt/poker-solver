"""Closing a run is writing an INFERENCE into the record, so the wrong calls matter.

Three ways to get it wrong, each pinned below: closing a run that is still
training (deletes a live ladder, once `prune-checkpoints` believes it), closing
a run nothing can speak for (a guess recorded as a fact), and recording every
death as a failure when the last task was cancelled.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from src.interfaces.commands import reconcile_runs
from src.pipeline.training.run_tracker.tracker import RunTracker
from src.shared import run_events
from src.shared.config import Config


class _Task:
    """The fields of `task_history.TaskRow` this command reads."""

    def __init__(
        self,
        run_id: str,
        cause: str,
        task_id: str = "t1",
        ended_at: str = "2026-01-01",
        op: str = "train",
    ):
        self.op = op
        self.run_id = run_id
        self.cause = cause
        self.task_id = task_id
        self.ended_at = ended_at
        self.cause_source = "node"


def _run_dir(tmp_path, name: str, status: str | None):
    """A run built by the REAL writer, then given a status.

    `_status_of` goes through `RunMetadata.load`, so a hand-rolled `created`
    event makes the run unreadable rather than merely sparse -- the fold refuses
    a missing config. Using `RunTracker` means the fixture cannot drift from
    what a run actually looks like, which is the whole point here.
    """
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    tracker = RunTracker(
        run_dir=directory,
        config_name="quick_test",
        config=Config.default(),
        action_config_hash="abc123",
    )
    # Construction alone writes nothing; the record exists once something is
    # recorded into it.
    tracker.update(iterations=10, runtime_seconds=1.0, num_infosets=5, storage_capacity=100)
    if status is not None and status != "running":
        run_events.append(directory, run_events.STATUS, status=status)
    return directory


def _plan(monkeypatch, tmp_path, tasks: list[Any], **args: Any):
    # Patched at the COMPOSE seam, because that is where the rows now come from:
    # `reconcile-runs` asks `tasks`, which materialises legs/ and reconciles the
    # unresolved ones against Batch. Patching lower would test a path the
    # command no longer takes -- and would reach the network from a unit test.
    # `Command` is a FROZEN dataclass, so the fake replaces the module
    # attribute rather than one of its fields.
    monkeypatch.setattr(
        reconcile_runs.tasks_command,
        "COMMAND",
        type("C", (), {"invoke_as": staticmethod(lambda _cls: type("P", (), {"rows": tasks})())}),
    )

    class _Config:
        share_name = "share"

        @staticmethod
        def load():
            return _Config()

    import contextlib

    @contextlib.contextmanager
    def _root(_args):
        yield tmp_path

    monkeypatch.setattr(reconcile_runs, "records_root", _root)
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.interfaces.cloud.config",
        type("m", (), {"CloudConfig": _Config}),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.interfaces.cloud.store",
        type("m", (), {"share": type("s", (), {"share_client": staticmethod(lambda _c: None)})}),
    )
    namespace = argparse.Namespace(apply=False, runs=None, **args)
    return reconcile_runs.run(namespace)


class TestItDoesNotCloseWhatIsAlive:
    # The ON-DISK spellings, which are not the phase names: `cause_of` maps
    # QUEUED to "active" and STARTING to "preparing", because those are the
    # words records written months ago already carry.
    @pytest.mark.parametrize("cause", ["active", "preparing", "running", "unknown"])
    def test_a_run_with_an_in_flight_task_is_left_open(self, monkeypatch, tmp_path, cause):
        """The expensive mistake: `prune-checkpoints` trusts this status, so a
        wrongly-closed run has its live ladder deleted."""
        _run_dir(tmp_path, "run-a", "running")
        plan = _plan(monkeypatch, tmp_path, [_Task("run-a", cause)])
        assert plan.closures == []
        assert plan.unsettled == ["run-a"]

    def test_one_live_task_protects_a_run_with_several_dead_ones(self, monkeypatch, tmp_path):
        """A resumed run has many tasks; ANY live one means it is training."""
        _run_dir(tmp_path, "run-a", "running")
        tasks = [
            _Task("run-a", "killed", "t1", "2026-01-01"),
            _Task("run-a", "failed", "t2", "2026-01-02"),
            _Task("run-a", "running", "t3", ""),
        ]
        assert _plan(monkeypatch, tmp_path, tasks).closures == []


class TestAbsenceOfEvidenceProtects:
    def test_a_run_with_no_task_record_is_left_open(self, monkeypatch, tmp_path):
        """Nothing can speak for it, and a terminal status invented here would
        be a guess written into the record as a fact."""
        _run_dir(tmp_path, "run-a", "running")
        plan = _plan(monkeypatch, tmp_path, [])
        assert plan.closures == []
        assert plan.no_evidence == ["run-a"]


class TestAZeroedLogIsNotARunningRun:
    """The measured failure: publish truncation zeroes a `run.jsonl`, and an
    empty log made `tail_value` return its `running` default. Seven records got
    a status event appended to a file that held nothing else."""

    def test_an_empty_log_is_not_closable(self, monkeypatch, tmp_path):
        directory = tmp_path / "run-a"
        directory.mkdir()
        (directory / "run.jsonl").write_text("")
        plan = _plan(monkeypatch, tmp_path, [_Task("run-a", "killed")])
        assert plan.open_runs == 0, "an empty log must not count as an open run"
        assert plan.closures == []

    def test_a_legacy_run_json_run_is_read_from_it_not_defaulted(self, monkeypatch, tmp_path):
        """The real miss: runs written before the event log carry `.run.json`
        and no log. Reading only the log made nine of them report `running`
        when their snapshot said `completed`."""
        directory = tmp_path / "run-a"
        directory.mkdir()
        (directory / "run.jsonl").write_text("")
        (directory / ".run.json").write_text(
            json.dumps(
                {
                    "run_id": "run-a",
                    "config_name": "production",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "iterations": 30_000_000,
                    "config": {},
                }
            )
        )
        plan = _plan(monkeypatch, tmp_path, [_Task("run-a", "killed")])
        assert plan.open_runs == 0, "it is completed, and the snapshot says so"
        assert plan.closures == []

    def test_a_log_with_no_created_event_is_not_closable(self, monkeypatch, tmp_path):
        """Half-written is the same problem: without `created` there is no run
        identity, so there is nothing a terminal status would be about."""
        directory = tmp_path / "run-a"
        directory.mkdir()
        run_events.append(directory, run_events.PROGRESS, iteration=10)
        assert _plan(monkeypatch, tmp_path, [_Task("run-a", "killed")]).closures == []


class TestItReadsTheCauseRatherThanAssuming:
    def test_a_cancelled_task_makes_a_cancelled_run(self, monkeypatch, tmp_path):
        _run_dir(tmp_path, "run-a", "running")
        (closure,) = _plan(monkeypatch, tmp_path, [_Task("run-a", "cancelled")]).closures
        assert closure.status == "cancelled"

    @pytest.mark.parametrize("cause", ["killed", "timeout", "failed", "partial"])
    def test_a_task_that_died_makes_a_failed_run(self, monkeypatch, tmp_path, cause):
        _run_dir(tmp_path, "run-a", "running")
        (closure,) = _plan(monkeypatch, tmp_path, [_Task("run-a", cause)]).closures
        assert closure.status == "failed"

    def test_a_task_that_exited_cleanly_makes_an_abandoned_run_not_a_failed_one(
        self, monkeypatch, tmp_path
    ):
        """The distinction the real dry run forced: 15 of 17 closable runs had a
        last task that exited cleanly. Nothing failed -- the task finished its
        chunk and nobody continued the run. Recording that as `failed` writes a
        false statement into the record."""
        _run_dir(tmp_path, "run-a", "running")
        (closure,) = _plan(monkeypatch, tmp_path, [_Task("run-a", "completed")]).closures
        assert closure.status == "abandoned"

    def test_the_last_task_decides(self, monkeypatch, tmp_path):
        _run_dir(tmp_path, "run-a", "running")
        tasks = [
            _Task("run-a", "killed", "t1", "2026-01-01"),
            _Task("run-a", "cancelled", "t2", "2026-01-09"),
        ]
        (closure,) = _plan(monkeypatch, tmp_path, tasks).closures
        assert closure.status == "cancelled"
        assert closure.task_id == "t2"


class TestOnlyATrainingTaskSpeaksForARun:
    def test_a_score_task_alone_is_not_evidence(self, monkeypatch, tmp_path):
        """A `score` task carries the run_id of the run it SCORES. Read as
        evidence it would pick the run's status from an evaluation's cause,
        and `abandoned` versus `failed` is precisely the question it cannot
        answer."""
        _run_dir(tmp_path, "run-a", "running")
        plan = _plan(monkeypatch, tmp_path, [_Task("run-a", "completed", op="evaluate")])
        assert plan.closures == []
        assert plan.no_evidence == ["run-a"]

    def test_a_training_task_decides_even_when_a_later_score_exists(self, monkeypatch, tmp_path):
        _run_dir(tmp_path, "run-a", "running")
        tasks = [
            _Task("run-a", "killed", "t1", "2026-01-01", op="train"),
            _Task("run-a", "completed", "t2", "2026-01-09", op="evaluate"),
        ]
        (closure,) = _plan(monkeypatch, tmp_path, tasks).closures
        assert closure.status == "failed", "the TRAINING task's cause, not the score's"
        assert closure.task_id == "t1"


class TestItOnlyLooksAtOpenRuns:
    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
    def test_an_already_closed_run_is_not_reopened_or_recounted(
        self, monkeypatch, tmp_path, status
    ):
        _run_dir(tmp_path, "run-a", status)
        plan = _plan(monkeypatch, tmp_path, [_Task("run-a", "killed")])
        assert plan.open_runs == 0
        assert plan.closures == []

    def test_dry_run_writes_nothing(self, monkeypatch, tmp_path):
        _run_dir(tmp_path, "run-a", "running")
        plan = _plan(monkeypatch, tmp_path, [_Task("run-a", "killed")])
        assert plan.applied is False
        assert plan.written == 0
        assert len(plan.closures) == 1, "but it still says what it WOULD do"
