"""The run digest, and the gaps it is really there to state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.pipeline.services import run_digest
from src.pipeline.training.run_tracker import RunMetadata, RunTracker
from src.shared import run_events
from src.shared.config import Config
from tests.legacy_legs import write_leg

if TYPE_CHECKING:
    from pathlib import Path


def _run(
    tmp_path, record, *, run_id="run-a", dirty=False, status="completed", abstraction="ae5a"
) -> Path:
    run_dir = tmp_path / run_id
    tracker = RunTracker(
        run_dir=run_dir,
        config_name="quick_test",
        config=Config.default(),
        action_config_hash="abc123",
        card_abstraction_hash=abstraction,
        sink=record,
        source=record,
    )
    # Set BEFORE initialize(): git state is captured in the `created` event, and
    # the suite runs from whatever tree the developer has, clean or not.
    tracker.metadata.git_dirty = dirty
    tracker.initialize()
    tracker.update(iterations=2000, runtime_seconds=10.0, num_infosets=100, storage_capacity=1000)
    if status == "completed":
        tracker.mark_completed()
    elif status != "running":
        record.closed(run_id, status, {"status": status})
    return run_dir


def _progress(run_dir, record, iteration, coverage):
    record.emit(
        run_dir.name,
        run_events.CHECKPOINT,
        {
            "ts": "2026-08-02T00:00:00+00:00",
            "iteration": iteration,
            "coverage": coverage,
            "mean_visits_per_touched": 2.0,
        },
    )


def _digest(run_dir, tmp_path, record, **kw):
    return run_digest(run_dir, ledger_path=tmp_path / "ledger.jsonl", record_source=record, **kw)


class TestIdentity:
    def test_carries_provenance_and_lineage(self, tmp_path, record):
        run_dir = _run(tmp_path, record)
        digest = _digest(run_dir, tmp_path, record)
        assert digest.run_id == "run-a"
        assert digest.config_name == "quick_test"
        assert digest.card_abstraction_hash == "ae5a"
        assert digest.attempts == 1

    def test_reads_the_progress_history(self, tmp_path, record):
        run_dir = _run(tmp_path, record)
        _progress(run_dir, record, 1000, 0.10)
        _progress(run_dir, record, 2000, 0.20)
        assert [r["iteration"] for r in _digest(run_dir, tmp_path, record).progress] == [1000, 2000]


class TestGaps:
    """Each of these has been an actual source of a wrong reading here."""

    def test_no_progress_history_is_stated(self, tmp_path, record):
        digest = _digest(_run(tmp_path, record), tmp_path, record)
        assert any("no progress history" in g for g in digest.gaps)

    def test_an_unscored_run_is_stated(self, tmp_path, record):
        digest = _digest(_run(tmp_path, record), tmp_path, record)
        assert any("no evaluations recorded" in g for g in digest.gaps)

    def test_a_dirty_tree_is_stated(self, tmp_path, record):
        """A commit does not identify the code when the tree had edits."""
        digest = _digest(_run(tmp_path, record, dirty=True), tmp_path, record)
        assert any("dirty working tree" in g for g in digest.gaps)

    def test_a_missing_abstraction_hash_is_stated(self, tmp_path, record):
        digest = _digest(_run(tmp_path, record, abstraction=None), tmp_path, record)
        assert any("no abstraction hash" in g for g in digest.gaps)

    def test_an_unfinished_run_is_stated(self, tmp_path, record):
        digest = _digest(_run(tmp_path, record, status="running"), tmp_path, record)
        assert any("status is 'running'" in g for g in digest.gaps)

    def test_a_clean_scored_run_has_none_of_them(self, tmp_path, record):
        run_dir = _run(tmp_path, record, dirty=False)
        _progress(run_dir, record, 1000, 0.10)
        digest = _digest(run_dir, tmp_path, record)
        assert not any("no progress history" in g for g in digest.gaps)
        assert not any("dirty working tree" in g for g in digest.gaps)
        assert not any("status is" in g for g in digest.gaps)


class TestTasks:
    def test_tasks_are_omitted_without_a_directory(self, tmp_path, record):
        """A purely local run has none, and asking for them must not fail."""
        assert _digest(_run(tmp_path, record), tmp_path, record).tasks == []

    def test_only_this_run_s_tasks_are_joined(self, tmp_path, record):
        """The share holds every run's tasks; a digest must not borrow another's."""
        share = tmp_path / "share"
        for task, run_id in (("t-1", "run-a"), ("t-2", "run-other")):
            write_leg(share, task_id=task, event="started", run_id=run_id)
            write_leg(share, task_id=task, event="finished", run_id=run_id, cause="completed")

        digest = _digest(_run(tmp_path, record), tmp_path, record, tasks_dir=share)
        assert [task.task_id for task in digest.tasks] == ["t-1"]

    def test_an_unresolved_task_becomes_a_gap(self, tmp_path, record):
        share = tmp_path / "share"
        write_leg(share, task_id="t-1", event="started", run_id="run-a")

        digest = _digest(_run(tmp_path, record), tmp_path, record, tasks_dir=share)
        assert any("no terminal record" in g for g in digest.gaps)

    def test_a_finished_task_is_not_a_gap(self, tmp_path, record):
        share = tmp_path / "share"
        write_leg(share, task_id="t-1", event="started", run_id="run-a")
        write_leg(share, task_id="t-1", event="finished", run_id="run-a", cause="killed")

        digest = _digest(_run(tmp_path, record), tmp_path, record, tasks_dir=share)
        assert digest.tasks[0].cause == "killed"
        assert not any("no terminal record" in g for g in digest.gaps)


class TestCurveJoin:
    def test_an_absent_ladder_is_not_an_error(self, tmp_path, record):
        """A metrics-only run has no manifest left to read."""
        digest = _digest(_run(tmp_path, record), tmp_path, record)
        assert digest.curve.points == []

    def test_a_torn_manifest_still_renders_what_it_can(self, tmp_path, record):
        """A reporting command must not die on the ladder it cannot read."""
        run_dir = _run(tmp_path, record)
        (run_dir / "STATIC_CHECKPOINT.json").write_text('{"iteration": 2000}')
        digest = _digest(run_dir, tmp_path, record)
        assert digest.curve.missing_iterations == []

    def test_unscored_rungs_are_listed(self, tmp_path, record):
        run_dir = _run(tmp_path, record)
        (run_dir / "STATIC_CHECKPOINT.json").write_text(
            json.dumps(
                {
                    "iteration": 2000,
                    "zarr": "static-2000.zarr",
                    "fingerprint": "abc123",
                    "retained": [{"iteration": 1000}, {"iteration": 2000}],
                }
            )
        )
        digest = _digest(run_dir, tmp_path, record)
        assert any("unscored ladder rungs" in g for g in digest.gaps)


class TestPhantomRungs:
    """A run that reads `completed` but cannot be scored: a mid-publish failure
    left the manifest naming a rung the share never received, and the only
    symptom was every score of it dying minutes into a task (gamma3-s103)."""

    def _ladder(self, run_dir: Path) -> None:
        (run_dir / "STATIC_CHECKPOINT.json").write_text(
            json.dumps(
                {
                    "iteration": 2000,
                    "zarr": "static-2000.zarr",
                    "fingerprint": "abc123",
                    "retained": [{"iteration": 1000, "zarr": "static-1000.zarr"}],
                }
            )
        )

    def test_a_rung_the_store_does_not_hold_is_named(self, tmp_path, record):
        """THE MARKER CARRIES THE OBJECT NAME. `pull_metadata` recreates markers
        from the container's listing, so a marker is `.complete-static-N.ckpt.zst`
        while the manifest still spells the rung `static-N.zarr`. This fixture
        used the manifest's spelling on both sides, which is why it stayed green
        while `runinfo` told a run holding three usable rungs that the store
        could supply none of them."""
        run_dir = _run(tmp_path, record)
        self._ladder(run_dir)
        (run_dir / ".complete-static-1000.ckpt.zst").touch()

        gaps = _digest(run_dir, tmp_path, record).gaps

        assert any("static-2000.ckpt.zst" in g and "does not hold" in g for g in gaps), gaps
        assert not any("static-1000" in g for g in gaps), gaps

    def test_a_run_with_no_markers_at_all_says_nothing(self, tmp_path, record):
        """A local run, or a share read that never listed markers -- absence of
        every marker is "no evidence", not "every rung is missing"."""
        run_dir = _run(tmp_path, record)
        self._ladder(run_dir)

        assert not any("cannot supply" in g for g in _digest(run_dir, tmp_path, record).gaps)

    def test_published_rungs_with_no_manifest_are_named(self, tmp_path, record):
        """gamma3-s103's real state: five marked rungs on the share and no
        STATIC_CHECKPOINT.json, so nothing could name a rung to score."""
        run_dir = _run(tmp_path, record)
        (run_dir / ".complete-static-1000.zarr").touch()

        gaps = _digest(run_dir, tmp_path, record).gaps

        assert any("no manifest names them" in g for g in gaps), gaps


class TestMalformedInput:
    def test_a_missing_run_dir_raises_rather_than_reporting_nothing(self, tmp_path, record):
        with pytest.raises((FileNotFoundError, ValueError)):
            _digest(tmp_path / "absent", tmp_path, record)

    def test_a_torn_progress_line_does_not_break_the_digest(self, tmp_path, record):
        run_dir = _run(tmp_path, record)
        _progress(run_dir, record, 1000, 0.10)
        with run_events.log_path(run_dir).open("a") as handle:
            handle.write('{"iteration": 20')
        assert len(_digest(run_dir, tmp_path, record).progress) == 1


def test_metadata_round_trips_through_the_digest(tmp_path, record):
    """The digest reads what RunTracker wrote, not a parallel notion of a run."""
    run_dir = _run(tmp_path, record)
    metadata = RunMetadata.load(run_dir, record)
    digest = run_digest(run_dir, ledger_path=tmp_path / "ledger.jsonl", record_source=record)
    assert (digest.run_id, digest.iterations) == (metadata.run_id, metadata.iterations)
