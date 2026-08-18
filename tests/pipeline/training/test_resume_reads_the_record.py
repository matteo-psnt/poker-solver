"""A resume folds the run's events, and the file is no longer the only place.

This is what `run.jsonl` was still needed for. `RunTracker.load` reads it to
decide whether a task may continue -- the config it was trained under, the
abstraction it is pinned to, the kernel. Without those a resume mints fresh
metadata over a live ladder and trains from zero, which is the failure `opened`
calls unrecoverable.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core.actions.action_model import ActionModel
from src.pipeline.training.run_tracker import RunTracker
from src.pipeline.training.run_tracker.metadata import RunMetadata
from src.shared import run_events
from src.shared.config import Config
from tests.memory_record import MemoryRecord


class _Source:
    """A `RecordSource` holding one run's events AS THE DATABASE STORES THEM.

    `event` is a column there and a key in the file, and the fold looks it up by
    key. A test fed events straight off the file cannot see the difference --
    this one strips and restores it the way the adapter does, because the first
    version did not and a resume died on a node looking for `created`.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        stored = [{k: v for k, v in e.items() if k != "event"} for e in events]
        kinds = [e.get("event") for e in events]
        self._events = [{**body, "event": kind} for body, kind in zip(stored, kinds, strict=True)]
        self.asked: list[str] = []

    def events(self, run_id: str) -> list[Any]:
        self.asked.append(run_id)
        return self._events


def _record_at(where) -> list[dict[str, Any]]:
    """A real run record, written the way a run writes one -- to the SINK.

    Nothing lands on disk: this is what a run created after the flip leaves
    behind, which is exactly the input the source-first path has to handle.
    """
    config = Config.default()
    record = MemoryRecord()
    tracker = RunTracker(
        run_dir=where,
        config_name="test",
        config=config,
        action_config_hash=ActionModel(config).get_config_hash(),
        sink=record,
        source=record,
    )
    tracker.mark_completed()
    return [dict(event) for event in record.events(tracker.run_id)]


def _legacy_record_at(where) -> list[dict[str, Any]]:
    """The same record as a FILE, the way every run written before the flip
    left one. The fallback tests need a real `run.jsonl`; the tracker no longer
    writes one, so these events are appended directly."""
    where.mkdir(parents=True, exist_ok=True)
    for event in _record_at(where.parent / f"{where.name}-source"):
        run_events.append(where, event.pop("event"), **event)
    return run_events.read(where)


class TestAResumeCanFoldFromTheDatabase:
    def test_it_asks_the_source_before_the_file(self, tmp_path):
        source = _Source(_record_at(tmp_path / "run-a"))
        # A directory with NO run record at all -- only the source has it.
        loaded = RunMetadata.load(tmp_path / "run-elsewhere", source)
        assert source.asked == ["run-elsewhere"]
        assert loaded.config_name == "test"

    def test_a_run_the_source_does_not_know_falls_back_to_the_file(self, tmp_path):
        """A run that predates the database resumes exactly as it always did."""
        _legacy_record_at(tmp_path / "run-b")
        loaded = RunMetadata.load(tmp_path / "run-b", _Source([]))
        assert loaded.config_name == "test"

    def test_the_tracker_finds_a_run_that_exists_only_in_the_source(self, tmp_path):
        """`_has_run_record` decided from the FILESYSTEM. Once the log stops
        being published, a run that exists would read as one that does not --
        and a resume that believes that trains from zero over a live ladder."""
        source = _Source(_record_at(tmp_path / "run-c"))
        tracker = RunTracker.load(tmp_path / "run-nowhere-on-disk", source)
        assert tracker.metadata.config_name == "test"

    def test_no_source_and_no_file_still_refuses(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RunTracker.load(tmp_path / "run-missing")

    def test_the_fold_is_the_same_one_either_way(self, tmp_path):
        """The stored body IS the line the log holds, so the metadata a resume
        gets from the database is the metadata it got from the file."""
        events = _legacy_record_at(tmp_path / "run-d")
        from_file = RunMetadata.load(tmp_path / "run-d")
        from_db = RunMetadata.load(tmp_path / "run-d", _Source(events))
        assert from_db.to_dict() == from_file.to_dict()
