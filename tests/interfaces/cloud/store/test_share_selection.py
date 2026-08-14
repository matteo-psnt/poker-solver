"""What `fetch` decides to pull -- the pure half, which was wrong twice.

These are the classifiers a green test run could not see. Both defects below
shipped past a full suite, `ty`, and a live fetch that printed a healthy
"Fetched 53 file(s)": the run reported success because the eval records it
silently skipped were already on disk from an earlier era, so `ledger --rebuild`
found them anyway.
"""

import json
from unittest.mock import MagicMock

from src.interfaces.cloud.store import share
from src.shared.records import STATIC_CHECKPOINT


class TestSnapshotClassification:
    """Checkpoint data is identified by a `.zarr` component, never by depth.

    Depth was the original rule, and `<run>/evals/record-1.json` is three
    components deep -- so every eval record was classified as checkpoint data
    and excluded from BOTH modes. Those are exactly the files
    `rebuild_ledger` globs, which is the entire point of the command.
    """

    def test_eval_records_are_not_snapshot_data(self):
        assert not share.is_snapshot_path("run-a/evals/record-1.json")

    def test_nested_eval_records_are_still_not_snapshot_data(self):
        assert not share.is_snapshot_path("run-a/evals/nested/record-1.json")

    def test_zarr_chunks_are_snapshot_data(self):
        assert share.is_snapshot_path("run-a/static-3000.zarr/0.0")

    def test_deeply_nested_zarr_chunks_are_snapshot_data(self):
        assert share.is_snapshot_path("run-a/static-3000.zarr/regrets/0.0")

    def test_top_level_manifest_is_not_snapshot_data(self):
        assert not share.is_snapshot_path(f"run-a/{STATIC_CHECKPOINT}")

    def test_eval_records_survive_a_metadata_fetch(self):
        """The two predicates together are what a metadata fetch applies."""
        record = "run-a/evals/record-1.json"
        assert not share.is_snapshot_path(record)
        assert share.is_metadata(record.rsplit("/", 1)[-1])


class TestManifestMembers:
    """The manifest names what is complete -- all of it, under its real name."""

    @staticmethod
    def _stub(monkeypatch, payload):
        captured = {}

        def _read_text(_service, _share, path):
            captured["path"] = path
            return None if payload is None else json.dumps(payload)

        monkeypatch.setattr(share, "read_text", _read_text)
        return captured

    def test_it_reads_the_static_manifest_not_a_guessed_name(self, monkeypatch):
        """The static tree is the only backend and its manifest is
        STATIC_CHECKPOINT.json. A hardcoded `CHECKPOINT.json` fails OPEN: the
        read returns None, the guard degrades to 'fetch everything', and the
        orphan it exists to exclude is pulled down."""
        captured = self._stub(monkeypatch, {"zarr": "static-1.zarr", "retained": []})

        share.manifest_members(MagicMock(), "share", "archive/run-a")

        assert captured["path"] == f"archive/run-a/{STATIC_CHECKPOINT}"

    def test_every_retained_rung_is_a_member(self, monkeypatch):
        """Naming only the current snapshot trades over-fetching orphans for
        silently dropping the ladder, which leaves `curve` a single point."""
        self._stub(
            monkeypatch,
            {
                "zarr": "static-3000.zarr",
                "retained": [{"iteration": 1000, "zarr": "static-1000.zarr"}],
            },
        )

        members = share.manifest_members(MagicMock(), "share", "archive/run-a")

        assert members == {"static-3000.zarr", "static-1000.zarr"}

    def test_an_absent_manifest_is_reported_as_absent(self, monkeypatch):
        self._stub(monkeypatch, None)
        assert share.manifest_members(MagicMock(), "share", "archive/run-a") is None

    def test_a_manifest_with_no_ladder_still_names_the_current_snapshot(self, monkeypatch):
        self._stub(monkeypatch, {"zarr": "static-3000.zarr"})
        assert share.manifest_members(MagicMock(), "share", "archive/run-a") == {"static-3000.zarr"}


class TestTaskReconcileSeam:
    """`tasks` feeds Batch's task records straight into `task_history.reconcile`.

    The two were written against different shapes -- reconcile against the old
    `az batch task list` JSON, batch.py against its own vocabulary -- so the
    reconcile path raised KeyError on the first unresolved task and, past that,
    silently matched nothing. Nothing caught it because each side was tested
    alone.
    """

    def test_batch_task_records_carry_every_field_reconcile_reads(self):
        from src.interfaces.cloud.tasks import batch

        class _Info:
            code, message, category = "TaskEnded", "boom", None

        class _Exec:
            result, exit_code = "BatchTaskExecutionResult.FAILURE", 137
            start_time = end_time = None
            failure_info = _Info()

        class _Node:
            node_id = "tvmps_x"

        class _Task:
            id, state = "task-1", "BatchTaskState.COMPLETED"
            creation_time = None
            execution_info, node_info = _Exec(), _Node()

        # Dumped, because that is how it crosses into `reconcile`: the record is
        # a model now, and reconcile reads it with `.get()` on the far side of
        # `tasks._observed_by_batch`. The keys are what has to line up.
        record = batch._task_record(_Task()).model_dump()
        for field in ("task", "state", "result", "exit_code", "failure", "start_time", "end_time"):
            assert field in record, f"reconcile reads {field!r}"
        assert record["exit_code"] == 137
        assert record["failure"]["code"] == "TaskEnded"
        # SHORTENED here rather than by the caller: `observed_cause` compares
        # against a bare `failure`, and the raw enum repr matches nothing.
        assert record["result"] == "failure"
        # Not read by reconcile, but by the console: a task WAITING for a node
        # has no start_time, so submission order is the only thing that can put
        # the queue in the order Batch will dispatch it.
        assert "created" in record
