"""What each kind of task does on the node, and what it refuses to do."""

from __future__ import annotations

import json

from src.shared.cloudtask import kinds, task_log
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import handlers
from src.shared.cloudtask.node import plan as node_plan


class TestEvaluateFetch:
    """`score --run X` with no `--at` means the latest checkpoint."""

    def _published(self, paths, *, marked=True):
        share = paths.archive / "run-a"
        (share / "static-2000.zarr").mkdir(parents=True)
        (share / "static-2000.zarr" / "chunk").write_text("data")
        if marked:
            (share / ".complete-static-2000.zarr").write_text("")
        (share / "STATIC_CHECKPOINT.json").write_text(
            '{"zarr": "static-2000.zarr", "iteration": 2000, "retained": []}'
        )
        return share

    def test_no_rung_fetches_the_manifest_s_current_one(self, paths, log, monkeypatch):
        """The shell had no branch for this and fell to a catch-all that copied
        the WHOLE published directory -- the entire ladder, to score one rung."""
        self._published(paths)
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.EVALUATE, run_id="run-a")

        assert handlers._evaluate(task, paths, log) == (0, None)
        assert (paths.runs / "run-a" / "static-2000.zarr" / "chunk").exists()

    def test_a_run_with_nothing_published_is_refused(self, paths, log):
        (paths.archive / "run-a").mkdir(parents=True)
        task = node_plan.TaskPlan(op=TaskName.EVALUATE, run_id="run-a")
        assert handlers._evaluate(task, paths, log) == (1, None)
        assert "no published checkpoint to score" in log.path.read_text()

    def test_a_partial_sweep_is_reported_as_partial(self, paths, log, monkeypatch):
        """Exit 0 keeps Batch from retrying 30 rungs to redo one, but it is not
        a claim that all 30 scored."""
        self._published(paths)
        (paths.archive / "run-a" / "static-1000.zarr").mkdir()
        (paths.archive / "run-a" / "static-1000.zarr" / "chunk").write_text("d")
        (paths.archive / "run-a" / ".complete-static-1000.zarr").write_text("")
        codes = iter([0, 1])
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: next(codes))
        task = node_plan.TaskPlan(op=TaskName.EVALUATE, run_id="run-a", eval_rungs=("1000", "2000"))

        assert handlers._evaluate(task, paths, log) == (0, task_log.CAUSE_PARTIAL)

    def test_a_clean_sweep_of_failures_is_worth_a_retry(self, paths, log, monkeypatch):
        """That is what a transient node fault looks like."""
        self._published(paths)
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 1)
        task = node_plan.TaskPlan(op=TaskName.EVALUATE, run_id="run-a", eval_rungs=("2000",))

        assert handlers._evaluate(task, paths, log) == (1, None)


class TestPrecompute:
    """Never probed on the pool -- it is a rare, expensive op -- so the guard
    that makes it safe to run in the cloud is only checked here."""

    def _wrote(self, paths, name="production"):
        output = paths.data / "combo_abstraction" / name
        (output / "buckets.npy").parent.mkdir(parents=True, exist_ok=True)
        (output / "buckets.npy").write_text("buckets")
        (paths.work / "precompute.json").parent.mkdir(parents=True, exist_ok=True)
        (paths.work / "precompute.json").write_text(json.dumps({"output_dir": str(output)}))
        return output

    def test_a_fresh_abstraction_is_published(self, paths, log, monkeypatch):
        self._wrote(paths)
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.PRECOMPUTE, config="production")

        assert handlers._precompute(task, paths, log) == (0, None)
        published = paths.share / "combo_abstraction" / "production" / "buckets.npy"
        assert published.read_text() == "buckets"

    def test_republishing_over_an_existing_name_is_refused(self, paths, log, monkeypatch):
        """Bucket ASSIGNMENT is not pinned by card_abstraction_hash, so
        replacing it silently changes which bucket a hand lands in while every
        run trained against the old copy keeps a provenance check that still
        passes. This guard is what makes precompute-in-the-cloud as safe as on
        a laptop."""
        self._wrote(paths)
        existing = paths.share / "combo_abstraction" / "production"
        existing.mkdir(parents=True)
        (existing / "buckets.npy").write_text("THE ORIGINAL")
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.PRECOMPUTE, config="production")

        assert handlers._precompute(task, paths, log) == (1, None)
        assert (existing / "buckets.npy").read_text() == "THE ORIGINAL"
        assert "REFUSING to republish" in log.path.read_text()

    def test_force_publish_overrides_it(self, paths, log, monkeypatch):
        self._wrote(paths)
        existing = paths.share / "combo_abstraction" / "production"
        existing.mkdir(parents=True)
        (existing / "buckets.npy").write_text("THE ORIGINAL")
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.PRECOMPUTE, config="production", force_publish=True)

        assert handlers._precompute(task, paths, log) == (0, None)
        assert (existing / "buckets.npy").read_text() == "buckets"

    def test_a_failed_build_publishes_nothing(self, paths, log, monkeypatch):
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 2)
        task = node_plan.TaskPlan(op=TaskName.PRECOMPUTE, config="production")

        assert handlers._precompute(task, paths, log) == (2, None)
        assert not (paths.share / "combo_abstraction").exists()

    def test_an_unreadable_payload_is_not_a_traceback(self, paths, log, monkeypatch):
        """The command REPORTS where it wrote; the directory name is never
        re-derived. If that report is missing, guessing would publish the
        wrong thing under a name that can never be corrected."""
        (paths.work).mkdir(parents=True, exist_ok=True)
        (paths.work / "precompute.json").write_text("not json")
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.PRECOMPUTE, config="production")

        assert handlers._precompute(task, paths, log) == (1, None)
        assert "no usable output_dir" in log.path.read_text()


def test_every_kind_the_submit_path_accepts_has_something_to_run_it():
    """The last way a kind could be half-added.

    `TaskName` and the registry are pinned against each other in
    tests/shared/cloudtask/test_kinds.py, so a kind cannot exist on one side
    only. This is the third place: a kind with no executor validates, builds an
    argv and earns a label, then raises KeyError on the node -- after a snapshot
    upload, a pool spin-up and every Batch retry. Exactly the "found three of
    four" failure the whole abstraction exists to remove.
    """
    missing = sorted(set(kinds.KINDS) - {str(name) for name in handlers.HANDLERS})
    assert not missing, f"these kinds can be submitted but not run: {missing}"
