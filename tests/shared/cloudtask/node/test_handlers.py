"""What each kind of task does on the node, and what it refuses to do."""

from __future__ import annotations

import json
from types import SimpleNamespace

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

    def test_the_evaluator_is_told_where_to_report(self, paths, log, monkeypatch):
        """IT NEVER WAS. Only precompute and vector-sweep filled the path in, so
        `--progress-file` never reached an evaluation's command line and the
        branch counter it keeps had nowhere to go: every score fell back to
        counting rungs, which is 1, so the bar read 0% for the whole ten
        minutes."""
        self._published(paths)
        seen: list[list[str]] = []
        monkeypatch.setattr(handlers, "run_guarded", lambda argv, **k: seen.append(argv) or 0)
        task = node_plan.TaskPlan(op=TaskName.EVALUATE, run_id="run-a", eval_rungs=("2000",))

        handlers._evaluate(task, paths, log)

        (argv,) = seen
        where = argv[argv.index("--progress-file") + 1]
        assert where == str(paths.work / "evaluate-progress.json")


class TestTrain:
    def test_the_trainer_is_told_where_to_report(self, paths, log, monkeypatch):
        """The same omission that blanked every evaluation's bar. Without it the
        only thing the wrapper can read is the checkpoint manifest, which lands
        once a million iterations -- minutes to half an hour of a bar that does
        not move, and nothing at all before the first one."""
        seen: list[list[str]] = []
        monkeypatch.setattr(handlers, "run_guarded", lambda argv, **k: seen.append(argv) or 0)
        task = node_plan.TaskPlan(op=TaskName.TRAIN, config="quick_test", to=1000, run_id="run-a")

        handlers._train(task, paths, log)

        (argv,) = seen
        assert argv[argv.index("--progress-file") + 1] == str(paths.work / "train-progress.json")

    def test_the_board_free_trainer_reports_into_the_same_file(self, paths, log, monkeypatch):
        """One task runs on a node and both trainers count the same thing
        against the same kind of target, so they share the file rather than
        keeping two names for one shape."""
        seen: list[list[str]] = []
        monkeypatch.setattr(handlers, "run_guarded", lambda argv, **k: seen.append(argv) or 0)
        task = node_plan.TaskPlan(
            op=TaskName.TRAIN_VECTOR, config="quick_test", to=1000, universe_boards=10
        )

        handlers._train(task, paths, log)

        (argv,) = seen
        assert argv[argv.index("--progress-file") + 1] == str(paths.work / "train-progress.json")

    def test_a_kind_that_reports_no_file_is_not_handed_the_scratch_dir(self, paths, monkeypatch):
        """`paths.work / ""` is the DIRECTORY, and a command asked to write a
        JSON document into one fails on it. Every kind declares a file today, so
        the guard is pinned against a stand-in rather than against whichever
        kind happens not to."""
        monkeypatch.setattr(handlers.kinds, "kind", lambda _op: SimpleNamespace(progress_file=""))
        task = node_plan.TaskPlan(op=TaskName.TRAIN, config="quick_test", to=1000)

        assert handlers._reporting(task, paths).progress_path == ""


class TestAbstractionRefresh:
    """A node that booted before an abstraction was precomputed must still see it.

    `infra/main.tf`'s start task copies the share's abstractions once per BOOT, so
    before this the precompute-then-train ordering could not work on a warm pool:
    the task died in the resolver minutes deep, after `uv sync`. Three abstractions
    sat published-but-unused on the share.
    """

    def _published_abstraction(self, paths, name="buckets-F400T1200R600-rexact-e5c873dc"):
        directory = paths.share / "combo_abstraction" / name
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text('{"config_hash": "e5c873dc4eabc925"}')
        return directory

    def test_training_pulls_an_abstraction_the_node_has_never_seen(self, paths, log, monkeypatch):
        self._published_abstraction(paths)
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.TRAIN, config="quick_test", to=1000, run_id="run-a")

        handlers._train(task, paths, log)

        landed = (
            paths.data
            / "combo_abstraction"
            / "buckets-F400T1200R600-rexact-e5c873dc"
            / "metadata.json"
        )
        assert landed.exists()

    def test_evaluation_pulls_it_too(self, paths, log, monkeypatch):
        """Scoring resolves the abstraction the checkpoint is PINNED to, so the
        same boot order breaks evaluation and not only training."""
        self._published_abstraction(paths)
        share = paths.archive / "run-a"
        (share / "static-2000.zarr").mkdir(parents=True)
        (share / "static-2000.zarr" / "chunk").write_text("data")
        (share / ".complete-static-2000.zarr").write_text("")
        (share / "STATIC_CHECKPOINT.json").write_text(
            '{"zarr": "static-2000.zarr", "iteration": 2000, "retained": []}'
        )
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.EVALUATE, run_id="run-a")

        handlers._evaluate(task, paths, log)

        assert (paths.data / "combo_abstraction" / "buckets-F400T1200R600-rexact-e5c873dc").is_dir()

    def test_a_share_without_abstractions_is_a_warning_not_a_failure(self, paths, log, monkeypatch):
        """The node may already hold what this task needs, and the resolver says
        so precisely if it does not -- refusing here would only move the error."""
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.TRAIN, config="quick_test", to=1000, run_id="run-a")

        assert handlers._train(task, paths, log)[0] == 0
        assert "no" in log.path.read_text()

    def test_an_abstraction_already_on_the_node_is_not_recopied(self, paths, log, monkeypatch):
        """`update=True` is the whole reason the steady-state cost is a directory
        walk rather than 400 MB per task on a busy pool."""
        self._published_abstraction(paths)
        landed = paths.data / "combo_abstraction" / "buckets-F400T1200R600-rexact-e5c873dc"
        landed.mkdir(parents=True)
        (landed / "metadata.json").write_text("newer-on-the-node")
        monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
        task = node_plan.TaskPlan(op=TaskName.TRAIN, config="quick_test", to=1000, run_id="run-a")

        handlers._train(task, paths, log)

        assert (landed / "metadata.json").read_text() == "newer-on-the-node"


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
