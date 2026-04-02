"""The process lifecycle: the guard, the tee, the watcher, the exit account.

Every case here stands in for something the shell got wrong once. The most
expensive was the exit trap reading `$?` as zero on a signal death, which made
`just cancel` record clean completions that were never reconciled.
"""

from __future__ import annotations

import json
import sys

import pytest

from src.shared import leg_log
from src.shared.node import plan as node_plan
from src.shared.node import runner


@pytest.fixture
def paths(tmp_path):
    return runner.NodePaths(
        work=tmp_path / "work", share=tmp_path / "share", code=tmp_path / "code"
    )


@pytest.fixture
def log(paths):
    logger = runner.LegLogger(paths.work / "leg.log", paths.share)
    yield logger
    logger.close()


def _python(*statements: str) -> list[str]:
    return [sys.executable, "-c", "; ".join(statements)]


class TestNodePaths:
    def test_the_defaults_are_the_batch_node_layout(self):
        resolved = runner.NodePaths.from_environment({})
        assert str(resolved.runs) == "/mnt/work/data/runs"
        assert str(resolved.share) == "/mnt/batch/tasks/fsmounts/shared"
        assert str(resolved.archive) == "/mnt/batch/tasks/fsmounts/shared/archive"

    def test_the_mount_root_comes_from_batch(self):
        resolved = runner.NodePaths.from_environment({"AZ_BATCH_NODE_MOUNTS_DIR": "/mnt/fs"})
        assert str(resolved.share) == "/mnt/fs/shared"

    def test_the_code_tree_is_task_owned(self):
        """Unique per task, so concurrent legs on one node cannot share a tree."""
        resolved = runner.NodePaths.from_environment({"CODE_DIR": "/mnt/work/code-task-7"})
        assert str(resolved.code) == "/mnt/work/code-task-7"


class TestRunGuarded:
    def test_a_clean_exit_reports_zero(self, paths, log):
        assert runner.run_guarded(_python("pass"), cwd=paths.work.parent, timeout=5, log=log) == 0

    def test_a_failing_command_reports_its_code(self, paths, log):
        argv = _python("import sys", "sys.exit(3)")
        assert runner.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log) == 3

    @pytest.mark.timeout(15)
    def test_the_deadline_reports_124_not_the_signal_it_sent(self, paths, log):
        """`timeout`'s convention. The guard sends TERM, so the child reports
        143 -- but 143 means "cancelled" to the leg record, and this is a hang."""
        argv = _python("import time", "time.sleep(30)")
        assert runner.run_guarded(argv, cwd=paths.work.parent, timeout=1, log=log) == (
            runner.EXIT_TIMEOUT
        )

    @pytest.mark.timeout(15)
    def test_a_child_killed_from_outside_reports_137(self, paths, log):
        """137 is the OOM killer on a training node. Reporting it as 124 would
        call an OOM a hang AND, being terminal, stop `legs` asking Batch."""
        argv = _python("import os, signal", "os.kill(os.getpid(), signal.SIGKILL)")
        assert runner.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log) == 137

    def test_a_command_that_cannot_start_is_not_an_exception(self, paths, log):
        code = runner.run_guarded(
            ["/nonexistent/binary"], cwd=paths.work.parent, timeout=5, log=log
        )
        assert code == 1
        assert "could not start" in log.path.read_text()

    def test_output_is_tee_d_to_the_leg_log(self, paths, log):
        argv = _python("print('hello from the trainer')")
        runner.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log)
        assert "hello from the trainer" in log.path.read_text()

    def test_stderr_is_captured_too(self, paths, log):
        argv = _python("import sys", "sys.stderr.write('boom\\n')")
        runner.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log)
        assert "boom" in log.path.read_text()

    def test_a_carriage_return_stream_still_reaches_the_log(self, paths, log):
        """tqdm emits `\\r`, not `\\n`. Reading by line would block for minutes
        and publish an empty log through exactly the window worth watching."""
        argv = _python(
            "import sys",
            "sys.stdout.write('50%\\r'); sys.stdout.flush()",
            "import time; time.sleep(0.2)",
        )
        runner.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log)
        assert "50%" in log.path.read_text()

    def test_json_stdout_is_captured_apart_from_the_log(self, paths, log):
        """precompute's stdout is a payload, not a log; its stderr still tees."""
        payload = paths.work / "out.json"
        argv = _python(
            "import sys",
            'sys.stdout.write(\'{"output_dir": "/x"}\')',
            "sys.stderr.write('progress\\n')",
        )
        runner.run_guarded(argv, cwd=paths.work.parent, timeout=5, log=log, stdout_to=payload)
        assert payload.read_text() == '{"output_dir": "/x"}'
        assert "progress" in log.path.read_text()
        assert "output_dir" not in log.path.read_text()


class TestLegLogger:
    def test_publishing_lands_the_log_on_the_share(self, paths, log):
        log("something worth reading later")
        log.publish()
        published = paths.share / "logs" / "leg.log"
        assert "something worth reading later" in published.read_text()

    def test_only_the_tail_is_published(self, paths, log, monkeypatch):
        """A multi-hour tqdm stream is mostly progress-bar repaints that cost
        more to copy than they inform."""
        monkeypatch.setattr(runner, "PUBLISHED_LOG_BYTES", 64)
        log("x" * 500)
        log("the end")
        log.publish()
        published = (paths.share / "logs" / "leg.log").read_bytes()
        assert len(published) == 64
        assert b"the end" in published

    def test_an_unwritable_share_does_not_kill_the_leg(self, paths, log, monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError("share went away")

        monkeypatch.setattr(runner.Path, "mkdir", refuse)
        log.publish()


class TestLadderWatcher:
    def test_it_publishes_when_the_ladder_moves(self, paths, tmp_path, log):
        run_dir = paths.runs / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / "static-10.zarr").mkdir()
        (run_dir / "static-10.zarr" / "chunk").write_text("data")
        (run_dir / "STATIC_CHECKPOINT.json").write_text(
            '{"zarr": "static-10.zarr", "iteration": 10, "retained": []}'
        )

        watcher = runner.LadderWatcher(paths, log, interval=0.01)
        watcher.start()
        try:
            _eventually(lambda: (paths.archive / "run-a" / "static-10.zarr").is_dir())
        finally:
            watcher.stop()
        assert (paths.archive / "run-a" / "static-10.zarr" / "chunk").read_text() == "data"

    def test_stop_joins_the_thread(self, paths, log):
        """Not merely signalled: publishing removes a marker, copies, rewrites
        it, so a second publisher can leave a known-good rung unmarked."""
        paths.runs.mkdir(parents=True)
        watcher = runner.LadderWatcher(paths, log, interval=0.01)
        watcher.start()
        watcher.stop()
        assert not watcher._thread.is_alive()


def _eventually(predicate, attempts: int = 200) -> None:
    import time

    for _ in range(attempts):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")


class TestExitAccounting:
    @pytest.mark.parametrize(
        ("code", "cause"),
        [
            (0, leg_log.CAUSE_COMPLETED),
            (runner.EXIT_TIMEOUT, leg_log.CAUSE_TIMEOUT),
            (143, leg_log.CAUSE_CANCELLED),
            (130, leg_log.CAUSE_CANCELLED),
            (137, leg_log.CAUSE_KILLED),
            (1, leg_log.CAUSE_FAILED),
            (2, leg_log.CAUSE_FAILED),
        ],
    )
    def test_each_death_maps_to_its_own_cause(self, code, cause):
        """A wrong terminal cause is permanent: it suppresses reconciliation,
        so the observer half of the join is lost for good."""
        assert runner._cause(code, None) == cause

    def test_an_explicit_outcome_wins_over_the_code(self):
        """An evaluate leg exits 0 for Batch's retry economics, which is not a
        claim that all 30 rungs scored."""
        assert runner._cause(0, leg_log.CAUSE_PARTIAL) == leg_log.CAUSE_PARTIAL


class TestMain:
    """The wiring `run_leg.sh`'s traps used to carry."""

    @pytest.fixture(autouse=True)
    def _node(self, paths, monkeypatch):
        monkeypatch.setattr(runner.NodePaths, "from_environment", classmethod(lambda cls: paths))
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "leg-1")
        for key, value in {
            "RUN_OP": "train",
            "RUN_CONFIG": "quick_test",
            "RUN_TO": "1000",
            "RUN_ID": "",
            "RUN_SETS_JSON": "[]",
        }.items():
            monkeypatch.setenv(key, value)

    def test_a_signalled_leg_records_cancelled_not_completed(self, paths, monkeypatch):
        """THE defect this port fixes. Bash's EXIT trap reads `$?` as zero when
        killed while blocked on a child -- measured: SIGTERM ran the trap with
        `$? = 0` and exited 143, so a cancelled leg was recorded as clean and
        never reconciled against Batch."""
        monkeypatch.setattr(runner, "_stage", lambda paths, log: 0)
        monkeypatch.setitem(runner.HANDLERS, node_plan.TRAIN, _signalled)

        assert runner.main() == 143
        (row,) = leg_log.read_legs(paths.share)
        assert row["cause"] == leg_log.CAUSE_CANCELLED
        assert row["exit_code"] == 143

    def test_a_leg_that_dies_before_the_sync_still_leaves_a_record(self, paths, monkeypatch):
        """The whole reason the started record is written first: a leg dying
        during dependency install must not be indistinguishable from one that
        never ran."""
        monkeypatch.setattr(runner, "_stage", lambda paths, log: 1)

        assert runner.main() == 1
        (row,) = leg_log.read_legs(paths.share)
        assert row["cause"] == leg_log.CAUSE_FAILED

    def test_a_bad_environment_is_a_message_not_a_traceback(self, paths, monkeypatch):
        monkeypatch.setenv("RUN_TO", "0")
        assert runner.main() == 1
        (row,) = leg_log.read_legs(paths.share)
        assert row["cause"] == leg_log.CAUSE_FAILED
        assert "ABSOLUTE" in (paths.share / "logs" / "leg-1.log").read_text()

    def test_progress_is_published_even_on_a_failure(self, paths, monkeypatch):
        """An operator-cancelled task still leaves its progress on the share."""
        run_dir = paths.runs / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / ".run.json").write_text("{}")
        monkeypatch.setattr(runner, "_stage", lambda paths, log: 0)
        monkeypatch.setitem(runner.HANDLERS, node_plan.TRAIN, lambda *a: (1, None))

        runner.main()
        assert (paths.archive / "run-a" / ".run.json").exists()


def _signalled(plan, paths, log):
    raise runner.Killed(15)


class TestStage:
    def test_the_data_symlink_points_at_the_node_disk(self, paths, monkeypatch):
        """`precompute` writes to <base>/data/, and <base> is the throwaway code
        tree; the symlink is what lands it on the data disk instead."""
        paths.code.mkdir(parents=True)
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        logger = runner.LegLogger(paths.work / "leg.log", paths.share)
        try:
            assert runner._stage(paths, logger) == 0
        finally:
            logger.close()
        assert (paths.code / "data").resolve() == paths.data.resolve()

    def test_a_stale_symlink_is_replaced(self, paths, monkeypatch):
        """A Batch retry reuses the extracted tree."""
        paths.code.mkdir(parents=True)
        (paths.code / "data").symlink_to(paths.work / "somewhere-else")
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        logger = runner.LegLogger(paths.work / "leg.log", paths.share)
        try:
            runner._stage(paths, logger)
        finally:
            logger.close()
        assert (paths.code / "data").resolve() == paths.data.resolve()


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
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        leg = node_plan.LegPlan(op=node_plan.EVALUATE, run_id="run-a")

        assert runner._evaluate(leg, paths, log) == (0, None)
        assert (paths.runs / "run-a" / "static-2000.zarr" / "chunk").exists()

    def test_a_run_with_nothing_published_is_refused(self, paths, log):
        (paths.archive / "run-a").mkdir(parents=True)
        leg = node_plan.LegPlan(op=node_plan.EVALUATE, run_id="run-a")
        assert runner._evaluate(leg, paths, log) == (1, None)
        assert "no published checkpoint to score" in log.path.read_text()

    def test_a_partial_sweep_is_reported_as_partial(self, paths, log, monkeypatch):
        """Exit 0 keeps Batch from retrying 30 rungs to redo one, but it is not
        a claim that all 30 scored."""
        self._published(paths)
        (paths.archive / "run-a" / "static-1000.zarr").mkdir()
        (paths.archive / "run-a" / "static-1000.zarr" / "chunk").write_text("d")
        (paths.archive / "run-a" / ".complete-static-1000.zarr").write_text("")
        codes = iter([0, 1])
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: next(codes))
        leg = node_plan.LegPlan(op=node_plan.EVALUATE, run_id="run-a", eval_rungs=("1000", "2000"))

        assert runner._evaluate(leg, paths, log) == (0, leg_log.CAUSE_PARTIAL)

    def test_a_clean_sweep_of_failures_is_worth_a_retry(self, paths, log, monkeypatch):
        """That is what a transient node fault looks like."""
        self._published(paths)
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 1)
        leg = node_plan.LegPlan(op=node_plan.EVALUATE, run_id="run-a", eval_rungs=("2000",))

        assert runner._evaluate(leg, paths, log) == (1, None)


class TestTheGuardReachesTheWholeTree:
    """`terminate()` reaches only `uv`; the trainer and its workers are
    grandchildren. A deadline that returned 124 while the workers kept running
    is how stale /dev/shm segments outlived a killed leg and took down the
    NEXT one for that run."""

    @pytest.mark.timeout(30)
    def test_the_deadline_kills_the_grandchild_too(self, paths, log, tmp_path, monkeypatch):
        # The real grace is 120s, so the trainer can flush. This child ignores
        # TERM on purpose -- the case that motivated the guard -- so the test
        # would otherwise sit through the whole window.
        monkeypatch.setattr(runner, "GRACE_SECONDS", 2)
        marker = tmp_path / "grandchild-alive"
        # A child that spawns a grandchild and then ignores TERM itself --
        # exactly the shape `uv run python -m trainer` has.
        child = _python(
            "import subprocess, sys, signal, time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            f"g = subprocess.Popen([sys.executable, '-c', "
            f'"import time, pathlib; p = pathlib.Path({str(marker)!r});\\n"'
            f'"[ (p.write_text(str(i)), time.sleep(0.1)) for i in range(200) ]"])',
            "time.sleep(60)",
        )
        assert runner.run_guarded(child, cwd=tmp_path, timeout=1, log=log) == runner.EXIT_TIMEOUT

        import time as _time

        _time.sleep(0.5)
        before = marker.read_text() if marker.exists() else ""
        _time.sleep(0.5)
        after = marker.read_text() if marker.exists() else ""
        assert before == after, "the grandchild outlived the guard's deadline"


class TestPrecompute:
    """Never probed on the pool -- it is a rare, expensive op -- so the guard
    that makes it safe to run in the cloud is only checked here."""

    def _wrote(self, paths, name="ochs_gate_ochs"):
        output = paths.data / "combo_abstraction" / name
        (output / "buckets.npy").parent.mkdir(parents=True, exist_ok=True)
        (output / "buckets.npy").write_text("buckets")
        (paths.work / "precompute.json").parent.mkdir(parents=True, exist_ok=True)
        (paths.work / "precompute.json").write_text(json.dumps({"output_dir": str(output)}))
        return output

    def test_a_fresh_abstraction_is_published(self, paths, log, monkeypatch):
        self._wrote(paths)
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        leg = node_plan.LegPlan(op=node_plan.PRECOMPUTE, config="ochs_gate_ochs")

        assert runner._precompute(leg, paths, log) == (0, None)
        published = paths.share / "combo_abstraction" / "ochs_gate_ochs" / "buckets.npy"
        assert published.read_text() == "buckets"

    def test_republishing_over_an_existing_name_is_refused(self, paths, log, monkeypatch):
        """Bucket ASSIGNMENT is not pinned by card_abstraction_hash, so
        replacing it silently changes which bucket a hand lands in while every
        run trained against the old copy keeps a provenance check that still
        passes. This guard is what makes precompute-in-the-cloud as safe as on
        a laptop."""
        self._wrote(paths)
        existing = paths.share / "combo_abstraction" / "ochs_gate_ochs"
        existing.mkdir(parents=True)
        (existing / "buckets.npy").write_text("THE ORIGINAL")
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        leg = node_plan.LegPlan(op=node_plan.PRECOMPUTE, config="ochs_gate_ochs")

        assert runner._precompute(leg, paths, log) == (1, None)
        assert (existing / "buckets.npy").read_text() == "THE ORIGINAL"
        assert "REFUSING to republish" in log.path.read_text()

    def test_force_publish_overrides_it(self, paths, log, monkeypatch):
        self._wrote(paths)
        existing = paths.share / "combo_abstraction" / "ochs_gate_ochs"
        existing.mkdir(parents=True)
        (existing / "buckets.npy").write_text("THE ORIGINAL")
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        leg = node_plan.LegPlan(
            op=node_plan.PRECOMPUTE, config="ochs_gate_ochs", force_publish=True
        )

        assert runner._precompute(leg, paths, log) == (0, None)
        assert (existing / "buckets.npy").read_text() == "buckets"

    def test_a_failed_build_publishes_nothing(self, paths, log, monkeypatch):
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 2)
        leg = node_plan.LegPlan(op=node_plan.PRECOMPUTE, config="ochs_gate_ochs")

        assert runner._precompute(leg, paths, log) == (2, None)
        assert not (paths.share / "combo_abstraction").exists()

    def test_an_unreadable_payload_is_not_a_traceback(self, paths, log, monkeypatch):
        """The command REPORTS where it wrote; the directory name is never
        re-derived. If that report is missing, guessing would publish the
        wrong thing under a name that can never be corrected."""
        (paths.work).mkdir(parents=True, exist_ok=True)
        (paths.work / "precompute.json").write_text("not json")
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        leg = node_plan.LegPlan(op=node_plan.PRECOMPUTE, config="ochs_gate_ochs")

        assert runner._precompute(leg, paths, log) == (1, None)
        assert "no usable output_dir" in log.path.read_text()


class TestRepairLadder:
    def test_it_fetches_nothing(self, paths, log, monkeypatch):
        """It reads the share IN PLACE. Without the exemption it once fell to a
        catch-all copy and spent 25+ minutes duplicating a 16 GB ladder it then
        ignored."""
        share = paths.archive / "run-a"
        (share / "static-1000.zarr").mkdir(parents=True)
        (share / "static-1000.zarr" / "chunk").write_text("data")
        monkeypatch.setattr(runner, "run_guarded", lambda *a, **k: 0)
        leg = node_plan.LegPlan(op=node_plan.REPAIR_LADDER, run_id="run-a", config="quick_test")

        assert runner._repair_ladder(leg, paths, log) == (0, None)
        assert not (paths.runs / "run-a").exists()

    def test_an_absent_run_is_a_message_not_a_traceback(self, paths, log):
        leg = node_plan.LegPlan(op=node_plan.REPAIR_LADDER, run_id="ghost", config="quick_test")
        assert runner._repair_ladder(leg, paths, log) == (1, None)
        assert "no such run on the share" in log.path.read_text()
