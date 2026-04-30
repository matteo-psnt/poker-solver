"""The heartbeat: how far along, published where a reader can see it.

Without this, a multi-hour build is indistinguishable from a hung one -- and the
count a finished task reports is what every later estimate is built from, so
reading the WRONG run's manifest is not a cosmetic bug.
"""

from __future__ import annotations

import json

from src.shared.cloudtask import task_log
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import plan as node_plan
from src.shared.cloudtask.node import progress
from tests.shared.cloudtask.node.conftest import eventually


class TestLadderWatcher:
    def test_it_publishes_when_the_ladder_moves(self, paths, tmp_path, log):
        run_dir = paths.runs / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / "static-10.zarr").mkdir()
        (run_dir / "static-10.zarr" / "chunk").write_text("data")
        (run_dir / "STATIC_CHECKPOINT.json").write_text(
            '{"zarr": "static-10.zarr", "iteration": 10, "retained": []}'
        )

        watcher = progress.LadderWatcher(paths, log, interval=0.01)
        watcher.start()
        try:
            eventually(lambda: (paths.archive / "run-a" / "static-10.zarr").is_dir())
        finally:
            watcher.stop()
        assert (paths.archive / "run-a" / "static-10.zarr" / "chunk").read_text() == "data"

    def test_stop_joins_the_thread(self, paths, log):
        """Not merely signalled: publishing removes a marker, copies, rewrites
        it, so a second publisher can leave a known-good rung unmarked."""
        paths.runs.mkdir(parents=True)
        watcher = progress.LadderWatcher(paths, log, interval=0.01)
        watcher.start()
        watcher.stop()
        assert not watcher._thread.is_alive()


class TestProgressHeartbeat:
    """The wire that makes a completion bar real rather than a display for
    data nobody emits."""

    @staticmethod
    def _plan(op: str = TaskName.TRAIN, to: int = 1000) -> node_plan.TaskPlan:
        return node_plan.TaskPlan(op=op, config="quick_test", to=to)

    def test_a_training_task_publishes_how_far_it_has_got(self, paths, monkeypatch):
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-1")
        progress.publish(paths, self._plan(), {"iteration": 250})

        (record,) = list((paths.share / "legs").glob("*.progress.json"))
        assert json.loads(record.read_text())["progress"] == {
            "done": 250.0,
            "total": 1000.0,
            "unit": "iterations",
        }

    def test_a_kind_with_nothing_to_say_writes_nothing(self, paths, monkeypatch):
        """No bar beats a bar frozen at zero, which reads as a stuck task."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-1")
        progress.publish(paths, self._plan(op=TaskName.PRECOMPUTE), {"iteration": 1})
        assert not list((paths.share / "legs").glob("*.progress.json"))

    def test_it_is_refreshed_in_place_rather_than_accumulated(self, paths, monkeypatch):
        """One file per tick per task would put the share's file COUNT -- the
        thing that makes every read slow -- up by thousands."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-1")
        for iteration in (100, 200, 300):
            progress.publish(paths, self._plan(), {"iteration": iteration})

        (record,) = list((paths.share / "legs").glob("*.progress.json"))
        assert json.loads(record.read_text())["progress"]["done"] == 300.0

    def test_an_unwritable_share_never_kills_the_task(self, paths, monkeypatch):
        """A task must not die because the thing DESCRIBING it could not be
        written. The reader treats a missing sample exactly as it did before
        this existed."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-1")
        monkeypatch.setattr(progress.task_log, "write_progress_record", lambda *a, **k: 1 / 0)
        progress.publish(paths, self._plan(), {"iteration": 1})

    def test_a_finished_task_stops_showing_a_bar(self, paths, monkeypatch):
        """A completed task at 62% is a sample that stopped arriving, not a
        task stuck at 62%."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-1")
        progress.publish(paths, self._plan(), {"iteration": 620})
        task_log.write_node_record(paths.share, task_id="t-1", event=task_log.EVENT_STARTED)
        task_log.write_node_record(
            paths.share, task_id="t-1", event=task_log.EVENT_FINISHED, cause="completed"
        )

        (row,) = task_log.read_tasks(paths.share)
        assert row["cause"] == "completed"
        assert row["progress"] is None


class TestTheWatcherReportsPromptly:
    def test_a_task_shorter_than_the_interval_still_reports(self, paths, log, monkeypatch):
        """The first probe ran 43s against a 120s interval and published nothing,
        so the whole heartbeat went unexercised -- and any short task would have
        shown no bar at all."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-1")
        plan = node_plan.TaskPlan(op=TaskName.TRAIN, config="quick_test", to=1000)
        monkeypatch.setattr(progress, "node_state", lambda *a: {"iteration": 400})

        watcher = progress.LadderWatcher(paths, log, interval=9999, plan=plan)
        watcher.start()
        watcher.stop()

        (record,) = list((paths.share / "legs").glob("*.progress.json"))
        assert json.loads(record.read_text())["progress"]["done"] == 400.0


class TestProgressReadsThisTasksRun:
    """A node is REUSED, and that is what broke this.

    It ran a task to 40k, then one to 80k. `_training_state` returned the first
    run directory it found, which was still the earlier one -- so the bar showed
    the old run's 40,000 against the new task's 80,000 target, a plausible-looking
    50%, while baseline and final reading were the same number and the task
    recorded zero work done.
    """

    @staticmethod
    def _manifest(paths, run_id, iteration):
        run_dir = paths.runs / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "STATIC_CHECKPOINT.json").write_text(
            json.dumps({"iteration": iteration, "zarr": f"static-{iteration}.zarr"})
        )

    def test_an_earlier_runs_manifest_is_not_mistaken_for_this_one(self, paths, monkeypatch):
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-80k")
        self._manifest(paths, "run-train-quick_test-to40k-aaa", 40_000)
        self._manifest(paths, "run-train-quick_test-to80k-zzz", 80_000)
        plan = node_plan.TaskPlan(
            op=TaskName.TRAIN,
            config="quick_test",
            to=80_000,
            run_id="run-train-quick_test-to80k-zzz",
        )

        assert progress._training_state(paths, plan) == {"iteration": 80_000}

    def test_work_done_is_measured_against_this_tasks_own_baseline(self, paths, monkeypatch):
        """A resumed task did the DIFFERENCE, not the counter's value."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "t-1")
        for key in ("RUN_OP", "RUN_CONFIG", "RUN_TO", "RUN_ID"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("RUN_OP", "train")
        monkeypatch.setenv("RUN_CONFIG", "quick_test")
        monkeypatch.setenv("RUN_TO", "150000000")
        monkeypatch.setenv("RUN_ID", "run-a")
        plan = node_plan.parse_environment()

        self._manifest(paths, "run-a", 140_000_000)
        progress.note_baseline(paths, plan)
        self._manifest(paths, "run-a", 150_000_000)

        assert progress.units_done(paths) == 10_000_000
