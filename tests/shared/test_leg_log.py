"""Per-leg outcome records: the node's account, Batch's, and their join."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from src.shared import leg_log

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LEG_LOG_SOURCE = REPO_ROOT / "src" / "shared" / "leg_log.py"


def _node(share, task_id, event, cause=None, **kw):
    return leg_log.write_node_record(share, task_id=task_id, event=event, cause=cause, **kw)


class TestNodeRecord:
    def test_started_record_is_not_terminal(self, tmp_path):
        _node(tmp_path, "task-a", "started")
        assert leg_log.read_legs(tmp_path)[0]["cause"] == "unresolved"

    def test_terminal_record_supersedes_started(self, tmp_path):
        _node(tmp_path, "task-a", "started")
        _node(tmp_path, "task-a", "finished", cause="completed", exit_code=0)

        rows = leg_log.read_legs(tmp_path)
        assert len(rows) == 1, "start + exit are one attempt, not two rows"
        assert rows[0]["cause"] == "completed"
        assert rows[0]["exit_code"] == 0
        # Both from the node's own two records.
        assert rows[0]["started_at"]
        assert rows[0]["ended_at"]

    def test_carries_the_run_identity(self, tmp_path):
        _node(
            tmp_path, "task-a", "started", run_id="run-xyz", op="train-static", config="production"
        )
        row = leg_log.read_legs(tmp_path)[0]
        assert (row["run_id"], row["op"], row["config"]) == (
            "run-xyz",
            "train-static",
            "production",
        )

    def test_timeout_stays_distinct_from_failure(self, tmp_path):
        """The RUN_TIMEOUT guard is a hang; Batch reports it as plain failure."""
        _node(tmp_path, "hung", "finished", cause="timeout", exit_code=124)
        _node(tmp_path, "crashed", "finished", cause="failed", exit_code=1)

        causes = {r["task_id"]: r["cause"] for r in leg_log.read_legs(tmp_path)}
        assert causes == {"hung": "timeout", "crashed": "failed"}


class TestJoin:
    """The case the module exists for: a leg killed before its trap could run."""

    def test_observer_explains_a_leg_the_node_never_finished(self, tmp_path):
        _node(tmp_path, "task-oom", "started", run_id="run-xyz")
        leg_log.write_observed_record(
            tmp_path,
            task_id="task-oom",
            job_id="poker-20260801",
            state="completed",
            result="failure",
            exit_code=137,
            failure={"code": "TaskEnded", "message": "node lost"},
            end_time="2026-08-01T10:00:00Z",
        )

        row = leg_log.read_legs(tmp_path)[0]
        assert row["cause"] == "failed"
        assert row["cause_source"] == "batch"
        assert row["run_id"] == "run-xyz", "the run identity comes from the node half"
        assert row["failure"]["code"] == "TaskEnded"
        assert row["exit_code"] == 137, (
            "the node's record carries a null exit_code, which must not shadow "
            "the only code that exists for a leg killed before its trap ran"
        )

    def test_node_account_wins_when_it_reached_a_terminal_event(self, tmp_path):
        """Batch calls a timed-out leg 'failure'; the node knows it was a hang."""
        _node(tmp_path, "task-hang", "finished", cause="timeout", exit_code=124)
        leg_log.write_observed_record(
            tmp_path, task_id="task-hang", job_id="j", state="completed", result="failure"
        )

        row = leg_log.read_legs(tmp_path)[0]
        assert row["cause"] == "timeout"
        assert row["cause_source"] == "node"

    def test_observer_only_leg_still_appears(self, tmp_path):
        """A task killed before the node wrote anything must not vanish."""
        leg_log.write_observed_record(
            tmp_path, task_id="task-ghost", job_id="j", state="completed", result="failure"
        )
        assert leg_log.read_legs(tmp_path)[0]["task_id"] == "task-ghost"

    def test_running_task_is_not_called_dead(self, tmp_path):
        leg_log.write_observed_record(tmp_path, task_id="t", job_id="j", state="running")
        assert leg_log.read_legs(tmp_path)[0]["cause"] == "running"


class TestBatchRetry:
    """A retry reuses the task id; the failed attempt must survive it."""

    def test_a_retry_does_not_erase_the_failed_attempt(self, tmp_path):
        _node(tmp_path, "leg-1", "started")
        _node(tmp_path, "leg-1", "finished", cause="killed", exit_code=137)
        _node(tmp_path, "leg-1", "started")  # Batch retries with the SAME id
        _node(tmp_path, "leg-1", "finished", cause="completed", exit_code=0)

        rows = sorted(leg_log.read_legs(tmp_path), key=lambda r: r["attempt"])
        assert [r["attempt"] for r in rows] == [1, 2]
        assert [r["cause"] for r in rows] == ["killed", "completed"], (
            "the OOM that caused the retry is the whole point of the record"
        )

    def test_the_observer_explains_only_the_latest_attempt(self, tmp_path):
        """Batch's executionInfo describes no earlier attempt, so it must not
        be attached to one -- that would explain the wrong death."""
        _node(tmp_path, "leg-1", "started")
        _node(tmp_path, "leg-1", "finished", cause="killed", exit_code=137)
        _node(tmp_path, "leg-1", "started")
        leg_log.write_observed_record(tmp_path, task_id="leg-1", job_id="j", state="running")

        rows = {r["attempt"]: r for r in leg_log.read_legs(tmp_path)}
        assert rows[1]["cause"] == "killed"
        assert rows[2]["cause"] == "running"

    def test_unresolved_reports_each_task_once(self, tmp_path):
        _node(tmp_path, "leg-1", "started")
        _node(tmp_path, "leg-1", "finished", cause="failed", exit_code=1)
        _node(tmp_path, "leg-1", "started")

        assert leg_log.unresolved_task_ids(tmp_path) == ["leg-1"]


class TestTornTerminalWrite:
    def test_a_torn_exit_record_leaves_the_leg_unresolved_not_absent(self, tmp_path):
        """write_text truncates, so the SIGKILL window can tear the exit file.

        The leg must still appear -- and as unresolved, so reconciliation asks
        Batch. Vanishing would be worse than never having written anything.
        """
        _node(tmp_path, "leg-torn", "started")
        (leg_log.legs_dir(tmp_path) / "leg-torn.1.exit.json").write_text('{"task_id": "leg')

        rows = leg_log.read_legs(tmp_path)
        assert [r["task_id"] for r in rows] == ["leg-torn"]
        assert rows[0]["cause"] == "unresolved"
        assert leg_log.unresolved_task_ids(tmp_path) == ["leg-torn"]


class TestCauseVocabulary:
    """A wrong terminal cause is worse than none: it suppresses reconciliation."""

    @pytest.mark.parametrize(
        "cause",
        [
            leg_log.CAUSE_COMPLETED,
            leg_log.CAUSE_FAILED,
            leg_log.CAUSE_TIMEOUT,
            leg_log.CAUSE_KILLED,
            leg_log.CAUSE_CANCELLED,
            leg_log.CAUSE_PARTIAL,
        ],
    )
    def test_every_node_cause_is_terminal(self, tmp_path, cause):
        _node(tmp_path, "t", "started")
        _node(tmp_path, "t", "finished", cause=cause)
        assert leg_log.read_legs(tmp_path)[0]["cause"] == cause
        assert leg_log.unresolved_task_ids(tmp_path) == []

    def test_an_oom_is_not_recorded_as_a_hang(self, tmp_path):
        """137 is SIGKILL from outside; `timeout` returns 124 even after its
        own --kill-after fires, so 137 never means the guard."""
        _node(tmp_path, "oom", "finished", cause=leg_log.CAUSE_KILLED, exit_code=137)
        _node(tmp_path, "hang", "finished", cause=leg_log.CAUSE_TIMEOUT, exit_code=124)

        causes = {r["task_id"]: r["cause"] for r in leg_log.read_legs(tmp_path)}
        assert causes == {"oom": "killed", "hang": "timeout"}

    def test_a_cancelled_leg_is_not_a_clean_completion(self, tmp_path):
        _node(tmp_path, "c", "finished", cause=leg_log.CAUSE_CANCELLED, exit_code=143)
        assert leg_log.read_legs(tmp_path)[0]["cause"] == "cancelled"


class TestReconcile:
    def test_only_unresolved_legs_are_written(self, tmp_path):
        _node(tmp_path, "done", "finished", cause="completed", exit_code=0)
        _node(tmp_path, "vanished", "started")

        explained = leg_log.reconcile(
            tmp_path,
            [
                {"id": "done", "state": "completed", "executionInfo": {"result": "success"}},
                {"id": "vanished", "state": "completed", "executionInfo": {"result": "failure"}},
            ],
        )

        assert explained == ["vanished"]
        assert not (leg_log.legs_dir(tmp_path) / "done.observed.json").exists(), (
            "a leg that reported its own exit needs no external explanation"
        )

    def test_unknown_tasks_are_ignored(self, tmp_path):
        _node(tmp_path, "mine", "started")
        assert leg_log.reconcile(tmp_path, [{"id": "someone-elses", "state": "completed"}]) == []


class TestRobustness:
    def test_a_half_written_record_does_not_break_the_listing(self, tmp_path):
        """Truncated files are the expected residue of the kills this explains."""
        _node(tmp_path, "good", "finished", cause="completed", exit_code=0)
        (leg_log.legs_dir(tmp_path) / "torn.1.exit.json").write_text('{"task_id": "torn"')

        rows = leg_log.read_legs(tmp_path)
        assert [r["task_id"] for r in rows] == ["good"]

    def test_missing_directory_reads_as_empty(self, tmp_path):
        assert leg_log.read_legs(tmp_path / "nothing-here") == []

    def test_format_table_handles_no_legs(self, tmp_path):
        assert "no leg records" in leg_log.format_table([])


class TestNodeSideConstraints:
    def test_module_imports_only_the_stdlib(self):
        """run_leg.sh imports this with system python3, before `uv sync`.

        A third-party import here would make the record unavailable for exactly
        the early failures (dependency install, staging) that leave nothing else
        behind.
        """
        source = LEG_LOG_SOURCE.read_text()
        third_party = ("numpy", "pydantic", "zarr", "yaml", "xxhash", "tqdm")
        for name in third_party:
            assert f"import {name}" not in source, f"leg_log must not import {name}"

    def test_importable_without_the_project_environment(self, tmp_path):
        """Proves the node-side contract on a bare interpreter, not just in-suite."""
        script = (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
            "from src.shared.leg_log import write_node_record;"
            f"write_node_record({str(tmp_path)!r}, task_id='t', event='started');"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
        assert (leg_log.legs_dir(tmp_path) / "t.1.start.json").exists()


class TestCli:
    def test_list_prints_a_row_per_leg(self, tmp_path, capsys):
        _node(tmp_path, "task-a", "finished", cause="completed", exit_code=0)
        assert leg_log._main(["list", str(tmp_path)]) == 0
        assert "task-a" in capsys.readouterr().out

    def test_reconcile_reads_task_json(self, tmp_path, capsys):
        _node(tmp_path, "vanished", "started")
        tasks = tmp_path / "tasks.json"
        tasks.write_text(
            json.dumps(
                [{"id": "vanished", "state": "completed", "executionInfo": {"result": "failure"}}]
            )
        )

        assert leg_log._main(["reconcile", str(tmp_path), "--tasks-json", str(tasks)]) == 0
        assert "explained 1" in capsys.readouterr().out
        assert leg_log.read_legs(tmp_path)[0]["cause"] == "failed"

    def test_unparseable_task_json_is_reported_not_raised(self, tmp_path, capsys):
        tasks = tmp_path / "tasks.json"
        tasks.write_text("{not json")
        assert leg_log._main(["reconcile", str(tmp_path), "--tasks-json", str(tasks)]) == 1


@pytest.mark.timeout(30)
class TestRunLegWiring:
    """The shell half: the trap must classify by the leg's real exit status."""

    def _run_leg_source(self) -> str:
        return (REPO_ROOT / "infra" / "run_leg.sh").read_text()

    def test_exit_trap_reads_status_first(self):
        source = self._run_leg_source()
        body = source.split("on_exit() {", 1)[1]
        assert body.lstrip().startswith("LEG_EXIT_CODE=$?"), (
            "anything before `$?` overwrites the status the leg exited with"
        )

    def test_trap_maps_the_timeout_guard_to_its_own_cause(self):
        source = self._run_leg_source()
        assert "124) leg_record finished timeout" in source

    def test_a_started_record_is_written_before_any_work(self):
        source = self._run_leg_source()
        assert "leg_record started" in source
        assert source.index("leg_record started") < source.index("syncing dependencies"), (
            "a leg that dies during dependency sync must still leave a record"
        )


class TestNodeInterpreterCompatibility:
    """The node's python3 is OLDER than this project's.

    infra/main.tf pins `batch.node.ubuntu 22.04`, whose system python3 is 3.10.
    run_leg.sh imports this module with THAT interpreter, before `uv sync`, and
    swallows the result -- so a 3.11+ construct here is not an error anyone
    sees: the leg records simply never appear, and `just legs` reports "no leg
    records", indistinguishable from "no legs ran". The stdlib-only test above
    cannot catch it, because it runs under this project's interpreter.
    """

    # Names that do not exist on 3.10. Extend when the floor moves.
    FORBIDDEN = (
        ("datetime import UTC", "datetime.UTC is 3.11+; use timezone.utc"),
        ("from typing import Self", "typing.Self is 3.11+"),
        ("ExceptionGroup", "ExceptionGroup is 3.11+"),
        ("tomllib", "tomllib is 3.11+"),
        ("itertools.batched", "itertools.batched is 3.12+"),
        ("@override", "typing.override is 3.12+"),
    )

    def test_no_construct_newer_than_the_node_interpreter(self):
        source = LEG_LOG_SOURCE.read_text()
        # Prose in the docstring names the hazard; only code may not use it.
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("#", "*"))
        )
        body = code.split('"""', 2)[-1]
        for needle, why in self.FORBIDDEN:
            assert needle not in body, f"{needle} in leg_log: {why}"

    def test_the_pinned_node_image_is_still_what_this_assumes(self):
        """If the image moves, the floor above moves with it."""
        main_tf = (REPO_ROOT / "infra" / "main.tf").read_text()
        assert "batch.node.ubuntu 22.04" in main_tf, (
            "the node image changed; re-check the system python3 version this "
            "module must import under"
        )
