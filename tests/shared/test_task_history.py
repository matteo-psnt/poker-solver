"""The join: the node's account, Batch's, and the one row per attempt they make.

The node's WRITING side is `tests/shared/cloudtask/test_task_log.py`. Everything
here reads, and the fixtures write node records only to have something to read.
"""

from __future__ import annotations

import json

import pytest

from src.shared import records, task_history
from src.shared.cloudtask import task_log


def _node(share, task_id, event, cause=None, **kw):
    return task_log.write_node_record(share, task_id=task_id, event=event, cause=cause, **kw)


class TestNodeRecord:
    def test_started_record_is_not_terminal(self, tmp_path):
        _node(tmp_path, "task-a", "started")
        assert task_history.read_tasks(tmp_path)[0].cause == "unresolved"

    def test_terminal_record_supersedes_started(self, tmp_path):
        _node(tmp_path, "task-a", "started")
        _node(tmp_path, "task-a", "finished", cause="completed", exit_code=0)

        rows = task_history.read_tasks(tmp_path)
        assert len(rows) == 1, "start + exit are one attempt, not two rows"
        assert rows[0].cause == "completed"
        assert rows[0].exit_code == 0
        # Both from the node's own two records.
        assert rows[0].started_at
        assert rows[0].ended_at

    def test_carries_the_run_identity(self, tmp_path):
        _node(
            tmp_path, "task-a", "started", run_id="run-xyz", op="train-static", config="production"
        )
        row = task_history.read_tasks(tmp_path)[0]
        assert (row.run_id, row.op, row.config) == (
            "run-xyz",
            "train-static",
            "production",
        )

    def test_timeout_stays_distinct_from_failure(self, tmp_path):
        """The RUN_TIMEOUT guard is a hang; Batch reports it as plain failure."""
        _node(tmp_path, "hung", "finished", cause="timeout", exit_code=124)
        _node(tmp_path, "crashed", "finished", cause="failed", exit_code=1)

        causes = {r.task_id: r.cause for r in task_history.read_tasks(tmp_path)}
        assert causes == {"hung": "timeout", "crashed": "failed"}


class TestJoin:
    """The case the module exists for: a task killed before its trap could run."""

    def test_observer_explains_a_task_the_node_never_finished(self, tmp_path):
        _node(tmp_path, "task-oom", "started", run_id="run-xyz")
        task_history.write_observed_record(
            tmp_path,
            task_id="task-oom",
            job_id="poker-20260801",
            state="completed",
            result="failure",
            exit_code=137,
            failure={"code": "TaskEnded", "message": "node lost"},
            end_time="2026-08-01T10:00:00Z",
        )

        row = task_history.read_tasks(tmp_path)[0]
        assert row.cause == "failed"
        assert row.cause_source == "batch"
        assert row.run_id == "run-xyz", "the run identity comes from the node half"
        assert row.failure is not None
        assert row.failure["code"] == "TaskEnded"
        assert row.exit_code == 137, (
            "the node's record carries a null exit_code, which must not shadow "
            "the only code that exists for a task killed before its trap ran"
        )

    def test_node_account_wins_when_it_reached_a_terminal_event(self, tmp_path):
        """Batch calls a timed-out task 'failure'; the node knows it was a hang."""
        _node(tmp_path, "task-hang", "finished", cause="timeout", exit_code=124)
        task_history.write_observed_record(
            tmp_path, task_id="task-hang", job_id="j", state="completed", result="failure"
        )

        row = task_history.read_tasks(tmp_path)[0]
        assert row.cause == "timeout"
        assert row.cause_source == "node"

    def test_observer_only_task_still_appears(self, tmp_path):
        """A task killed before the node wrote anything must not vanish."""
        task_history.write_observed_record(
            tmp_path, task_id="task-ghost", job_id="j", state="completed", result="failure"
        )
        assert task_history.read_tasks(tmp_path)[0].task_id == "task-ghost"

    def test_running_task_is_not_called_dead(self, tmp_path):
        task_history.write_observed_record(tmp_path, task_id="t", job_id="j", state="running")
        assert task_history.read_tasks(tmp_path)[0].cause == "running"


class TestBatchRetry:
    """A retry reuses the task id; the failed attempt must survive it."""

    def test_a_retry_does_not_erase_the_failed_attempt(self, tmp_path):
        _node(tmp_path, "task-1", "started")
        _node(tmp_path, "task-1", "finished", cause="killed", exit_code=137)
        _node(tmp_path, "task-1", "started")  # Batch retries with the SAME id
        _node(tmp_path, "task-1", "finished", cause="completed", exit_code=0)

        rows = sorted(task_history.read_tasks(tmp_path), key=lambda r: r.attempt)
        assert [r.attempt for r in rows] == [1, 2]
        assert [r.cause for r in rows] == ["killed", "completed"], (
            "the OOM that caused the retry is the whole point of the record"
        )

    def test_the_observer_explains_only_the_latest_attempt(self, tmp_path):
        """Batch's executionInfo describes no earlier attempt, so it must not
        be attached to one -- that would explain the wrong death."""
        _node(tmp_path, "task-1", "started")
        _node(tmp_path, "task-1", "finished", cause="killed", exit_code=137)
        _node(tmp_path, "task-1", "started")
        task_history.write_observed_record(tmp_path, task_id="task-1", job_id="j", state="running")

        rows = {r.attempt: r for r in task_history.read_tasks(tmp_path)}
        assert rows[1].cause == "killed"
        assert rows[2].cause == "running"

    def test_unresolved_reports_each_task_once(self, tmp_path):
        _node(tmp_path, "task-1", "started")
        _node(tmp_path, "task-1", "finished", cause="failed", exit_code=1)
        _node(tmp_path, "task-1", "started")

        assert task_history.unresolved_task_ids(tmp_path) == ["task-1"]


class TestTornTerminalWrite:
    def test_a_torn_exit_record_leaves_the_task_unresolved_not_absent(self, tmp_path):
        """write_text truncates, so the SIGKILL window can tear the exit file.

        The task must still appear -- and as unresolved, so reconciliation asks
        Batch. Vanishing would be worse than never having written anything.
        """
        _node(tmp_path, "task-torn", "started")
        (task_log.tasks_dir(tmp_path) / "task-torn.1.exit.json").write_text('{"task_id": "task')

        rows = task_history.read_tasks(tmp_path)
        assert [r.task_id for r in rows] == ["task-torn"]
        assert rows[0].cause == "unresolved"
        assert task_history.unresolved_task_ids(tmp_path) == ["task-torn"]


class TestCauseVocabulary:
    """A wrong terminal cause is worse than none: it suppresses reconciliation.

    The vocabulary is the WRITER's (`task_log`); which of them are final is this
    module's, and this is where the two have to agree."""

    @pytest.mark.parametrize(
        "cause",
        [
            task_log.CAUSE_COMPLETED,
            task_log.CAUSE_FAILED,
            task_log.CAUSE_TIMEOUT,
            task_log.CAUSE_KILLED,
            task_log.CAUSE_CANCELLED,
            task_log.CAUSE_PARTIAL,
        ],
    )
    def test_every_node_cause_is_terminal(self, tmp_path, cause):
        _node(tmp_path, "t", "started")
        _node(tmp_path, "t", "finished", cause=cause)
        assert task_history.read_tasks(tmp_path)[0].cause == cause
        assert task_history.unresolved_task_ids(tmp_path) == []

    def test_an_oom_is_not_recorded_as_a_hang(self, tmp_path):
        """137 is SIGKILL from outside; `timeout` returns 124 even after its
        own --kill-after fires, so 137 never means the guard."""
        _node(tmp_path, "oom", "finished", cause=task_log.CAUSE_KILLED, exit_code=137)
        _node(tmp_path, "hang", "finished", cause=task_log.CAUSE_TIMEOUT, exit_code=124)

        causes = {r.task_id: r.cause for r in task_history.read_tasks(tmp_path)}
        assert causes == {"oom": "killed", "hang": "timeout"}

    def test_a_cancelled_task_is_not_a_clean_completion(self, tmp_path):
        _node(tmp_path, "c", "finished", cause=task_log.CAUSE_CANCELLED, exit_code=143)
        assert task_history.read_tasks(tmp_path)[0].cause == "cancelled"


class TestReconcile:
    def test_only_unresolved_tasks_are_written(self, tmp_path):
        _node(tmp_path, "done", "finished", cause="completed", exit_code=0)
        _node(tmp_path, "vanished", "started")

        explained = task_history.reconcile(
            tmp_path,
            [
                {"task": "done", "state": "completed", "result": "success"},
                {"task": "vanished", "state": "completed", "result": "failure"},
            ],
        )

        assert explained == ["vanished"]
        assert not (task_log.tasks_dir(tmp_path) / "done.observed.json").exists(), (
            "a task that reported its own exit needs no external explanation"
        )

    def test_unknown_tasks_are_ignored(self, tmp_path):
        _node(tmp_path, "mine", "started")
        assert (
            task_history.reconcile(tmp_path, [{"task": "someone-elses", "state": "completed"}])
            == []
        )

    def test_an_explained_task_reads_back_as_an_outcome_not_a_state_string(self, tmp_path):
        """The whole join is worthless if the cause column says
        `batchtaskstate.completed`, so the shape reconcile consumes is pinned to
        the shape `batch.list_jobs_with_tasks` produces."""
        _node(tmp_path, "vanished", "started")
        task_history.reconcile(
            tmp_path,
            [{"task": "vanished", "job": "poker-1", "state": "completed", "result": "failure"}],
        )
        row = next(r for r in task_history.read_tasks(tmp_path) if r.task_id == "vanished")
        assert row.cause == task_log.CAUSE_FAILED


class TestAnObservationIsOnlyWrittenWhenItSaysSomethingNew:
    """`observed_at` is stamped on every read, so an unchanged observation was
    always a fresh document -- and a task that can never resolve (one Batch
    still calls `running`) was re-written and re-uploaded on every single read.
    Measured on the console: six such records, 14.1s of serial share writes,
    per poll, forever, carrying no information.
    """

    def test_repeating_an_observation_reports_nothing_to_publish(self, tmp_path):
        _node(tmp_path, "running-forever", "started")
        observation = [{"task": "running-forever", "job": "poker-1", "state": "running"}]

        assert task_history.reconcile(tmp_path, observation) == ["running-forever"]
        assert task_history.reconcile(tmp_path, observation) == []
        assert task_history.reconcile(tmp_path, observation) == []

    def test_the_stored_record_is_left_alone(self, tmp_path):
        _node(tmp_path, "running-forever", "started")
        observation = [{"task": "running-forever", "job": "poker-1", "state": "running"}]
        task_history.reconcile(tmp_path, observation)
        path = task_log.tasks_dir(tmp_path) / "running-forever.observed.json"
        before = path.read_text()

        task_history.reconcile(tmp_path, observation)

        assert path.read_text() == before

    def test_two_readers_of_one_shared_tree_publish_it_once(self, tmp_path):
        """`/api/tasks` and `/api/cost` are separate cache keys answering the
        same page, so they run at once and are handed ONE legs tree. Two of them
        writing `<task>.observed.json` to a share with no atomic rename breaks
        the one-writer-per-file rule that makes writing there safe at all.
        """
        _node(tmp_path, "vanished", "started")
        observation = [{"task": "vanished", "job": "j", "state": "completed", "result": "failure"}]

        first = task_history.reconcile(tmp_path, observation)
        second = task_history.reconcile(tmp_path, observation)

        assert (first, second) == (["vanished"], []), "both readers published the same record"

    def test_news_is_still_written(self, tmp_path):
        """The property this must not cost: a task that has since FINISHED is a
        different observation, and losing it would leave a death unexplained."""
        _node(tmp_path, "vanished", "started")
        task_history.reconcile(tmp_path, [{"task": "vanished", "job": "j", "state": "running"}])

        explained = task_history.reconcile(
            tmp_path,
            [{"task": "vanished", "job": "j", "state": "completed", "result": "failure"}],
        )

        assert explained == ["vanished"]
        row = next(r for r in task_history.read_tasks(tmp_path) if r.task_id == "vanished")
        assert row.cause == task_log.CAUSE_FAILED


class TestRobustness:
    def test_a_half_written_record_does_not_break_the_listing(self, tmp_path):
        """Truncated files are the expected residue of the kills this explains."""
        _node(tmp_path, "good", "finished", cause="completed", exit_code=0)
        (task_log.tasks_dir(tmp_path) / "torn.1.exit.json").write_text('{"task_id": "torn"')

        rows = task_history.read_tasks(tmp_path)
        assert [r.task_id for r in rows] == ["good"]

    def test_missing_directory_reads_as_empty(self, tmp_path):
        assert task_history.read_tasks(tmp_path / "nothing-here") == []


class TestWhatATaskDid:
    """`target_iteration` is RUN_TO, which an evaluate task never sets.

    38 evaluate tasks on the share therefore recorded a target of `0`, and their
    rung and board seed were written down nowhere — while the eval documents
    they produced carry no task reference to join back on. This is what stops
    the next set going the same way.
    """

    def test_an_evaluation_records_the_rung_and_the_seed_it_scored(self, tmp_path):
        task_log.write_node_record(
            tmp_path,
            task_id="t1",
            event=task_log.EVENT_STARTED,
            run_id="run-production-025433-1095",
            op="evaluate",
            eval_at="150000000",
            eval_flags=("--br-flops", "4", "--br-board-seed", "7"),
        )
        (row,) = task_history.read_tasks(tmp_path)
        assert row.eval_at == "150000000"
        assert row.what == "evaluate @150M seed7"

    def test_three_seeds_on_one_checkpoint_are_now_distinguishable(self, tmp_path):
        """The exact case that had to be kept in a scratchpad file."""
        for index, seed in enumerate(("7", "13", "29")):
            task_log.write_node_record(
                tmp_path,
                task_id=f"t{index}",
                event=task_log.EVENT_STARTED,
                run_id="run-production-025433-1095",
                op="evaluate",
                eval_at="150000000",
                eval_flags=("--br-board-seed", seed),
            )
        assert len({row.what for row in task_history.read_tasks(tmp_path)}) == 3

    def test_a_training_task_says_what_it_was_aiming_at(self, tmp_path):
        task_log.write_node_record(
            tmp_path,
            task_id="t1",
            event=task_log.EVENT_STARTED,
            op="train",
            target_iteration="5000000",
        )
        (row,) = task_history.read_tasks(tmp_path)
        assert row.what == "train ->5M"

    def test_a_task_from_before_these_fields_degrades_to_its_op(self, tmp_path):
        """Honest: those records genuinely hold nothing more to show."""
        task_log.write_node_record(
            tmp_path,
            task_id="t1",
            event=task_log.EVENT_STARTED,
            op="evaluate",
            target_iteration="0",
        )
        (row,) = task_history.read_tasks(tmp_path)
        assert row.what == "evaluate"


class TestWhichCodeRan:
    """The record has to distinguish two arms of the same experiment.

    Work here runs in several git worktrees at once, and a worktree carries its
    change UNCOMMITTED for as long as it is being iterated on — so the arms
    routinely share a commit and a dirty bit, and `c13dcb7-dirty` has described
    four different programs. The snapshot id names the actual bytes and had been
    travelling to the node as a fetch instruction, recorded nowhere.

    How it is SPELLED for a reader is the terminal's business; see
    `tests/interfaces/commands/test_tasks_table.py`.
    """

    def test_the_snapshot_that_ran_is_recorded(self, tmp_path):
        task_log.write_node_record(
            tmp_path,
            task_id="t1",
            event=task_log.EVENT_STARTED,
            op="train",
            code_snapshot="code-20260805_111229",
            git_commit="c13dcb7aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            git_dirty="1",
            git_branch="worktree-hybrid-kernels",
        )
        (row,) = task_history.read_tasks(tmp_path)
        assert row.code_snapshot == "code-20260805_111229"
        assert row.git_branch == "worktree-hybrid-kernels"

    def test_two_worktrees_on_one_dirty_commit_are_distinguishable(self, tmp_path):
        """The case the commit alone cannot answer, which is the normal one."""
        for index, branch in enumerate(("worktree-hybrid-kernels", "worktree-vector-cfr")):
            task_log.write_node_record(
                tmp_path,
                task_id=f"t{index}",
                event=task_log.EVENT_STARTED,
                op="train",
                code_snapshot=f"code-2026080{index}_000000",
                git_commit="c13dcb7aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                git_dirty="1",
                git_branch=branch,
            )
        rows = task_history.read_tasks(tmp_path)
        assert len({row.git_commit for row in rows}) == 1, "same commit — that is the premise"
        assert len({row.git_branch for row in rows}) == 2
        assert len({row.code_snapshot for row in rows}) == 2

    def test_provenance_survives_a_task_that_died_before_finishing(self, tmp_path):
        """The start record carries it, and that is the only record such a task has."""
        task_log.write_node_record(
            tmp_path,
            task_id="t1",
            event=task_log.EVENT_STARTED,
            op="train",
            code_snapshot="code-20260805_111229",
            git_branch="worktree-vector-cfr",
        )
        (row,) = task_history.read_tasks(tmp_path)
        assert row.cause == "unresolved", "died before its exit record — the case under test"
        assert row.code_snapshot == "code-20260805_111229"


class TestBundlesAreJustAnotherContainer:
    """A bundled document must join EXACTLY as the loose file it replaced.

    The whole compaction rests on this: `legs/` is 375 tiny files and reading it
    costs a round trip each, but the join over them (attempt numbering, which
    record explains a death, whose view wins) is subtle enough that the bundle
    stores documents verbatim rather than digested.
    """

    def _bundle(self, tmp_path, names):
        """What `compact-legs` does, minus the share.

        Including reading back the bundle already at this name and passing it
        as `previous` — which is the whole of what stops a second round from
        overwriting the first. A helper that skipped it would make the
        regression below pass for the wrong reason.
        """
        directory = task_log.tasks_dir(tmp_path)
        documents = task_log.read_documents(directory)
        path = directory / f"test{task_log.BUNDLE_SUFFIX}"
        payload = task_history.bundle_document(
            {n: documents[n] for n in names},
            previous=records.read_snapshot(path) if path.exists() else None,
        )
        records.write_snapshot(path, payload, records.REGISTRY[f"legs/*{task_log.BUNDLE_SUFFIX}"])
        for name in names:
            (directory / name).unlink()

    def test_bundling_every_sealed_document_changes_no_row(self, tmp_path):
        _node(tmp_path, "a", "started")
        _node(tmp_path, "a", "finished", cause="completed", exit_code=0)
        _node(tmp_path, "b", "started")
        _node(tmp_path, "b", "finished", cause=task_log.CAUSE_KILLED, exit_code=137)
        before = task_history.read_tasks(tmp_path)

        _, names = task_history.compactable(task_log.tasks_dir(tmp_path))
        assert names, "nothing was judged compactable"
        self._bundle(tmp_path, names)

        assert task_history.read_tasks(tmp_path) == before

    def test_an_unsealed_attempt_is_never_bundled(self, tmp_path):
        """Its `.observed.json` reconciliation writes to a FILENAME, so bundling
        the other half would strand it."""
        _node(tmp_path, "sealed", "started")
        _node(tmp_path, "sealed", "finished", cause="completed", exit_code=0)
        _node(tmp_path, "still-open", "started")

        _, names = task_history.compactable(task_log.tasks_dir(tmp_path))

        assert all("still-open" not in name for name in names)
        assert any("sealed" in name for name in names)

    def test_a_retry_in_flight_keeps_its_earlier_attempt_loose(self, tmp_path):
        """Attempt 1 died and attempt 2 is running. They share a task id, and
        moving only the sealed half would split one task across two containers."""
        _node(tmp_path, "retried", "started")
        _node(tmp_path, "retried", "finished", cause=task_log.CAUSE_KILLED, exit_code=137)
        _node(tmp_path, "retried", "started")  # Batch retries with the SAME id

        _, names = task_history.compactable(task_log.tasks_dir(tmp_path))

        assert names == []

    def test_a_second_compaction_does_not_drop_the_first_ones_records(self, tmp_path):
        """The data-loss path, and it was the ORDINARY one.

        `--label` defaults to `sealed`, so a second compaction targets the same
        filename as the first, and wanting a second one is simply what happens
        as tasks accumulate. `compactable` offers only LOOSE documents -- right,
        since it is answering which files may be deleted -- so a bundle built
        from its answer alone holds round two and not round one, and writing it
        at the same name drops every task round one absorbed.

        Measured before the fix: task `a` vanished from the join entirely.
        """
        _node(tmp_path, "a", "started")
        _node(tmp_path, "a", "finished", cause="completed", exit_code=0)
        _, first = task_history.compactable(task_log.tasks_dir(tmp_path))
        self._bundle(tmp_path, first)

        _node(tmp_path, "b", "started")
        _node(tmp_path, "b", "finished", cause="completed", exit_code=0)
        _, second = task_history.compactable(task_log.tasks_dir(tmp_path))
        assert second, "round two found nothing to bundle — the test proves nothing"
        self._bundle(tmp_path, second)

        assert {row.task_id for row in task_history.read_tasks(tmp_path)} == {"a", "b"}

    def test_the_bundle_records_that_a_compaction_happened(self, tmp_path):
        """Nothing else does.

        It can remove hundreds of files from the share — the only copy of the
        task record — and the sole trace used to be a backup directory on
        whichever laptop ran it.
        """
        _node(tmp_path, "a", "started")
        _node(tmp_path, "a", "finished", cause="completed", exit_code=0)
        documents = task_log.read_documents(task_log.tasks_dir(tmp_path))

        bundle = task_history.bundle_document(
            documents, compaction={"at": "2026-08-11T00:00:00+00:00", "backup": "/b"}
        )

        assert bundle["compactions"] == [{"at": "2026-08-11T00:00:00+00:00", "backup": "/b"}]

    def test_each_round_appends_its_own_entry(self, tmp_path):
        """A bundle carries its history, not just its most recent rewrite."""
        first = task_history.bundle_document({}, compaction={"at": "one"})
        second = task_history.bundle_document({}, previous=first, compaction={"at": "two"})

        assert [entry["at"] for entry in second["compactions"]] == ["one", "two"]

    def test_provenance_never_reaches_the_join(self, tmp_path):
        """It sits beside `records`, not inside it.

        `read_documents` reads only that key, and a compaction entry is not a
        leg document — one that leaked in would be a row with no task id.
        """
        _node(tmp_path, "a", "started")
        _node(tmp_path, "a", "finished", cause="completed", exit_code=0)
        before = task_history.read_tasks(tmp_path)

        directory = task_log.tasks_dir(tmp_path)
        _, names = task_history.compactable(directory)
        documents = task_log.read_documents(directory)
        records.write_snapshot(
            directory / f"test{task_log.BUNDLE_SUFFIX}",
            task_history.bundle_document(
                {name: documents[name] for name in names},
                compaction={"at": "now", "host": "laptop", "bundled": len(names)},
            ),
            records.REGISTRY[f"legs/*{task_log.BUNDLE_SUFFIX}"],
        )
        for name in names:
            (directory / name).unlink()

        assert task_history.read_tasks(tmp_path) == before

    def test_a_loose_file_wins_over_a_bundled_copy(self, tmp_path):
        """A bundle is a snapshot of the past; a loose file is what a node most
        recently wrote."""
        _node(tmp_path, "a", "started")
        _node(tmp_path, "a", "finished", cause="completed", exit_code=0)
        _, names = task_history.compactable(task_log.tasks_dir(tmp_path))
        self._bundle(tmp_path, names)
        # The node writes again, in the shape it always does.
        _node(tmp_path, "a", "finished", cause=task_log.CAUSE_TIMEOUT, exit_code=124)

        causes = {row.cause for row in task_history.read_tasks(tmp_path)}
        assert task_log.CAUSE_TIMEOUT in causes


class TestOneMalformedDocumentCannotTakeDownEveryReader:
    """`read_tasks` builds models now, and `shared` must stay tolerant.

    These are untyped JSON off an SMB share, written by a wrapper that may be an
    older version or killed halfway through a write. `read_tasks` feeds
    `tasks`, `cost`, `runinfo` and -- through `unresolved_tasks` -- `reconcile`,
    so one strict field would take all four down together. `kinds.Progress`
    reads the same bytes and tolerates both of these; the model must agree, or
    two readers of one record disagree about whether it is readable.
    """

    def test_a_progress_record_with_no_unit_still_reads(self, tmp_path):
        _node(tmp_path, "task-a", "started")
        (task_log.tasks_dir(tmp_path) / "task-a.progress.json").write_text(
            json.dumps({"task_id": "task-a", "progress": {"done": 5, "total": 10}})
        )
        rows = task_history.read_tasks(tmp_path)
        assert rows[0].progress is not None
        assert rows[0].progress.unit == ""

    def test_a_null_unit_is_a_blank_one(self, tmp_path):
        """The wrapper writes the key with no value for a kind that cannot name
        its unit, so null is a shape that occurs rather than a broken record."""
        _node(tmp_path, "task-b", "started")
        (task_log.tasks_dir(tmp_path) / "task-b.progress.json").write_text(
            json.dumps({"task_id": "task-b", "progress": {"done": 1, "total": 2, "unit": None}})
        )
        assert task_history.read_tasks(tmp_path)[0].progress.unit == ""
