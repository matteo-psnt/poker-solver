"""The read-only cloud verbs, pinned on the two things a port can silently drop.

Neither of these is about formatting. Each encodes a fact about how Batch
reports state that took a live incident to learn, and a rewrite that loses
either one still looks perfectly healthy.
"""

import pytest

from src.interfaces.cli.headless import build_parser
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands import evaluate, jobs, pool_status, score
from src.shared import task_states


def _job(job_state: str, task_state: str) -> batch.Job:
    return batch.Job(
        job="poker-20260802",
        state=job_state,
        tasks=[
            batch.BatchTask(
                task="t-1",
                job="poker-20260802",
                state=task_state,
                phase=task_states.phase_of(task_state),
                node="tvmps_x",
            )
        ],
    )


class TestLiveJobFilter:
    """A terminated job's tasks stay frozen in whatever state they last held.

    So task state alone cannot answer "is anything running?" -- the account
    accumulates tasks reading `running` under jobs that ended days ago, and
    trusting them reports an idle pool as busy.
    """

    def test_running_task_under_a_completed_job_is_not_live(self):
        stale = _job("BatchJobState.COMPLETED", "BatchTaskState.RUNNING")
        assert not jobs.is_live(stale)

    def test_active_task_under_a_completed_job_is_not_live(self):
        stale = _job("BatchJobState.COMPLETED", "BatchTaskState.ACTIVE")
        assert not jobs.is_live(stale)

    def test_running_task_under_an_active_job_is_live(self):
        assert jobs.is_live(_job("BatchJobState.ACTIVE", "BatchTaskState.RUNNING"))

    def test_finished_task_under_an_active_job_is_not_live(self):
        assert not jobs.is_live(_job("BatchJobState.ACTIVE", "BatchTaskState.COMPLETED"))


class TestResizeErrorValuesAreUnpacked:
    """Batch answers every allocation problem with a generic `AllocationFailed`.

    The actionable cause is a JSON *string* nested inside a resize error's
    values -- printing the value raw gives one unreadable line, and dropping
    the values entirely gives a status command that cannot explain a failure.
    Finding the Gen2-only requirement on `als_v6` depended on reading it.
    """

    def test_escaped_json_value_is_expanded(self, capsys):
        pool_status._print_values({"ResultJson": '{"code": "AllocationFailed"}'})
        out = capsys.readouterr().out
        assert '"code": "AllocationFailed"' in out
        assert "ResultJson" in out

    def test_non_json_value_survives_rather_than_being_swallowed(self, capsys):
        pool_status._print_values({"Surprise": "a cause under an unfamiliar name"})
        assert "a cause under an unfamiliar name" in capsys.readouterr().out

    def test_absent_value_is_skipped(self, capsys):
        pool_status._print_values({"Empty": None})
        assert capsys.readouterr().out == ""


class TestVcpuArithmetic:
    """Pool sizes diverged (D16/D32/D64), so the console reads capacity in
    vCPUs; both inputs come from what pool-status already fetched, and a SKU
    or formula that does not carry the number must yield None, not a guess."""

    def test_d_series_core_count_comes_from_the_sku_name(self):
        assert pool_status.vcpus_per_node("standard_d32als_v6") == 32
        assert pool_status.vcpus_per_node("STANDARD_D64als_v6") == 64

    def test_non_d_series_and_absent_sku_yield_none(self):
        # E16-4ads has 4 usable cores, not 16 -- a D-only parse refuses rather
        # than reading the wrong number off the name.
        assert pool_status.vcpus_per_node("standard_e16-4ads_v5") is None
        assert pool_status.vcpus_per_node(None) is None

    def test_ceiling_is_the_formulas_own_max_nodes(self):
        status = batch.PoolStatus(
            pool_id="train-big",
            allocation_state="steady",
            current_dedicated_nodes=2,
            target_dedicated_nodes=2,
            vm_size="standard_d32als_v6",
            autoscale=batch.AutoscaleRun(variables={"maxNodes": "2", "pending": "10"}),
        )
        view = pool_status._view(status, "$1.376/hr/node")
        assert (view.max_nodes, view.vcpus, view.max_vcpus) == (2, 64, 64)

    def test_no_autoscale_run_means_no_ceiling(self):
        status = batch.PoolStatus(
            pool_id="train",
            allocation_state="steady",
            current_dedicated_nodes=1,
            target_dedicated_nodes=1,
            vm_size="standard_d16als_v6",
        )
        view = pool_status._view(status, None)
        assert (view.max_nodes, view.max_vcpus) == (None, None)
        assert view.vcpus == 16


class TestScorePassthrough:
    """Extra evaluate flags reach the node, and the `--` does not.

    argparse will not hand a bare `--br-flops` to a REMAINDER positional -- it
    rejects it as an unrecognised argument of `score` itself -- so the
    separator is mandatory. Forwarding it on would then make `evaluate` read a
    bare `--` as the end of its own options.
    """

    def test_separator_is_stripped_before_the_flags_travel(self):
        args = build_parser().parse_args(["score", "--run", "r", "--", "--br-flops", "8"])
        assert score._passthrough(args.flags) == ("--br-flops", "8")

    def test_no_passthrough_is_an_empty_tuple(self):
        args = build_parser().parse_args(["score", "--run", "r"])
        assert score._passthrough(args.flags) == ()

    def test_the_parent_json_flag_still_binds_to_the_command(self):
        args = build_parser().parse_args(["score", "--run", "r", "--json"])
        assert args.json is True
        assert score._passthrough(args.flags) == ()


class TestEstimatorNamesAgree:
    """`score` must not accept a method the node will reject.

    A value that validates locally but fails `evaluate`'s argparse costs a
    snapshot upload, a ~3-minute pool spin-up, and three node allocations
    (the task retries twice) before anyone learns it was a typo. `rollout`
    was offered here after the estimator itself had been deleted.
    """

    def test_score_offers_exactly_what_evaluate_accepts(self):
        assert set(score.EVAL_METHODS) == set(evaluate.EVAL_METHODS)

    def test_a_deleted_estimator_is_rejected_before_a_node_is_allocated(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["score", "--run", "r", "--method", "rollout"])


class TestReportedLedgerPath:
    """`ledger` must name the file it actually read.

    The index is DERIVED into a temp tree on every read, so a payload that
    echoed a configured path would have an empty result blaming a file nothing
    had opened. There is no configured path any more, which makes reporting the
    real one the only option -- pinned so it stays that way.
    """

    def test_it_names_the_derived_path(self, tmp_path, published, monkeypatch):
        import argparse

        from src.interfaces.commands import ledger as ledger_cmd

        derived = tmp_path / "derived.jsonl"
        derived.write_text("")
        monkeypatch.setattr(ledger_cmd, "ledger_for", lambda root: derived)
        args = argparse.Namespace(
            run=None,
            experiment=None,
            method=None,
            since=None,
            limit=25,
            migrate=False,
            rebuild=False,
        )
        assert ledger_cmd.run(args).ledger == str(derived)


class TestAJobListingExplainsHowATaskDied:
    """`jobs` printed a bare exit code, and the code is the half nobody recalls.

    The classification rides on the payload (`outcome`, `exit_meaning`, from
    `src/shared/task_states.py`) so both surfaces word it identically. It shipped
    once with no reader at all -- the console had been carrying its own copy of
    these meanings, and deleting that left the server's version unread.
    """

    @staticmethod
    def _finished(exit_code: int) -> batch.Job:
        return batch.Job(
            job="poker-20260802",
            state="BatchJobState.ACTIVE",
            tasks=[
                batch.BatchTask(
                    task="t-1",
                    job="poker-20260802",
                    state="BatchTaskState.COMPLETED",
                    phase=task_states.Phase.FINISHED,
                    outcome=task_states.outcome_of(exit_code),
                    exit_code=exit_code,
                    exit_meaning=task_states.exit_meaning(exit_code),
                )
            ],
        )

    def test_an_oom_kill_says_so_rather_than_printing_137(self, capsys):
        jobs.render(jobs.JobsPayload(jobs=[self._finished(137)], total_jobs=1, hidden_jobs=0))
        out = capsys.readouterr().out
        assert "failed" in out
        assert "OOM killer" in out

    def test_a_hang_is_not_reported_as_a_crash(self, capsys):
        """124 is the wall-clock guard; conflating it with 137 sends someone
        hunting for memory pressure that never happened."""
        jobs.render(jobs.JobsPayload(jobs=[self._finished(124)], total_jobs=1, hidden_jobs=0))
        out = capsys.readouterr().out
        assert "timed out" in out
        assert "hang, not a crash" in out

    def test_an_unfamiliar_code_stays_a_bare_number(self, capsys):
        jobs.render(jobs.JobsPayload(jobs=[self._finished(42)], total_jobs=1, hidden_jobs=0))
        out = capsys.readouterr().out
        assert "exit=42" in out
        assert "(" not in out.split("exit=42")[1]
