"""The read-only cloud verbs, pinned on the two things a port can silently drop.

Neither of these is about formatting. Each encodes a fact about how Batch
reports state that took a live incident to learn, and a rewrite that loses
either one still looks perfectly healthy.
"""

import pytest

from src.interfaces.cli.commands import evaluate, jobs, pool_status, score
from src.interfaces.cli.flows import training
from src.interfaces.cli.headless import build_parser


def _job(job_state: str, task_state: str) -> dict:
    return {
        "job": "poker-20260802",
        "state": job_state,
        "tasks": [
            {"task": "t-1", "state": task_state, "exit_code": None, "node": "tvmps_x"},
        ],
    }


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

    def test_the_interactive_menu_offers_the_same_set(self):
        assert set(training.MENU_EVAL_METHODS) == set(evaluate.EVAL_METHODS)

    def test_a_deleted_estimator_is_rejected_before_a_node_is_allocated(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["score", "--run", "r", "--method", "rollout"])


class TestReportedLedgerPath:
    """`ledger` must name the file it read, not the one it was asked for.

    Under `--source share` the index is derived into a temp dir, so echoing
    `--ledger` had an empty share blaming a local file nothing had opened.
    """

    def test_it_names_the_derived_path_not_the_requested_one(self, tmp_path, monkeypatch):
        import argparse

        from src.interfaces.cli.commands import ledger as ledger_cmd

        derived = tmp_path / "derived.jsonl"
        derived.write_text("")
        monkeypatch.setattr(ledger_cmd, "ledger_for", lambda args, root: derived)
        args = argparse.Namespace(
            source="local",
            runs_dir=str(tmp_path),
            ledger="data/eval_ledger.jsonl",
            run=None,
            experiment=None,
            method=None,
            since=None,
            limit=25,
            migrate=False,
            rebuild=False,
        )
        assert ledger_cmd.run(args)["ledger"] == str(derived)
