"""Reading the RUN_* contract, and building the argv from it.

The shell version of this was untestable in principle: the decoding lived in a
`python3 -c` heredoc feeding a NUL-separated temp file feeding `read -d ''`,
and the only way to know whether an override survived was to run a task.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.shared.node import plan as node_plan

BASE = {
    "RUN_OP": "train",
    "RUN_CONFIG": "production",
    "RUN_TO": "1000000",
    "RUN_ID": "",
    "RUN_EXPERIMENT": "",
    "RUN_ARM": "",
    "RUN_PARENT": "",
    "RUN_SETS_JSON": "[]",
    "RUN_TIMEOUT": "6h",
    "RUN_WORKERS": "16",
    "RUN_CHECKPOINT_EVERY": "1000000",
    "RUN_EVAL_METHOD": "",
    "RUN_EVAL_AT": "",
    "RUN_EVAL_FLAGS_JSON": "[]",
    "RUN_FORCE_PUBLISH": "",
}


def _plan(**overrides):
    return node_plan.parse_environment({**BASE, **overrides})


class TestOverrides:
    def test_a_value_containing_an_equals_survives(self):
        """The single behaviour the old hex encoding existed to protect."""
        assert _plan(RUN_SETS_JSON=json.dumps(["solver__dcfr=1.5"])).sets == ("solver__dcfr=1.5",)

    def test_a_value_containing_a_space_survives(self):
        """`for kv in $RUN_SETS` split on whitespace and cut this in half."""
        assert _plan(RUN_SETS_JSON=json.dumps(["system__note=two words"])).sets == (
            "system__note=two words",
        )

    def test_a_value_containing_a_newline_survives(self):
        """What line-splitting could not carry, and why the shell needed NUL
        separators and `read -d ''` to get where json.loads starts."""
        assert _plan(RUN_SETS_JSON=json.dumps(["a=one\ntwo"])).sets == ("a=one\ntwo",)

    def test_overrides_reach_the_command_line_one_flag_each(self):
        argv = _plan(RUN_SETS_JSON=json.dumps(["a=1", "b=2"])).train_argv()
        assert argv[argv.index("--set") :] == ["--set", "a=1", "--set", "b=2"]

    def test_a_malformed_payload_is_fatal_not_empty(self):
        """Zero overrides would train an experiment arm with the BASE config --
        an arm silently running as its own control, recorded that way in
        .run.json. That class of silent rebucketing has cost a curve."""
        with pytest.raises(node_plan.BadEnvironmentError, match="RUN_SETS_JSON"):
            _plan(RUN_SETS_JSON="{not json")

    def test_a_json_scalar_is_also_fatal(self):
        with pytest.raises(node_plan.BadEnvironmentError, match="array of strings"):
            _plan(RUN_SETS_JSON='"a=1"')


class TestTrainArgv:
    def test_the_target_is_absolute(self):
        """What makes a Batch retry converge instead of training twice."""
        argv = _plan(RUN_TO="25000000").train_argv()
        assert argv[argv.index("--iterations") + 1] == "25000000"

    def test_workers_is_always_passed(self):
        """Omitting it trained single-threaded on a 16-vCPU node: `train-static`
        defaults to 1, so the miss reads as a slow task, not a misconfiguration."""
        assert "--workers" in _plan().train_argv()

    def test_an_empty_worker_count_falls_back_to_the_node_cpus(self, monkeypatch):
        monkeypatch.setattr(node_plan.os, "cpu_count", lambda: 16)
        argv = _plan(RUN_WORKERS="").train_argv()
        assert argv[argv.index("--workers") + 1] == "16"

    def test_an_unknowable_cpu_count_degrades_rather_than_kills(self, monkeypatch):
        monkeypatch.setattr(node_plan.os, "cpu_count", lambda: None)
        argv = _plan(RUN_WORKERS="").train_argv()
        assert argv[argv.index("--workers") + 1] == "1"

    def test_unset_tags_are_omitted_entirely(self):
        """`--arm ""` records an arm literally named empty string rather than
        an unaffiliated run."""
        argv = _plan().train_argv()
        assert "--arm" not in argv
        assert "--experiment" not in argv
        assert "--parent" not in argv

    def test_set_tags_are_passed(self):
        argv = _plan(RUN_EXPERIMENT="exp-7", RUN_ARM="control", RUN_PARENT="run-x").train_argv()
        assert argv[argv.index("--experiment") + 1] == "exp-7"
        assert argv[argv.index("--arm") + 1] == "control"
        assert argv[argv.index("--parent") + 1] == "run-x"

    def test_an_absent_checkpoint_interval_is_left_to_the_cli(self):
        assert "--checkpoint-every" not in _plan(RUN_CHECKPOINT_EVERY="").train_argv()

    def test_a_given_run_id_is_continued(self):
        assert _plan(RUN_ID="run-abc").train_run_id == "run-abc"

    def test_a_fresh_run_id_is_derived_from_the_task(self, monkeypatch):
        """A Batch retry keeps the task id, so it continues this run rather
        than starting a second one from zero."""
        monkeypatch.setenv("AZ_BATCH_TASK_ID", "task-120000-7")
        assert _plan(RUN_ID="").train_run_id == "run-task-120000-7"


class TestEvaluateArgv:
    def test_each_rung_is_its_own_command(self):
        task = _plan(RUN_OP="evaluate", RUN_ID="run-a", RUN_EVAL_AT="1000,2000")
        assert task.eval_rungs == ("1000", "2000")
        assert task.evaluate_argv("1000")[-2:] == ["--at", "1000"]

    def test_no_rung_means_the_latest_checkpoint(self):
        task = _plan(RUN_OP="evaluate", RUN_ID="run-a", RUN_EVAL_AT="")
        assert task.eval_rungs == ()
        assert "--at" not in task.evaluate_argv("")

    def test_the_method_defaults_to_the_zero_variance_gate(self):
        task = _plan(RUN_OP="evaluate", RUN_ID="run-a", RUN_EVAL_METHOD="")
        assert task.evaluate_argv("")[-1] == "exact_br"

    def test_passthrough_flags_survive_a_space(self):
        task = _plan(
            RUN_OP="evaluate",
            RUN_ID="run-a",
            RUN_EVAL_FLAGS_JSON=json.dumps(["--opponent", "always call"]),
        )
        assert task.evaluate_argv("")[-2:] == ["--opponent", "always call"]

    def test_trailing_commas_do_not_become_an_empty_rung(self):
        assert _plan(RUN_OP="evaluate", RUN_ID="r", RUN_EVAL_AT="1000,").eval_rungs == ("1000",)


class TestValidation:
    def test_an_unknown_op_is_refused(self):
        with pytest.raises(node_plan.BadEnvironmentError, match="unknown RUN_OP"):
            _plan(RUN_OP="trian")

    def test_a_training_task_without_a_config_is_refused(self):
        """The config builds the tree and the solver; the checkpoint stores
        neither, so a CONTINUING task needs it too."""
        with pytest.raises(node_plan.BadEnvironmentError, match="RUN_CONFIG"):
            _plan(RUN_CONFIG="")

    def test_a_relative_target_is_refused(self):
        with pytest.raises(node_plan.BadEnvironmentError, match="ABSOLUTE"):
            _plan(RUN_TO="0")

    @pytest.mark.parametrize("op", ["evaluate"])
    def test_an_op_on_an_existing_run_needs_one(self, op):
        with pytest.raises(node_plan.BadEnvironmentError, match="RUN_ID"):
            _plan(RUN_OP=op, RUN_ID="", RUN_CONFIG="production")

    def test_precompute_needs_a_config(self):
        with pytest.raises(node_plan.BadEnvironmentError, match="RUN_CONFIG"):
            _plan(RUN_OP="precompute", RUN_CONFIG="")


class TestDuration:
    @pytest.mark.parametrize(
        ("raw", "seconds"),
        [("6h", 21600), ("90m", 5400), ("30s", 30), ("2d", 172800), ("3600", 3600)],
    )
    def test_it_reads_the_forms_a_submission_can_carry(self, raw, seconds):
        assert node_plan.parse_duration(raw) == seconds

    @pytest.mark.parametrize("raw", ["", None, "garbage", "0h", "-1"])
    def test_anything_unreadable_falls_back_to_the_default_ceiling(self, raw):
        """A task with no guard bills a full node-day before Batch acts."""
        assert node_plan.parse_duration(raw) == node_plan.DEFAULT_TIMEOUT_SECONDS


class TestTheSubmitterContract:
    """`spec.TaskSpec.environment` writes exactly what this reads."""

    def test_every_key_the_submitter_emits_is_consumed(self):
        from src.interfaces.cloud import spec

        emitted = set(spec.TaskSpec(code_snapshot="s", config="p", to=1).environment())
        source = pathlib.Path(str(node_plan.__file__)).read_text()
        unread = {key for key in emitted if key.startswith("RUN_") and key not in source}
        assert not unread, f"the submitter sets {sorted(unread)} and the node never reads it"

    def test_a_full_submission_round_trips(self):
        from src.interfaces.cloud import spec

        task = spec.TaskSpec(
            code_snapshot="snap",
            config="production",
            to=25_000_000,
            run_id="run-a",
            experiment="exp-7",
            arm="variant",
            sets=("solver__dcfr=1.5", "system__note=two words"),
            workers=16,
        )
        parsed = node_plan.parse_environment(task.environment())
        assert parsed.config == "production"
        assert parsed.to == 25_000_000
        assert parsed.run_id == "run-a"
        assert parsed.experiment == "exp-7"
        assert parsed.arm == "variant"
        assert parsed.sets == ("solver__dcfr=1.5", "system__note=two words")
        assert parsed.workers == 16
