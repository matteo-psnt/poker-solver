"""Reading the RUN_* contract, and building the argv from it.

The shell version of this was untestable in principle: the decoding lived in a
`python3 -c` heredoc feeding a NUL-separated temp file feeding `read -d ''`,
and the only way to know whether an override survived was to run a task.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.shared.cloudtask import wire
from src.shared.cloudtask.node import plan as node_plan

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
    "RUN_RETAIN_EVERY": "",
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
        argv = _plan(RUN_SETS_JSON=json.dumps(["a=1", "b=2"])).commands[0]
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
        argv = _plan(RUN_TO="25000000").commands[0]
        assert argv[argv.index("--iterations") + 1] == "25000000"

    def test_workers_is_always_passed(self):
        """Omitting it trained single-threaded on a 16-vCPU node: `train-static`
        defaults to 1, so the miss reads as a slow task, not a misconfiguration."""
        assert "--workers" in _plan().commands[0]

    def test_an_empty_worker_count_falls_back_to_all_node_cpus_but_one(self, monkeypatch):
        """One hardware thread stays free for the OS and coordinator: a fully
        subscribed box measured 15% slower (worker-curve, 08-24)."""
        monkeypatch.setattr(node_plan.os, "cpu_count", lambda: 16)
        argv = _plan(RUN_WORKERS="").commands[0]
        assert argv[argv.index("--workers") + 1] == "15"

    def test_an_unknowable_cpu_count_degrades_rather_than_kills(self, monkeypatch):
        monkeypatch.setattr(node_plan.os, "cpu_count", lambda: None)
        argv = _plan(RUN_WORKERS="").commands[0]
        assert argv[argv.index("--workers") + 1] == "1"

    def test_unset_tags_are_omitted_entirely(self):
        """`--arm ""` records an arm literally named empty string rather than
        an unaffiliated run."""
        argv = _plan().commands[0]
        assert "--arm" not in argv
        assert "--experiment" not in argv
        assert "--parent" not in argv

    def test_set_tags_are_passed(self):
        argv = _plan(RUN_EXPERIMENT="exp-7", RUN_ARM="control", RUN_PARENT="run-x").commands[0]
        assert argv[argv.index("--experiment") + 1] == "exp-7"
        assert argv[argv.index("--arm") + 1] == "control"
        assert argv[argv.index("--parent") + 1] == "run-x"

    def test_an_absent_checkpoint_interval_is_left_to_the_cli(self):
        assert "--checkpoint-every" not in _plan(RUN_CHECKPOINT_EVERY="").commands[0]

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
        assert [argv[argv.index("--at") + 1] for argv in task.commands] == ["1000", "2000"]

    def test_no_rung_means_the_latest_checkpoint(self):
        task = _plan(RUN_OP="evaluate", RUN_ID="run-a", RUN_EVAL_AT="")
        assert task.eval_rungs == ()
        assert "--at" not in task.commands[0]

    def test_the_method_defaults_to_the_zero_variance_gate(self):
        task = _plan(RUN_OP="evaluate", RUN_ID="run-a", RUN_EVAL_METHOD="")
        argv = task.commands[0]
        assert argv[argv.index("--method") + 1] == "exact_br"

    def test_passthrough_flags_survive_a_space(self):
        task = _plan(
            RUN_OP="evaluate",
            RUN_ID="run-a",
            RUN_EVAL_FLAGS_JSON=json.dumps(["--opponent", "always call"]),
        )
        assert task.commands[0][-2:] == ["--opponent", "always call"]

    def test_trailing_commas_do_not_become_an_empty_rung(self):
        assert _plan(RUN_OP="evaluate", RUN_ID="r", RUN_EVAL_AT="1000,").eval_rungs == ("1000",)


class TestValidation:
    def test_a_training_task_without_a_config_is_refused(self):
        """The config builds the tree and the solver; the checkpoint stores
        neither, so a CONTINUING task needs it too."""
        with pytest.raises(node_plan.BadEnvironmentError, match="needs a config"):
            _plan(RUN_CONFIG="")

    def test_a_relative_target_is_refused(self):
        with pytest.raises(node_plan.BadEnvironmentError, match="ABSOLUTE"):
            _plan(RUN_TO="0")

    def test_an_evaluation_needs_a_run(self):
        with pytest.raises(node_plan.BadEnvironmentError, match="run id"):
            _plan(RUN_OP="evaluate", RUN_ID="", RUN_CONFIG="production")

    def test_precompute_needs_a_config(self):
        with pytest.raises(node_plan.BadEnvironmentError, match="abstraction config"):
            _plan(RUN_OP="precompute", RUN_CONFIG="")

    def test_the_node_refuses_exactly_what_the_submitter_would_have(self):
        """Both ends call the same kind, so neither can drift from the other."""
        with pytest.raises(node_plan.BadEnvironmentError, match="Unknown task kind"):
            _plan(RUN_OP="trian")


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


# One distinctive value per wire key, chosen to look wrong if it lands anywhere
# else, so a key decoded into the WRONG FIELD fails. Adding a wire key without a
# row here fails `test_every_wire_key_is_exercised`.
WIRE_SAMPLES: dict[str, tuple[Any, Any]] = {
    "CODE_SNAPSHOT": ("snap-1", "snap-1"),
    "RUN_OP": ("evaluate", "evaluate"),
    "RUN_CONFIG": ("production", "production"),
    "RUN_TO": (25_000_000, 25_000_000),
    "RUN_ID": ("run-a", "run-a"),
    "RUN_EXPERIMENT": ("exp-7", "exp-7"),
    "RUN_ARM": ("variant", "variant"),
    "RUN_PARENT": ("run-parent", "run-parent"),
    "RUN_SETS_JSON": (
        ("solver__dcfr=1.5", "system__note=two words"),
        ("solver__dcfr=1.5", "system__note=two words"),
    ),
    # Written as a duration, read as seconds.
    "RUN_TIMEOUT": ("90m", 5400),
    "RUN_WORKERS": (16, 16),
    "RUN_CHECKPOINT_EVERY": (500_000, 500_000),
    "RUN_RETAIN_EVERY": (800, 800),
    "RUN_EVAL_METHOD": ("lbr", "lbr"),
    # One rung at submit; a ladder on the node.
    "RUN_EVAL_AT": ("5000000,10000000", ("5000000", "10000000")),
    "RUN_EVAL_FLAGS_JSON": (("--br-flops", "64"), ("--br-flops", "64")),
    "RUN_FORCE_PUBLISH": (True, True),
    "RUN_GIT_COMMIT": ("c13dcb7", "c13dcb7"),
    "RUN_GIT_DIRTY": ("1", "1"),
    "RUN_GIT_BRANCH": ("wire-and-share", "wire-and-share"),
    "RUN_UNIVERSE_BOARDS": (4096, 4096),
    "RUN_UNIVERSE_SEED": (17, 17),
    "RUN_DTYPE": ("float32", "float32"),
    "RUN_WARM_START_FROM": ("run-prior", "run-prior"),
    "RUN_WARM_START_WEIGHT": (3000, 3000),
    "RUN_WARM_START_AT": (9, 9),
    "RUN_WARM_START_SHAPE": ("confidence", "confidence"),
    "RUN_EQUITY_PRIOR": (250, 250),
    "RUN_EQUITY_PRIOR_TEMPERATURE": (0.4, 0.4),
}


class TestTheSubmitterContract:
    """`spec.TaskSpec.environment` writes exactly what this reads.

    Behavioural, not textual: it round-trips real values through the real
    encoder and decoder rather than looking for key names in a file.
    """

    def test_every_wire_key_is_exercised(self):
        """A new key with no sample is a key nothing round-trips."""
        assert set(WIRE_SAMPLES) == {key.env for key in wire.KEYS}

    def test_the_submitter_emits_exactly_the_declared_keys(self):
        from src.interfaces.cloud.tasks import spec

        emitted = set(spec.TaskSpec(code_snapshot="s", config="p", to=1).environment())
        assert emitted == {key.env for key in wire.KEYS}

    @pytest.mark.parametrize("key", wire.KEYS, ids=lambda k: k.env)
    def test_each_key_survives_the_crossing_into_the_right_field(self, key):
        """Every key, one at a time, from a spec attribute to a plan attribute.

        Per-key rather than one fat round-trip so a failure names the field that
        broke instead of the whole contract.
        """
        from src.interfaces.cloud.tasks import spec

        sent, expected = WIRE_SAMPLES[key.env]
        # A base that every kind accepts, so this measures the crossing rather
        # than re-testing `kinds.validate`.
        base: dict[str, Any] = {
            "code_snapshot": "s",
            "config": "p",
            "to": 1,
            "run_id": "run-base",
        }
        task = spec.TaskSpec(**{**base, key.spec: sent})
        parsed = node_plan.parse_environment(task.environment())
        assert getattr(parsed, key.plan) == expected

    def test_a_key_the_submitter_stops_sending_still_parses(self):
        """An absent variable and an empty one mean the same thing to the node."""
        parsed = node_plan.parse_environment({"RUN_OP": "train", "RUN_CONFIG": "p", "RUN_TO": "1"})
        assert parsed.op == "train"
        assert parsed.eval_rungs == ()
        assert parsed.timeout_seconds == wire.DEFAULT_TIMEOUT_SECONDS

    def test_a_full_submission_round_trips(self):
        from src.interfaces.cloud.tasks import spec

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


class TestAnEvaluationGetsTheNodesCores:
    """`--workers` defaults to 1 and nothing passed it.

    So every evaluation ever run on the pool used ONE core of a 16-core box:
    `exact_br` took its serial path and `lbr` its single-process one. The box was
    already paid for and idle for the whole score.
    """

    def test_the_resolved_core_count_is_passed(self):
        argv = _plan(RUN_OP="evaluate", RUN_ID="run-a", RUN_WORKERS="16").commands[0]
        assert argv[argv.index("--workers") + 1] == "16"

    def test_an_explicit_workers_flag_still_wins(self):
        """Ours goes first and the passthrough last; argparse takes the last."""
        argv = _plan(
            RUN_OP="evaluate",
            RUN_ID="run-a",
            RUN_WORKERS="16",
            RUN_EVAL_FLAGS_JSON='["--workers", "2"]',
        ).commands[0]
        assert argv[-2:] == ["--workers", "2"]

    def test_every_rung_of_a_ladder_gets_them(self):
        commands = _plan(
            RUN_OP="evaluate", RUN_ID="run-a", RUN_WORKERS="8", RUN_EVAL_AT="1000,2000"
        ).commands
        assert len(commands) == 2
        assert all("--workers" in argv for argv in commands)
