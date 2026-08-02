"""The leg spec: the half of a submission that can be wrong on purpose.

Dispatch is untested here by design -- it talks to a service. What is tested is
everything that decides WHAT gets dispatched, which is where the bugs that cost
a node live: a relative iteration target, an override that loses half its
value, a task id Batch will not accept.
"""

import json
from datetime import UTC, datetime

import pytest

from src.interfaces.cloud import spec

NOW = datetime(2026, 8, 2, 21, 38, 5, tzinfo=UTC)


class TestJobIds:
    def test_daily_job_id_is_one_per_utc_day(self):
        assert spec.daily_job_id(NOW) == "poker-20260802"

    def test_suffixed_id_differs_so_a_stopped_job_cannot_block_the_day(self):
        """A stopped job answers JobCompleted to every task creation. Without a
        second id, one `panic` blocks all submissions until midnight UTC."""
        assert spec.suffixed_job_id(NOW) == "poker-20260802-213805"
        assert spec.suffixed_job_id(NOW) != spec.daily_job_id(NOW)


class TestTaskIds:
    def test_run_id_characters_batch_rejects_are_replaced(self):
        got = spec.task_id("run-production/025433:1095", NOW, 7)
        assert got == "run-production-025433-1095-213805-7"

    def test_nonce_separates_two_submissions_in_the_same_second(self):
        first = spec.task_id("production", NOW, 1)
        second = spec.task_id("production", NOW, 2)
        assert first != second

    def test_empty_label_still_yields_a_usable_id(self):
        assert spec.task_id("", NOW, 3).startswith("leg-")


class TestLegCommand:
    def test_node_side_variables_are_not_expanded_locally(self):
        """The `$`-names must reach the node's shell intact; only the snapshot
        is substituted here."""
        command = spec.leg_command("code-20260802_000000")
        assert "$AZ_BATCH_TASK_ID" in command
        assert "$AZ_BATCH_NODE_MOUNTS_DIR" in command
        assert "code-20260802_000000.tar.gz" in command

    def test_it_bootstraps_the_wrapper_from_inside_the_tarball(self):
        command = spec.leg_command("snap")
        assert command.index("tar xzf") < command.index("run_leg.sh")


class TestEnvironment:
    def test_overrides_survive_their_equals_sign(self):
        """The single behaviour the old hex encoding existed to protect."""
        env = spec.LegSpec(
            code_snapshot="s", config="production", to=10, sets=("solver__dcfr=1.5",)
        ).environment()
        assert json.loads(env["RUN_SETS_JSON"]) == ["solver__dcfr=1.5"]

    def test_overrides_survive_spaces_that_the_old_form_split(self):
        """The space-joined shell form silently broke a value containing a
        space; a JSON array cannot."""
        env = spec.LegSpec(
            code_snapshot="s", config="production", to=10, sets=("system__note=two words",)
        ).environment()
        assert json.loads(env["RUN_SETS_JSON"]) == ["system__note=two words"]

    def test_multiple_overrides_stay_separate(self):
        env = spec.LegSpec(code_snapshot="s", config="p", to=10, sets=("a=1", "b=2")).environment()
        assert json.loads(env["RUN_SETS_JSON"]) == ["a=1", "b=2"]

    def test_eval_flags_round_trip_the_same_way(self):
        env = spec.LegSpec(
            code_snapshot="s", op=spec.EVALUATE, run_id="run-a", eval_flags=("--br-flops", "8")
        ).environment()
        assert json.loads(env["RUN_EVAL_FLAGS_JSON"]) == ["--br-flops", "8"]

    def test_every_key_the_wrapper_reads_is_present(self):
        """run_leg.sh reads these by name; a missing one is a silent default."""
        env = spec.LegSpec(code_snapshot="s", config="p", to=1).environment()
        assert set(env) == {
            "CODE_SNAPSHOT",
            "RUN_OP",
            "RUN_CONFIG",
            "RUN_TO",
            "RUN_ID",
            "RUN_EXPERIMENT",
            "RUN_ARM",
            "RUN_PARENT",
            "RUN_SETS_JSON",
            "RUN_TIMEOUT",
            "RUN_WORKERS",
            "RUN_CHECKPOINT_EVERY",
            "RUN_EVAL_METHOD",
            "RUN_EVAL_AT",
            "RUN_EVAL_FLAGS_JSON",
        }

    def test_every_value_is_a_string(self):
        """Batch environment values are strings; an int here fails at the wire."""
        env = spec.LegSpec(code_snapshot="s", config="p", to=25_000_000, workers=8).environment()
        assert all(isinstance(value, str) for value in env.values())

    def test_absent_workers_means_all_cpus_not_zero(self):
        env = spec.LegSpec(code_snapshot="s", config="p", to=1, workers=None).environment()
        assert env["RUN_WORKERS"] == ""


class TestLabel:
    def test_continuing_a_run_labels_by_run_id(self):
        assert spec.LegSpec(code_snapshot="s", config="production", run_id="run-a").label == "run-a"

    def test_a_fresh_run_labels_by_config(self):
        assert spec.LegSpec(code_snapshot="s", config="production").label == "production"


class TestValidate:
    def test_a_training_leg_needs_a_positive_absolute_target(self):
        with pytest.raises(ValueError, match="ABSOLUTE"):
            spec.LegSpec(code_snapshot="s", config="production", to=0).validate()

    def test_a_training_leg_needs_a_config_or_a_run(self):
        with pytest.raises(ValueError, match="--config"):
            spec.LegSpec(code_snapshot="s", to=1000).validate()

    def test_continuing_a_run_needs_no_config(self):
        spec.LegSpec(code_snapshot="s", run_id="run-a", to=1000).validate()

    def test_scoring_needs_a_run(self):
        with pytest.raises(ValueError, match="--run"):
            spec.LegSpec(code_snapshot="s", op=spec.EVALUATE).validate()

    def test_a_malformed_override_is_refused_before_it_reaches_a_node(self):
        with pytest.raises(ValueError, match="key=value"):
            spec.LegSpec(code_snapshot="s", config="p", to=1, sets=("solver__dcfr",)).validate()

    def test_an_unknown_op_is_refused(self):
        with pytest.raises(ValueError, match="Unknown op"):
            spec.LegSpec(code_snapshot="s", op="frobnicate", run_id="r").validate()
