"""The task spec: the half of a submission that can be wrong on purpose.

Dispatch is untested here by design -- it talks to a service. What is tested is
everything that decides WHAT gets dispatched, which is where the bugs that cost
a node live: a relative iteration target, an override that loses half its
value, a task id Batch will not accept.
"""

import json
import pathlib
from datetime import UTC, datetime

import pytest

from src.interfaces.cloud import spec

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

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
        assert spec.task_id("", NOW, 3).startswith("task-")

    def test_an_over_long_label_is_trimmed_not_rejected(self):
        """Batch caps an id at 64 characters and refuses a longer one.

        The nonce suffix is what keeps two submissions in one second apart, so
        it has to survive the trim -- otherwise the fix for unreadable ids
        would reintroduce collisions.
        """
        got = spec.task_id("x" * 200, NOW, 7)
        assert len(got) == spec.TASK_ID_LIMIT
        assert got.endswith("-213805-7")

    def test_trimming_never_leaves_a_dangling_separator(self):
        label = "a" * (spec.TASK_ID_LIMIT - len("-213805-7") - 1) + "-tail"
        assert "--" not in spec.task_id(label, NOW, 7)


class TestTaskCommand:
    def test_node_side_variables_are_not_expanded_locally(self):
        """The `$`-names must reach the node's shell intact; only the snapshot
        is substituted here."""
        command = spec.task_command("code-20260802_000000")
        assert "$AZ_BATCH_TASK_ID" in command
        assert "$AZ_BATCH_NODE_MOUNTS_DIR" in command
        assert "code-20260802_000000.tar.gz" in command

    def test_it_bootstraps_the_wrapper_from_inside_the_tarball(self):
        command = spec.task_command("snap")
        assert command.index("tar xzf") < command.index("run_task.py")

    def test_the_interpreter_is_the_one_the_start_task_installs(self):
        """A contract split across two languages, and nothing else joins them.

        The wrapper no longer runs on the system python3 (3.10 on the pinned
        image); it runs the interpreter `infra/main.tf` installs. If that line
        changes version, or moves uv's install dir, every task dies before it
        can write a start record -- which reads as "no tasks ran", not as a
        broken command line.
        """
        terraform = (REPO_ROOT / "infra" / "main.tf").read_text()
        assert f"uv python install {spec.NODE_PYTHON}" in terraform
        bin_dir = spec.NODE_PYTHON_BIN.rsplit("/", 1)[0]
        assert f"UV_PYTHON_BIN_DIR={bin_dir}" in terraform

    def test_the_interpreter_is_reachable_by_a_user_that_is_not_the_installer(self):
        """The failure this replaced, and the reason both dirs are explicit.

        uv resolves a managed interpreter under HOME. The start task runs as an
        elevated POOL-scoped auto-user; a task runs as a different one whose
        HOME is its own working directory. Installing to a shared dir is only
        half of it -- it also has to be readable by the user that did not
        install it.
        """
        terraform = (REPO_ROOT / "infra" / "main.tf").read_text()
        assert "UV_PYTHON_INSTALL_DIR=/opt/uv-python" in terraform
        assert "chmod -R a+rX /opt/uv-python" in terraform

    def test_no_uv_at_task_time(self):
        """The wrapper explains failures, so it cannot depend on a resolver that
        can fail. An absolute path to a plain interpreter cannot."""
        command = spec.task_command("snap")
        assert spec.NODE_PYTHON_BIN in command
        assert "uv " not in command


class TestEnvironment:
    def test_overrides_survive_their_equals_sign(self):
        """The single behaviour the old hex encoding existed to protect."""
        env = spec.TaskSpec(
            code_snapshot="s", config="production", to=10, sets=("solver__dcfr=1.5",)
        ).environment()
        assert json.loads(env["RUN_SETS_JSON"]) == ["solver__dcfr=1.5"]

    def test_overrides_survive_spaces_that_the_old_form_split(self):
        """The space-joined shell form silently broke a value containing a
        space; a JSON array cannot."""
        env = spec.TaskSpec(
            code_snapshot="s", config="production", to=10, sets=("system__note=two words",)
        ).environment()
        assert json.loads(env["RUN_SETS_JSON"]) == ["system__note=two words"]

    def test_multiple_overrides_stay_separate(self):
        env = spec.TaskSpec(code_snapshot="s", config="p", to=10, sets=("a=1", "b=2")).environment()
        assert json.loads(env["RUN_SETS_JSON"]) == ["a=1", "b=2"]

    def test_eval_flags_round_trip_the_same_way(self):
        env = spec.TaskSpec(
            code_snapshot="s", op=spec.EVALUATE, run_id="run-a", eval_flags=("--br-flops", "8")
        ).environment()
        assert json.loads(env["RUN_EVAL_FLAGS_JSON"]) == ["--br-flops", "8"]

    def test_every_key_the_wrapper_reads_is_present(self):
        """The node reads these by name; a missing one is a silent default."""
        env = spec.TaskSpec(code_snapshot="s", config="p", to=1).environment()
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
            "RUN_GIT_COMMIT",
            "RUN_GIT_DIRTY",
            "RUN_EVAL_METHOD",
            "RUN_EVAL_AT",
            "RUN_EVAL_FLAGS_JSON",
            "RUN_FORCE_PUBLISH",
        }

    def test_every_value_is_a_string(self):
        """Batch environment values are strings; an int here fails at the wire."""
        env = spec.TaskSpec(code_snapshot="s", config="p", to=25_000_000, workers=8).environment()
        assert all(isinstance(value, str) for value in env.values())

    def test_absent_workers_means_all_cpus_not_zero(self):
        env = spec.TaskSpec(code_snapshot="s", config="p", to=1, workers=None).environment()
        assert env["RUN_WORKERS"] == ""


class TestLabel:
    """A label says what the task DOES. See ``TaskSpec.label`` for why."""

    def test_continuing_a_run_labels_by_run_and_target(self):
        task = spec.TaskSpec(code_snapshot="s", config="production", run_id="run-a", to=200_000_000)
        assert task.label == "train-a-to200M"

    def test_a_fresh_run_labels_by_config(self):
        assert spec.TaskSpec(code_snapshot="s", config="production").label == "train-production"

    def test_evaluations_differing_only_by_board_seed_get_different_labels(self):
        """The case that motivated this: three seeds on ONE checkpoint.

        Before, all three were `run-production-025433-1095` and the ids differed
        only in a timestamp and a nonce, so the seed lived nowhere but the
        submitter's memory.
        """

        def score(seed: str) -> str:
            return spec.TaskSpec(
                code_snapshot="s",
                op=spec.EVALUATE,
                run_id="run-production-025433-1095",
                eval_at="150000000",
                eval_flags=("--br-flops", "4", "--br-board-seed", seed),
            ).label

        assert score("7") == "score-production-1095-150M-seed7"
        assert len({score("7"), score("13"), score("29")}) == 3

    def test_an_arm_is_kept_because_it_is_what_distinguishes_two_arms(self):
        task = spec.TaskSpec(
            code_snapshot="s", config="ochs_gate", to=1_000_000, experiment="ochs", arm="river"
        )
        assert task.label == "train-ochs_gate-to1M-river"


class TestRunToken:
    def test_the_unread_timestamp_in_the_middle_is_dropped(self):
        assert spec.run_token("run-production-025433-1095") == "production-1095"

    def test_a_two_segment_id_is_left_alone(self):
        assert spec.run_token("run-20260802_201939-ee77cb") == "20260802_201939-ee77cb"


class TestValidate:
    def test_a_training_task_needs_a_positive_absolute_target(self):
        with pytest.raises(ValueError, match="ABSOLUTE"):
            spec.TaskSpec(code_snapshot="s", config="production", to=0).validate()

    def test_a_training_task_needs_a_config(self):
        with pytest.raises(ValueError, match="--config"):
            spec.TaskSpec(code_snapshot="s", to=1000).validate()

    def test_continuing_a_run_needs_a_config_too(self):
        """The config builds the tree and the solver; the checkpoint stores
        neither. This was permitted, and died on the node with
        `Config file not found: config/training/.yaml` -- after the upload, the
        spin-up and three retries."""
        with pytest.raises(ValueError, match="CONTINUING"):
            spec.TaskSpec(code_snapshot="s", run_id="run-a", to=1000).validate()

    def test_scoring_needs_a_run(self):
        with pytest.raises(ValueError, match="--run"):
            spec.TaskSpec(code_snapshot="s", op=spec.EVALUATE).validate()

    def test_a_malformed_override_is_refused_before_it_reaches_a_node(self):
        with pytest.raises(ValueError, match="key=value"):
            spec.TaskSpec(code_snapshot="s", config="p", to=1, sets=("solver__dcfr",)).validate()

    def test_an_unknown_op_is_refused(self):
        with pytest.raises(ValueError, match="Unknown op"):
            spec.TaskSpec(code_snapshot="s", op="frobnicate", run_id="r").validate()


class TestPrecomputeTask:
    """Building an abstraction on a node is a first-class op.

    The invariant that kept precompute local was *computed once, never
    recomputed* -- never *computed locally*. What the node must refuse is a
    REPUBLISH, since bucket assignment is not pinned by the abstraction hash.
    """

    def test_it_needs_a_config(self):
        with pytest.raises(ValueError, match="abstraction config"):
            spec.TaskSpec(code_snapshot="s", op=spec.PRECOMPUTE).validate()

    def test_a_config_is_all_it_needs(self):
        spec.TaskSpec(code_snapshot="s", op=spec.PRECOMPUTE, config="ochs_gate_ochs").validate()

    def test_it_needs_no_iteration_target(self):
        """Unlike a training task -- there is nothing to converge to."""
        task = spec.TaskSpec(code_snapshot="s", op=spec.PRECOMPUTE, config="x", to=0)
        task.validate()
        assert task.environment()["RUN_TO"] == "0"

    def test_force_publish_is_off_unless_asked(self):
        env = spec.TaskSpec(code_snapshot="s", op=spec.PRECOMPUTE, config="x").environment()
        assert env["RUN_FORCE_PUBLISH"] == ""

    def test_force_publish_reaches_the_node(self):
        env = spec.TaskSpec(
            code_snapshot="s", op=spec.PRECOMPUTE, config="x", force_publish=True
        ).environment()
        assert env["RUN_FORCE_PUBLISH"] == "1"
