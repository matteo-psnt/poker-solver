"""One class per kind, and the guards that keep the set honest.

The defect this replaced was structural: behaviour keyed off an `op` string in
four modules, where finding three of the four still ran.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.shared import tasks
from src.shared.tasks import BadTaskError, Progress, TaskKind, TaskName


def _spec(**kwargs):
    base = {
        "op": TaskName.TRAIN,
        "config": "production",
        "to": 0,
        "run_id": "",
        "arm": "",
        "eval_at": "",
        "eval_flags": (),
    }
    return SimpleNamespace(**(base | kwargs))


def _plan(**kwargs):
    base = {
        "config": "production",
        "to": 0,
        "run_id": "",
        "train_run_id": "run-a",
        "workers": 16,
        "checkpoint_every": 0,
        "experiment": "",
        "arm": "",
        "parent": "",
        "sets": (),
        "eval_method": "exact_br",
        "eval_rungs": (),
        "eval_flags": (),
    }
    return SimpleNamespace(**(base | kwargs))


class TestTheRegistryStaysHonest:
    def test_every_wire_name_has_a_kind_and_the_reverse(self):
        """The one drift a closed enum plus an open registry could hide.

        An enum member with no class is an op that validates and then has no
        argv; a class with no member is a kind the environment read rejects.
        """
        assert set(TaskKind.KINDS) == {str(name) for name in TaskName}

    def test_defining_a_subclass_is_the_whole_registration(self):
        """`__init_subclass__`, so there is no list to forget."""

        class Probe(TaskKind):
            name = "probe-only"  # ty: ignore[invalid-assignment]
            unit = "things"

            validate = commands = label = describe = sample = staticmethod(lambda *a: None)

        try:
            assert tasks.kind_of("probe-only") is not None
        finally:
            TaskKind.KINDS.pop("probe-only", None)

    def test_no_registered_kind_left_a_method_abstract(self):
        for name, instance in TaskKind.KINDS.items():
            missing = getattr(type(instance), "__abstractmethods__", frozenset())
            assert not missing, f"{name} never implemented {sorted(missing)}"

    def test_the_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            TaskKind()


class TestLookup:
    def test_the_submit_path_refuses_an_op_no_node_can_run(self):
        with pytest.raises(BadTaskError, match="Unknown task kind"):
            tasks.kind("vector-sweep")

    def test_the_read_path_tolerates_its_own_history(self):
        """`vector-sweep` and `train-vector` are in the task log from deleted
        work. Listing history must not raise on it."""
        assert tasks.kind_of("vector-sweep") is None
        assert tasks.describe({"op": "vector-sweep"}) == "vector-sweep"

    def test_a_wire_string_and_its_enum_member_are_the_same_key(self):
        assert tasks.kind("train") is tasks.kind(TaskName.TRAIN)


class TestValidation:
    def test_a_continuing_train_still_needs_its_config(self):
        """The checkpoint stores neither the tree nor the solver, so `--run x`
        alone died on the node after a snapshot upload and every retry."""
        with pytest.raises(BadTaskError, match="config"):
            tasks.kind(TaskName.TRAIN).validate(_spec(config="", run_id="run-a", to=10))

    def test_a_relative_target_is_refused(self):
        with pytest.raises(BadTaskError, match="ABSOLUTE"):
            tasks.kind(TaskName.TRAIN).validate(_spec(to=0))

    def test_an_evaluation_needs_a_run(self):
        with pytest.raises(BadTaskError, match="run id"):
            tasks.kind(TaskName.EVALUATE).validate(_spec(run_id=""))


class TestCommands:
    def test_training_never_omits_the_worker_count(self):
        """An omitted count trained SINGLE-THREADED on a 16-vCPU node."""
        (argv,) = tasks.kind(TaskName.TRAIN).commands(_plan(to=1000, workers=16))
        assert "--workers" in argv
        assert argv[argv.index("--workers") + 1] == "16"

    def test_an_unset_arm_is_absent_rather_than_empty(self):
        """`--arm ""` records an arm literally named empty string."""
        (argv,) = tasks.kind(TaskName.TRAIN).commands(_plan(to=1000, arm=""))
        assert "--arm" not in argv

    def test_scoring_a_ladder_is_one_command_per_rung(self):
        commands = tasks.kind(TaskName.EVALUATE).commands(
            _plan(run_id="run-a", eval_rungs=("1000", "2000"))
        )
        assert [c[c.index("--at") + 1] for c in commands] == ["1000", "2000"]

    def test_scoring_the_latest_checkpoint_names_no_rung(self):
        (argv,) = tasks.kind(TaskName.EVALUATE).commands(_plan(run_id="run-a", eval_rungs=()))
        assert "--at" not in argv

    def test_passthrough_flags_reach_the_node_intact(self):
        (argv,) = tasks.kind(TaskName.EVALUATE).commands(
            _plan(run_id="run-a", eval_flags=("--br-flops", "8"))
        )
        assert argv[-2:] == ["--br-flops", "8"]


class TestLabels:
    def test_evaluations_of_one_checkpoint_differ_by_seed(self):
        """The case that motivated the whole rename: three seeds on one rung
        produced three ids differing only by a timestamp and a nonce."""

        def label(seed):
            return tasks.kind(TaskName.EVALUATE).label(
                _spec(
                    run_id="run-production-025433-1095",
                    eval_at="150000000",
                    eval_flags=("--br-board-seed", seed),
                )
            )

        assert label("7") == "score-production-1095-150M-seed7"
        assert len({label("7"), label("13"), label("29")}) == 3

    def test_training_names_its_target(self):
        assert tasks.kind(TaskName.TRAIN).label(_spec(to=200_000_000)) == "train-production-to200M"

    def test_a_precompute_names_its_abstraction(self):
        assert tasks.kind(TaskName.PRECOMPUTE).label(_spec(config="ochs")) == "precompute-ochs"


class TestDescribe:
    def test_it_reads_the_kinds_own_fields(self):
        assert tasks.describe({"op": "train", "target_iteration": "5000000"}) == "train ->5M"
        assert (
            tasks.describe(
                {"op": "evaluate", "eval_at": "150000000", "eval_flags": ["--br-board-seed", "7"]}
            )
            == "evaluate @150M seed7"
        )

    def test_a_record_from_before_these_fields_degrades_to_its_op(self):
        """Honest: those records genuinely hold nothing more to show."""
        assert tasks.describe({"op": "evaluate", "target_iteration": "0"}) == "evaluate"


class TestProgress:
    def test_training_reports_against_its_target(self):
        got = tasks.kind(TaskName.TRAIN).sample(_plan(to=150_000_000), {"iteration": 30_000_000})
        assert got == Progress(30_000_000, 150_000_000, "iterations")
        assert got is not None
        assert got.phrase == "30M / 150M iterations"

    def test_a_resumed_run_past_its_target_does_not_overflow_the_bar(self):
        got = tasks.kind(TaskName.TRAIN).sample(_plan(to=1000), {"iteration": 5000})
        assert got is not None
        assert got.fraction == 1.0

    def test_a_kind_that_cannot_say_says_nothing(self):
        """None renders as no bar; zero renders as a bar that looks stuck."""
        assert tasks.kind(TaskName.TRAIN).sample(_plan(to=100), {}) is None
        assert tasks.kind(TaskName.PRECOMPUTE).sample(_plan(), {"anything": 1}) is None

    def test_it_survives_a_record_written_by_an_older_wrapper(self):
        assert Progress.from_record(None) is None
        assert Progress.from_record({"done": 1}) is None
        assert Progress.from_record({"done": 1, "total": 2, "unit": "x"}) == Progress(1, 2, "x")


class TestEstimate:
    KIND = tasks.kind(TaskName.TRAIN)

    def test_a_task_measuring_itself_beats_any_prior(self):
        """Halfway after 100s means about 100s left, whatever history says."""
        remaining = self.KIND.estimate(Progress(50, 100, "x"), elapsed=100, history=[9999])
        assert remaining == pytest.approx(100)

    def test_before_it_reports_it_falls_back_to_the_median_of_its_kind(self):
        assert self.KIND.estimate(None, elapsed=10, history=[100, 200, 300]) == pytest.approx(190)

    def test_the_first_fraction_of_a_percent_is_not_extrapolated_from(self):
        """Startup cost would otherwise project a wildly wrong total."""
        remaining = self.KIND.estimate(Progress(1, 1000, "x"), elapsed=60, history=[600])
        assert remaining == pytest.approx(540)

    def test_with_no_history_and_no_progress_it_declines_to_guess(self):
        assert self.KIND.estimate(None, elapsed=10, history=[]) is None

    def test_an_overdue_task_reports_zero_rather_than_negative(self):
        assert self.KIND.estimate(None, elapsed=500, history=[100]) == 0.0
