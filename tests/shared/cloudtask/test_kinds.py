"""One class per kind, and the guards that keep the set honest.

The defect this replaced was structural: behaviour keyed off an `op` string in
four modules, where finding three of the four still ran.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.shared.cloudtask import kinds
from src.shared.cloudtask.kinds import BadTaskError, Progress, Sample, TaskKind, TaskName
from src.shared.cloudtask.node import plan as node_plan

# Deliberately not a real retired op -- see
# `TestLookup.test_the_read_path_tolerates_its_own_history` for why naming one
# has broken this test twice.
RETIRED_OP = "train-dynamic"


def _spec(**kwargs):
    base = {
        "op": TaskName.TRAIN,
        "config": "production",
        "to": 0,
        "run_id": "",
        "arm": "",
        "eval_at": "",
        "eval_flags": (),
        "universe_boards": 0,
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
        # Every field of `tasks.NodePlan`, including the ones a given kind does
        # not read: a stand-in that is missing one only proves the kinds tolerate
        # a shape the node never hands them.
        "progress_path": "",
        "universe_boards": 0,
        "universe_seed": 0,
        "dtype": "",
        "warm_start_from": "",
        "warm_start_weight": 0,
        "warm_start_at": 0,
    }
    return SimpleNamespace(**(base | kwargs))


class TestTheRegistryStaysHonest:
    def test_every_wire_name_has_a_kind_and_the_reverse(self):
        """The one drift a closed enum plus an open registry could hide.

        An enum member with no class is an op that validates and then has no
        argv; a class with no member is a kind the environment read rejects.
        """
        assert set(kinds.KINDS) == {str(name) for name in TaskName}

    def test_no_registered_kind_left_a_method_abstract(self):
        for name, instance in kinds.KINDS.items():
            missing = getattr(type(instance), "__abstractmethods__", frozenset())
            assert not missing, f"{name} never implemented {sorted(missing)}"

    def test_the_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            TaskKind()


class TestLookup:
    def test_the_submit_path_refuses_an_op_no_node_can_run(self):
        with pytest.raises(BadTaskError, match="Unknown task kind"):
            kinds.kind(RETIRED_OP)

    def test_the_read_path_tolerates_its_own_history(self):
        """An op the log holds and this code no longer defines must still READ.

        The name is synthetic ON PURPOSE. This test twice named a real op --
        `vector-sweep`, then `train-vector` -- and twice broke when that op came
        back as a live kind. Every op string this project has ever defined is
        live again today, so there is no retired one to point at, and pinning
        the property to whichever is currently dead only schedules the next
        failure. What is being tested is the SHAPE: unknown reads as ``None``
        and degrades to its bare op, while a live one still resolves.
        """
        assert kinds.kind_of(RETIRED_OP) is None
        assert kinds.describe({"op": RETIRED_OP}) == RETIRED_OP
        assert kinds.kind_of("vector-sweep") is not None
        assert kinds.kind_of("train-vector") is not None

    def test_a_wire_string_and_its_enum_member_are_the_same_key(self):
        assert kinds.kind("train") is kinds.kind(TaskName.TRAIN)


class TestValidation:
    def test_a_continuing_train_still_needs_its_config(self):
        """The checkpoint stores neither the tree nor the solver, so `--run x`
        alone died on the node after a snapshot upload and every retry."""
        with pytest.raises(BadTaskError, match="config"):
            kinds.kind(TaskName.TRAIN).validate(_spec(config="", run_id="run-a", to=10))

    def test_a_relative_target_is_refused(self):
        with pytest.raises(BadTaskError, match="ABSOLUTE"):
            kinds.kind(TaskName.TRAIN).validate(_spec(to=0))

    def test_an_evaluation_needs_a_run(self):
        with pytest.raises(BadTaskError, match="run id"):
            kinds.kind(TaskName.EVALUATE).validate(_spec(run_id=""))


class TestCommands:
    def test_training_never_omits_the_worker_count(self):
        """An omitted count trained SINGLE-THREADED on a 16-vCPU node."""
        (argv,) = kinds.kind(TaskName.TRAIN).commands(_plan(to=1000, workers=16))
        assert "--workers" in argv
        assert argv[argv.index("--workers") + 1] == "16"

    def test_an_unset_arm_is_absent_rather_than_empty(self):
        """`--arm ""` records an arm literally named empty string."""
        (argv,) = kinds.kind(TaskName.TRAIN).commands(_plan(to=1000, arm=""))
        assert "--arm" not in argv

    def test_scoring_a_ladder_is_one_command_per_rung(self):
        commands = kinds.kind(TaskName.EVALUATE).commands(
            _plan(run_id="run-a", eval_rungs=("1000", "2000"))
        )
        assert [c[c.index("--at") + 1] for c in commands] == ["1000", "2000"]

    def test_scoring_the_latest_checkpoint_names_no_rung(self):
        (argv,) = kinds.kind(TaskName.EVALUATE).commands(_plan(run_id="run-a", eval_rungs=()))
        assert "--at" not in argv

    def test_passthrough_flags_reach_the_node_intact(self):
        (argv,) = kinds.kind(TaskName.EVALUATE).commands(
            _plan(run_id="run-a", eval_flags=("--br-flops", "8"))
        )
        assert argv[-2:] == ["--br-flops", "8"]


class TestLabels:
    def test_evaluations_of_one_checkpoint_differ_by_seed(self):
        """The case that motivated the whole rename: three seeds on one rung
        produced three ids differing only by a timestamp and a nonce."""

        def label(seed):
            return kinds.kind(TaskName.EVALUATE).label(
                _spec(
                    run_id="run-production-025433-1095",
                    eval_at="150000000",
                    eval_flags=("--br-board-seed", seed),
                )
            )

        assert label("7") == "score-production-1095-150M-seed7"
        assert len({label("7"), label("13"), label("29")}) == 3

    def test_training_names_its_target(self):
        assert kinds.kind(TaskName.TRAIN).label(_spec(to=200_000_000)) == "train-production-to200M"

    def test_a_precompute_names_its_abstraction(self):
        assert kinds.kind(TaskName.PRECOMPUTE).label(_spec(config="ochs")) == "precompute-ochs"


class TestDescribe:
    def test_it_reads_the_kinds_own_fields(self):
        assert kinds.describe({"op": "train", "target_iteration": "5000000"}) == "train ->5M"
        assert (
            kinds.describe(
                {"op": "evaluate", "eval_at": "150000000", "eval_flags": ["--br-board-seed", "7"]}
            )
            == "evaluate @150M seed7"
        )

    def test_a_record_from_before_these_fields_degrades_to_its_op(self):
        """Honest: those records genuinely hold nothing more to show."""
        assert kinds.describe({"op": "evaluate", "target_iteration": "0"}) == "evaluate"


class TestProgress:
    def test_training_reports_against_its_target(self):
        got = kinds.kind(TaskName.TRAIN).sample(_plan(to=150_000_000), {"iteration": 30_000_000})
        assert got == Progress(30_000_000, 150_000_000, "iterations")
        assert got is not None
        assert got.phrase == "30M / 150M iterations"

    def test_a_resumed_run_past_its_target_does_not_overflow_the_bar(self):
        got = kinds.kind(TaskName.TRAIN).sample(_plan(to=1000), {"iteration": 5000})
        assert got is not None
        assert got.fraction == 1.0

    def test_a_build_reports_the_runouts_it_has_enumerated(self):
        """NOT streets. Three of them means a bar that moves twice across hours,
        and they are nowhere near equal -- a canonical flop is 1,176 runouts
        against a river's one."""
        got = kinds.kind(TaskName.PRECOMPUTE).sample(
            _plan(), {"done": 2_000_000, "total": 3_000_000}
        )
        assert got == Progress(2_000_000, 3_000_000, "board runouts")

    def test_both_trainers_prefer_the_live_count_to_the_banked_one(self):
        """The manifest is a rung behind by construction. It is the FLOOR --
        for the window before the trainer's writer starts, and for a task whose
        wrapper predates it -- never the answer when a live count exists."""
        for name in (TaskName.TRAIN, TaskName.TRAIN_VECTOR):
            kind = kinds.kind(name)
            live = kind.sample(_plan(to=1000), {"iteration": 200, "done": 450, "total": 1000})
            assert live == Progress(450, 1000, "iterations"), name
            # And the other way round, so a stale file cannot walk the bar back.
            banked = kind.sample(_plan(to=1000), {"iteration": 700, "done": 450})
            assert banked == Progress(700, 1000, "iterations"), name

    def test_a_kind_with_nothing_yet_to_report_says_nothing(self):
        """None renders as no bar; zero renders as a bar that looks stuck."""
        assert kinds.kind(TaskName.TRAIN).sample(_plan(to=100), {}) is None
        assert kinds.kind(TaskName.PRECOMPUTE).sample(_plan(), {}) is None

    def test_a_build_names_where_it_should_write_its_progress(self):
        """Only the node knows its scratch dir, so the wrapper fills the path in
        and the kind puts it on the command line."""
        (argv,) = kinds.kind(TaskName.PRECOMPUTE).commands(_plan(progress_path="/w/p.json"))
        assert argv[-2:] == ["--progress-file", "/w/p.json"]
        (bare,) = kinds.kind(TaskName.PRECOMPUTE).commands(_plan())
        assert "--progress-file" not in bare

    def test_it_survives_a_record_written_by_an_older_wrapper(self):
        assert Progress.from_record(None) is None
        assert Progress.from_record({"done": 1}) is None
        assert Progress.from_record({"done": 1, "total": 2, "unit": "x"}) == Progress(1, 2, "x")


class TestEstimate:
    """Four sources, best first, and the worker count decides which history counts."""

    KIND = kinds.kind(TaskName.TRAIN)

    def test_a_task_measuring_itself_beats_any_prior(self):
        """50 units in 100s is half a unit a second, so the 50 left are ~100s --
        whatever history claims. It is measuring the machine it actually got, at
        the workers it got."""
        remaining = self.KIND.estimate(
            Progress(50, 100, "x", base=0, window_seconds=100),
            elapsed=100,
            history=[Sample(1, 9999, 16)],
            workers=16,
        )
        assert remaining == pytest.approx(100)

    def test_a_task_continuing_a_run_is_measured_on_its_own_work(self):
        """THE BUG THIS WINDOW EXISTS FOR. A task continuing a 30M run to 60M
        opens at fraction 0.5, and `elapsed x (1 - f) / f` read that as almost
        done from its first second: ~60s left on four more hours of training.

        It has run 100s and moved 1M of the 30M it owes, so it is ~50 minutes in
        and ~48 short of them."""
        remaining = self.KIND.estimate(
            Progress(31_000_000, 60_000_000, "iterations", base=30_000_000, window_seconds=100),
            elapsed=100,
            history=[],
            workers=16,
        )
        assert remaining == pytest.approx(2900)

    def test_a_window_holding_no_work_is_not_extrapolated_from(self):
        """A node spends its first minutes fetching, syncing and loading a
        ~773 MB abstraction. Nothing has moved, so there is no rate here and
        this falls through to history."""
        remaining = self.KIND.estimate(
            Progress(1, 1000, self.KIND.unit, base=1, window_seconds=60),
            elapsed=60,
            history=[Sample(1000, 600, 16)],
            workers=16,
        )
        # 999 iterations left at 1000/600 iterations per second.
        assert remaining == pytest.approx(599.4)

    def test_history_transfers_across_tasks_of_different_size(self):
        """The reason a sample carries units. A 200M task predicted from 5M
        tasks by median DURATION would be absurd; by rate it is right."""
        history = [Sample(units=5_000_000, seconds=1000, workers=16)]
        remaining = self.KIND.estimate(
            Progress(0, 200_000_000, "iterations"), elapsed=0, history=history, workers=16
        )
        assert remaining == pytest.approx(40_000)


class TestEstimateAndWorkers:
    """Throughput scales with workers, so a history that mixes them predicts
    nothing. These pin that the matching happens and that it is matching, not
    dividing."""

    KIND = kinds.kind(TaskName.TRAIN)

    HISTORY: ClassVar[list[Sample]] = [
        Sample(units=1_000_000, seconds=1000, workers=1),  # 1k/s
        Sample(units=1_000_000, seconds=100, workers=16),  # 10k/s
        Sample(units=1_000_000, seconds=90, workers=16),
    ]

    def test_it_predicts_from_tasks_that_ran_at_the_same_width(self):
        remaining = self.KIND.estimate(
            Progress(0, 1_000_000, "iterations"), elapsed=0, history=self.HISTORY, workers=16
        )
        # median of the two 16-worker rates, not of all three.
        assert remaining == pytest.approx(95, rel=0.02)

    def test_a_narrower_task_is_not_predicted_from_a_wider_one(self):
        """The defect this exists to prevent: a 1-worker task told it will
        finish in 95s because sixteen-worker tasks did."""
        remaining = self.KIND.estimate(
            Progress(0, 1_000_000, "iterations"), elapsed=0, history=self.HISTORY, workers=1
        )
        assert remaining == pytest.approx(1000)

    def test_the_two_widths_give_genuinely_different_answers(self):
        wide = self.KIND.estimate(Progress(0, 1e6, "i"), 0, self.HISTORY, workers=16)
        narrow = self.KIND.estimate(Progress(0, 1e6, "i"), 0, self.HISTORY, workers=1)
        assert wide is not None
        assert narrow is not None
        assert narrow > wide * 5

    def test_an_unseen_width_still_gets_an_estimate(self):
        """A rough answer beats none: the first task at a new worker count
        would otherwise never get one."""
        remaining = self.KIND.estimate(
            Progress(0, 1_000_000, "i"), elapsed=0, history=self.HISTORY, workers=8
        )
        assert remaining is not None

    def test_it_matches_on_workers_rather_than_dividing_by_them(self):
        """Scaling is sublinear and saturates -- 16 workers and 32 measured
        within noise past 10M iterations -- so normalising by dividing would be
        confidently wrong exactly where it matters."""
        history = [Sample(units=1_000_000, seconds=100, workers=16)]
        remaining = self.KIND.estimate(
            Progress(0, 1_000_000, "i"), elapsed=0, history=history, workers=32
        )
        # 100s, not 50s: no division happened.
        assert remaining == pytest.approx(100)


class TestEstimateDeclinesToGuess:
    KIND = kinds.kind(TaskName.TRAIN)

    def test_with_no_history_and_no_progress_it_says_nothing(self):
        assert self.KIND.estimate(None, elapsed=10, history=[], workers=16) is None

    def test_a_kind_that_cannot_report_falls_back_to_duration(self):
        """Nothing to scale, because there is no "remaining" without progress."""
        history = [Sample(units=0, seconds=100, workers=0)]
        assert self.KIND.estimate(None, elapsed=10, history=history) == pytest.approx(90)

    def test_an_overdue_task_reports_zero_rather_than_negative(self):
        history = [Sample(units=0, seconds=100, workers=0)]
        assert self.KIND.estimate(None, elapsed=500, history=history) == 0.0

    def test_a_zero_second_sample_cannot_divide_by_zero(self):
        history = [Sample(units=100, seconds=0, workers=16)]
        assert self.KIND.estimate(Progress(0, 100, "x"), 0, history, workers=16) is None

    def test_a_finished_task_has_nothing_left(self):
        remaining = self.KIND.estimate(
            Progress(100, 100, "x", base=0, window_seconds=50), elapsed=50, history=[], workers=16
        )
        assert remaining == pytest.approx(0)


class TestSamplesFromHistory:
    """What a finished task contributes to predicting the next one."""

    @staticmethod
    def _row(**kwargs):
        base = {
            "op": "train",
            "cause": "completed",
            "started_at": "2026-08-05T10:00:00+00:00",
            "ended_at": "2026-08-05T11:00:00+00:00",
            "units": 3_600_000,
            "workers": 16,
        }
        return base | kwargs

    def test_a_completed_task_becomes_a_rate(self):
        (sample,) = kinds.samples([self._row()], "train")
        assert sample.rate == pytest.approx(1000)
        assert sample.workers == 16

    def test_a_task_that_died_partway_is_not_a_rate(self):
        """It took the wall clock of a partial job, and counting it would drag
        every later estimate down."""
        assert kinds.samples([self._row(cause="killed")], "train") == []

    def test_a_record_from_before_units_existed_is_skipped(self):
        """There is no way to reconstruct what those achieved."""
        assert kinds.samples([self._row(units=0)], "train") == []

    def test_another_kinds_history_is_not_borrowed(self):
        assert kinds.samples([self._row(op="evaluate")], "train") == []

    def test_an_unreadable_timestamp_is_skipped_not_fatal(self):
        assert kinds.samples([self._row(ended_at="nonsense")], "train") == []

    def test_they_come_back_oldest_first_whatever_order_they_arrived_in(self):
        """The recency cut downstream is a tail slice, so the order is not a
        detail of whoever joined the rows."""
        newer = self._row(ended_at="2026-08-05T12:00:00+00:00", units=7_200_000)
        got = kinds.samples([newer, self._row()], "train")
        assert [sample.units for sample in got] == [3_600_000, 7_200_000]

    def test_an_estimate_looks_back_over_recent_tasks_only(self):
        """THE STALE-HISTORY BUG. The tree walk got 2.6x faster in one commit,
        and a median over everything the share holds keeps predicting the code
        that was replaced: twenty old tasks at 1,000 it/s outvote every task run
        since, and a 10M task is quoted at ~2.8 hours instead of ~1.

        Five rather than one because throughput varies several-fold between
        boxes on identical code."""
        old = [self._row(units=3_600_000) for _ in range(20)]
        new = [
            self._row(
                units=9_360_000,
                started_at="2026-08-06T10:00:00+00:00",
                ended_at="2026-08-06T11:00:00+00:00",
            )
            for _ in range(5)
        ]
        history = kinds.samples([*old, *new], "train")
        left = kinds.kind(TaskName.TRAIN).estimate(
            Progress(0, 10_000_000, "iterations"), elapsed=0, history=history, workers=16
        )
        assert left == pytest.approx(10_000_000 / 2600)


class TestRemaining:
    """The one call a surface makes."""

    HISTORY: ClassVar[list[dict]] = [
        {
            "op": "train",
            "cause": "completed",
            "started_at": "2026-08-05T10:00:00+00:00",
            "ended_at": "2026-08-05T11:00:00+00:00",
            "units": 3_600_000,
            "workers": 16,
        }
    ]

    def _running(self, **kwargs):
        base = {
            "op": "train",
            "cause": "running",
            "started_at": "2026-08-05T12:00:00+00:00",
            "workers": 16,
            "progress": {
                "done": 1_000_000.0,
                "total": 2_000_000.0,
                "unit": "iterations",
                "base": 0.0,
                "window_seconds": 3600.0,
            },
        }
        return base | kwargs

    def test_a_running_task_gets_an_estimate_from_its_own_rate(self):
        """A million iterations in an hour, a million to go."""
        left = kinds.remaining(self._running(), self.HISTORY, "2026-08-05T13:00:00+00:00")
        assert left == pytest.approx(3600, rel=0.01)

    def test_a_task_with_no_progress_yet_falls_back_to_history(self):
        left = kinds.remaining(
            self._running(progress=None), self.HISTORY, "2026-08-05T12:00:00+00:00"
        )
        assert left == pytest.approx(3600)

    def test_a_finished_task_has_no_estimate_at_all(self):
        """Not zero -- nothing. It is not waiting on anything."""
        assert kinds.remaining(self._running(cause="completed"), self.HISTORY, "x") is None

    def test_a_kind_this_code_no_longer_has_is_not_a_crash(self):
        """A synthetic op, not `vector-sweep`: that one came BACK, and with it
        the reason this test passed had nothing to do with the op."""
        assert kinds.remaining(self._running(op=RETIRED_OP), self.HISTORY, "x") is None


class TestAnEvaluationCanAlwaysBeEstimated:
    """A time estimate matters more here than a moving bar.

    Inside one rung is opaque -- the exact-BR walk is recursive over a public
    tree, so counting would sit in the hot path -- but an evaluation still knows
    how many rungs it was asked for, and that is enough to scale a known rate.
    """

    KIND = kinds.kind(TaskName.EVALUATE)

    def test_it_reports_zero_of_its_rungs_before_the_first_one_lands(self):
        """Not None: the bar sits at 0 for a single-rung score, but the TOTAL is
        what lets an estimate scale to the work actually asked for."""
        got = self.KIND.sample(_plan(eval_rungs=("1000", "2000")), {})
        assert got == Progress(0, 2, "rungs")

    def test_ten_rungs_are_not_predicted_by_the_duration_of_one(self):
        """The reason reporting the total matters. One rung took 600s; ten of
        them is 6000, not 600.

        It scales through the BRANCH total and not the rung count, because a
        rung is not the unit any history is recorded in: `sample` multiplies the
        rungs by the branches each one walks, so the work asked for is stated in
        the same unit the rate is."""
        history = [Sample(units=32, seconds=600, workers=16)]
        branches = self.KIND.unit
        one = self.KIND.estimate(Progress(0, 32, branches), 0, history, workers=16)
        ten = self.KIND.estimate(Progress(0, 320, branches), 0, history, workers=16)
        assert one == pytest.approx(600)
        assert ten == pytest.approx(6000)

    def test_a_rung_count_is_not_divided_by_a_board_branch_rate(self):
        """MEASURED ON THE POOL. Before its first branch lands an evaluation
        reports rungs, and 1 rung over ~500 branches/second is nothing at all:
        the task quoted `0m left` with six minutes to go. A count in one unit
        cannot be scaled by a rate in another, so this falls through to how long
        these tasks TOOK -- 600s, less the 100s elapsed."""
        history = [Sample(units=30, seconds=600, workers=16)]
        left = self.KIND.estimate(Progress(0, 1, "rungs"), elapsed=100, history=history, workers=16)
        assert left == pytest.approx(500)

    def test_it_still_has_an_estimate_with_no_progress_at_all(self):
        """A kind that cannot report position falls back to how long its past
        tasks took -- close enough, and far better than nothing."""
        history = [Sample(units=1, seconds=600, workers=16)]
        assert self.KIND.estimate(None, elapsed=100, history=history) == pytest.approx(500)


class TestAnEvaluationHasARealBar:
    """Rungs were the wrong denominator, and the denominator was the whole bar.

    `score` submits ONE TASK PER RUNG, so `len(eval_rungs)` is 1 — the bar read
    0% from the first second to the last of a ~10-minute score, which makes a
    long evaluation and a hung one look identical. Flop branches are the
    outermost thing the walk counts: four walks of `--br-flops` each.
    """

    def _plan(self, **over):
        return node_plan.TaskPlan(
            op=kinds.TaskName.EVALUATE, run_id="run-x", eval_rungs=("150000000",), **over
        )

    def test_branches_are_the_unit_once_the_walk_reports(self):
        progress = kinds.kind("evaluate").sample(self._plan(), {"done": 8, "total": 32})
        assert progress is not None
        assert (progress.done, progress.total) == (8.0, 32.0)
        assert progress.fraction == pytest.approx(0.25)

    def test_the_denominator_beats_the_rung_count_it_replaced(self):
        """The point of the change, stated as a number."""
        branches = kinds.kind("evaluate").sample(self._plan(), {"done": 0, "total": 32})
        rungs = kinds.kind("evaluate").sample(self._plan(), {})
        assert branches is not None
        assert rungs is not None
        assert branches.total == 32.0
        assert rungs.total == 1.0, "what it was before: one rung, so 0% until the end"

    def test_it_falls_back_to_rungs_before_the_first_branch_lands(self):
        """Not nothing: the rung count is what this reported before, and the
        handler still closes the bar with it at the end."""
        progress = kinds.kind("evaluate").sample(self._plan(), {"scored": 1})
        assert progress is not None
        assert (progress.done, progress.total) == (1.0, 1.0)

    def test_a_zero_total_does_not_divide(self):
        """A torn or just-created progress file must not become a crash or a NaN."""
        progress = kinds.kind("evaluate").sample(self._plan(), {"done": 0, "total": 0})
        assert progress is not None
        assert progress.total == 1.0, "fell through to the rung count"


class TestUnitsCarryTheirUnit:
    """A count is not a measurement without its unit.

    `evaluate` moved from rungs to flop branches. A rung-rate averaged into a
    branch-rate does not fail — it predicts ~30x wrong, silently, which is the
    exact shape of the lineage bugs this project keeps paying for.
    """

    def _row(self, **over):
        return {
            "op": "evaluate",
            "cause": "completed",
            "started_at": "2026-08-06T00:00:00+00:00",
            "ended_at": "2026-08-06T00:10:00+00:00",
            "units": 32.0,
            "workers": 1,
            **over,
        }

    def test_a_row_in_the_current_unit_is_used(self):
        assert len(kinds.samples([self._row(units_unit="board branches")], "evaluate")) == 1

    def test_a_row_in_a_retired_unit_is_skipped(self):
        assert kinds.samples([self._row(units_unit="rungs", units=1.0)], "evaluate") == []

    def test_a_legacy_row_without_one_is_taken_at_face_value(self):
        """Those rows predate the field, and no kind whose unit has changed has
        any of them — measured on the share: zero evaluate rows carry units."""
        assert len(kinds.samples([self._row()], "evaluate")) == 1

    def test_an_unknown_kind_still_yields_its_samples(self):
        """`vector-sweep` has no class here and its history must still count."""
        row = self._row(op="vector-sweep", units_unit="checkpoints")
        assert len(kinds.samples([row], "vector-sweep")) == 1
