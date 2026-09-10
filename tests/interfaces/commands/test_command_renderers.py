"""Every command's payload must survive human-mode rendering.

Renderers are pure formatting, so the only failure they can have is a key the
payload does not carry -- the failure that shipped once, when
a command with no renderer of its own fell through to the evaluate branch and died on
``payload["results"]``. These pin each op's payload shape against its renderer.

"Every subcommand HAS a renderer" is no longer tested: a ``Command`` cannot be
constructed without one, so it is a property of the type rather than a thing to
check.
"""

from typing import Any

import pytest

from src.interfaces.cloud.cost.billing import BilledPayload, ServiceCharge, StandingCharge
from src.interfaces.cloud.cost.node_time import ConcurrencyPoint
from src.interfaces.cloud.tasks.batch import BatchTask, Job, ResizeError

# CONSTRUCTOR CALLS, not dict literals. A literal here is a second declaration
# of a payload's shape and drifts with `contract.py` rather than against it --
# which is how renaming a REQUIRED field passed 1061 tests. A model instance is
# checked by `ty` against the same class the command constructs.
# The remaining dicts are the commands whose payload is not typed yet.
from src.interfaces.commands import (
    evaluate,
    load_all,
)
from src.interfaces.commands.arms import ArmsPayload
from src.interfaces.commands.autoscale_check import AutoscalePayload, AutoscaleView
from src.interfaces.commands.benchmark import BenchmarkPayload
from src.interfaces.commands.benchmark_board import BoardPayload, BoardRow
from src.interfaces.commands.benchmark_drain import DrainPayload
from src.interfaces.commands.blueprint_serve import BlueprintServePayload
from src.interfaces.commands.cancel import CancelledPayload
from src.interfaces.commands.chart import ChartPayload
from src.interfaces.commands.chipzen_seat import ChipzenSeatPayload
from src.interfaces.commands.configs import ConfigKind, ConfigsPayload
from src.interfaces.commands.cost import CostPayload
from src.interfaces.commands.curve import CurvePayload
from src.interfaces.commands.jobs import JobsPayload
from src.interfaces.commands.ledger import LedgerPayload, LedgerRow
from src.interfaces.commands.logs import LogsPayload
from src.interfaces.commands.pool_status import PoolPayload, PoolView
from src.interfaces.commands.precompute import PrecomputePayload
from src.interfaces.commands.profile import ProfilePayload
from src.interfaces.commands.progress import ProgressPayload, ProgressRow
from src.interfaces.commands.prune_checkpoints import PrunePlan
from src.interfaces.commands.push_code import PushedCodePayload
from src.interfaces.commands.reconcile_runs import Closure, ReconcilePlan
from src.interfaces.commands.record_admit import AdmittedPayload
from src.interfaces.commands.record_migrate import MigratedPayload
from src.interfaces.commands.runinfo import RunInfoPayload
from src.interfaces.commands.runs import RunsPayload, RunSummary
from src.interfaces.commands.score import ScorePayload
from src.interfaces.commands.serve import ServePayload
from src.interfaces.commands.serve_box import BoxPayload
from src.interfaces.commands.status import StatusPanel, StatusPayload
from src.interfaces.commands.submit import SubmitPayload
from src.interfaces.commands.submit_precompute import PrecomputeDispatchPayload
from src.interfaces.commands.tasks import TasksPayload
from src.interfaces.commands.train_pcs import PcsTrainingPayload
from src.interfaces.commands.train_static import StaticTrainingPayload
from src.pipeline.services import EvaluationPayload
from src.pipeline.services.experiments import ArmPoint, ArmsOutput, ArmTier, CurveOutput, CurvePoint
from src.shared.task_history import TaskProgress, TaskRow
from src.shared.task_states import Phase

PAYLOADS: dict[str, Any] = {
    "train-static": StaticTrainingPayload(
        run_id="run-a",
        runs_dir="data/runs",
        config_name="quick_test",
        iterations=1000,
        num_rows=32_240_608,
        touched_rows=31_970_418,
        coverage=0.9916,
        mean_visits_per_touched=32.5,
        runtime_seconds=1.5,
        iterations_per_second=666.7,
        dropped_updates=0,
        status="completed",
    ),
    "train-pcs": PcsTrainingPayload(
        run_id="run-pcs-a",
        runs_dir="data/runs",
        config_name="production",
        iterations=9_600,
        board_passes=9_600,
        workers=8,
        num_rows=32_240_608,
        touched_rows=32_100_000,
        coverage=0.9956,
        runtime_seconds=10_800.0,
        iterations_per_second=0.89,
        status="completed",
    ),
    "reconcile-runs": ReconcilePlan(
        runs_considered=301,
        open_runs=28,
        closures=[
            Closure(
                run="run-train-production-to300M-ctrlL-s101-224225-13802",
                status="failed",
                task_id="train-production-to300M-ctrlL-s101-224225-13802",
                cause="killed",
                cause_source="batch",
                ended_at="2026-08-24T22:42:25Z",
            )
        ],
        unsettled=["run-pcs-production-to4k-turn-river-072201-7245"],
        no_evidence=["run-production-025433-1095"],
    ),
    "precompute": PrecomputePayload(
        abstraction_config="production",
        output_dir="data/combo_abstraction/production",
    ),
    "curve": CurvePayload(
        run_id="run-a",
        tier="exact_br",
        points=[
            CurvePoint(
                iteration=1000,
                exploitability_mbb=900.0,
                std_error_mbb=0.0,
                num_hands=0,
                eval_git_commit="abcdef1234",
            ),
            CurvePoint(
                iteration=4000,
                exploitability_mbb=450.0,
                std_error_mbb=0.0,
                num_hands=0,
                eval_git_commit="abcdef1234",
            ),
        ],
        missing_iterations=[8000],
        other_tiers=["lbr/myopic"],
        retained_iterations=[1000, 4000, 8000],
        unplaceable_records=1,
    ),
    "arms": ArmsPayload(
        result=ArmsOutput(
            experiment_id="pcs-weighting",
            tiers=[
                ArmTier(
                    tier="exact_br num_flops=4 num_turns=16 num_rivers=16 seed=7",
                    control="linplus",
                    arms=["dcfr", "linplus"],
                    unmatched_iterations=[4000],
                    points=[
                        ArmPoint(
                            arm="linplus",
                            iteration=2000,
                            exploitability_mbb=900.0,
                            std_error_mbb=0.0,
                            run_id="run-linplus",
                        ),
                        ArmPoint(
                            arm="dcfr",
                            iteration=2000,
                            exploitability_mbb=780.0,
                            std_error_mbb=0.0,
                            run_id="run-dcfr",
                            vs_control_mbb=-120.0,
                        ),
                    ],
                )
            ],
            unplaceable_records=1,
        ),
    ),
    "ledger": LedgerPayload(
        matched=1,
        rows=[
            LedgerRow(
                run_id="run-a",
                eval_git_commit="abcdef1234",
                knobs={"scorer": "lookahead", "opponent": "blueprint", "base_seed": 1},
                results={"exploitability_mbb": 900.0, "std_error_mbb": 12.0, "num_hands": 1000},
            )
        ],
    ),
    "evaluate": EvaluationPayload(
        run_id="run-a",
        # `method` was absent from this fixture for as long as it existed; the
        # payload has always carried it.
        method="lbr",
        estimator="local_best_response",
        infosets=42,
        results={"exploitability_mbb": 900.0, "std_error_mbb": 12.0},
    ),
    "runinfo": RunInfoPayload(
        run_id="run-a",
        config_name="production",
        status="completed",
        experiment_id="exp-7",
        arm="control",
        parent_run_id=None,
        git_commit="cafebabe" * 5,
        git_dirty=False,
        card_abstraction_hash="ae5a7e6648d7cd02",
        iterations=30_000_000,
        runtime_seconds=9000.0,
        # The digest's own word, which `training_tasks` renames for the reader.
        attempts=4,
        training_tasks=4,
        trainer_knobs={
            "solver": {"cfr_plus": False, "iteration_weighting": "dcfr", "dcfr_gamma": 2.0},
            "pcs": {"cfr_br": "river", "runouts_per_flop": 1},
        },
        total_progress_rows=2,
        progress=[
            {
                "iteration": 1_000_000,
                "coverage": 0.08,
                "mean_visits_per_touched": 2.1,
                "iters_per_sec": 1204.0,
            },
            {
                "iteration": 30_000_000,
                "coverage": 0.287,
                "mean_visits_per_touched": 11.4,
                "iters_per_sec": 980.0,
            },
        ],
        coverage_flat_from=20_000_000,
        curve=CurveOutput(
            run_id="run-a",
            tier="exact_br flops=8",
            points=[
                CurvePoint(
                    iteration=10_000_000,
                    exploitability_mbb=1800.0,
                    std_error_mbb=0.0,
                    num_hands=0,
                    eval_git_commit=None,
                ),
            ],
            missing_iterations=[5_000_000, 20_000_000],
            other_tiers=[],
            retained_iterations=[10_000_000],
            unplaceable_records=0,
        ),
        tasks=[TaskRow(task_id="prod-101010-1", attempt=1, cause="killed", cause_source="batch")],
        gaps=["unscored ladder rungs: 5,000,000, 20,000,000"],
    ),
    "record-admit": AdmittedPayload(server="poker-solver-record", address="203.0.113.7"),
    "record-migrate": MigratedPayload(before="", after="9eb7f485ecad", applied=True),
    "serve-box": BoxPayload(
        action="status",
        vm="blueprint-server",
        resource_group="poker-solver-serve-rg",
        power="deallocated",
        usable=False,
        location="swedencentral",
    ),
    "blueprint-serve": BlueprintServePayload(
        run="run-production-025433-1095",
        run_dir="/mnt/work/runs/run-production-025433-1095",
        # `runs_dir` was absent while this was a literal, and `render` reads it:
        # nothing noticed, because this command is in SIDE_EFFECTING and its
        # renderer was never called.
        runs_dir="/mnt/work/runs",
        at_iteration=None,
        idle_timeout=1800,
        url="http://127.0.0.1:8790",
        host="127.0.0.1",
        port=8790,
    ),
    # Replay, not live: holding a seat never returns, so only this mode has a
    # payload a renderer can be handed. The warning fields are set so the
    # renderer's three conditional lines are the ones under test.
    # Two trained classes and one never visited, so the renderer's blank cell and
    # its shaded ones are both exercised.
    "chart": ChartPayload(
        run="run-production-025433-1095",
        path="",
        actor=0,
        actions=["f", "c", "r200", "A"],
        trained=2,
        untrained=167,
        rows={
            "AA": {
                "strategy": [0.0, 0.0, 0.6, 0.4],
                "aggression": 1.0,
                "fold": 0.0,
                "passive": 0.0,
                "reach_count": 41234,
            },
            "72o": {
                "strategy": [0.9, 0.1, 0.0, 0.0],
                "aggression": 0.0,
                "fold": 0.9,
                "passive": 0.1,
                "reach_count": 118,
            },
        },
    ),
    "chipzen-seat": ChipzenSeatPayload(
        run="run-production-025433-1095",
        run_dir="/mnt/work/runs/run-production-025433-1095",
        runs_dir="/mnt/work/runs",
        mode="replay",
        resolver=False,
        budget_ms=1200,
        decision={"action": "raise", "params": {"amount": 60}},
        depth_matches=False,
        their_depth=50.0,
        our_depth=100.0,
        off_tree=2,
        truncated=True,
    ),
    "serve": ServePayload(
        url="http://127.0.0.1:8765",
        host="127.0.0.1",
        port=8765,
        reload=False,
    ),
    "cost": CostPayload(
        hours=0.0,
        task_hours=68.87,
        tasks=47,
        peak_concurrency=4,
        unended=4,
        first_at="2026-08-02T22:06:18+00:00",
        last_at="2026-08-03T16:58:59+00:00",
        rate_per_node_hour=0.688,
        dollars=47.38,
        series=[
            ConcurrencyPoint(at="2026-08-02T22:06:18+00:00", running=1),
            ConcurrencyPoint(at="2026-08-02T22:08:22+00:00", running=3),
            ConcurrencyPoint(at="2026-08-03T16:58:59+00:00", running=0),
        ],
        billed_reason=None,
        # Present here rather than null, because the null case is the easy one:
        # the renderer that has to be pinned is the one with an invoice to show.
        billed=BilledPayload(
            total=328.94,
            other=90.62,
            currency="USD",
            pool_cost=214.90,
            pool_node_hours=313.50,
            # A standing VM in the fixture on purpose: folding one into pool
            # compute is the bug this shape exists to prevent.
            standing_cost=23.42,
            standing_hours=68.03,
            standing=[
                StandingCharge(resource_group="poker-solver-serve-rg", hours=67.83, cost=23.33),
            ],
            since="2025-08-10",
            first_at="2026-07-27",
            as_of="2026-08-08",
            by_service=[
                ServiceCharge(service="Virtual Machines", cost=238.32),
                ServiceCharge(service="Storage", cost=86.19),
                ServiceCharge(service="Load Balancer", cost=3.41),
                ServiceCharge(service="Virtual Network", cost=0.87),
            ],
        ),
    ),
    "runs": RunsPayload(
        runs=[
            RunSummary(
                name="run-production-025433-1095",
                commits_ago=None,
                git_dirty=False,
                has_checkpoint=True,
                loadable=True,
                blocker=None,
                iterations=76_000_000,
                num_infosets=32_240_608,
                config_name="production",
                status="completed",
                experiment_id="exp-7",
                arm="control",
            ),
            # A run that never checkpointed: still listed, with the reason, because
            # it is exactly the one someone is looking for when asking what happened.
            RunSummary(
                name="run-20260802_203312-8c4a2c",
                commits_ago=3,
                git_dirty=True,
                has_checkpoint=False,
                loadable=False,
                blocker="never checkpointed",
                iterations=None,
                num_infosets=None,
                config_name="quick_test",
                status="failed",
                experiment_id=None,
                arm=None,
            ),
        ],
    ),
    "tasks": TasksPayload(
        reconciled=1,
        rows=[
            TaskRow(
                task_id="train-production-1095-to150M-090456-1",
                attempt=1,
                op="train",
                what="train ->150M",
                run_id="run-production-025433-1095",
                cause="running",
                cause_source="node",
                started_at="2026-08-05T09:04:56+00:00",
                # A running task's bar. Absent on a finished one -- see below.
                progress=TaskProgress(done=38000000, total=150000000, unit="iterations"),
                workers=16,
                # Seconds left, derived at read time from this task's own rate
                # and the history of tasks that ran at the same width.
                eta_seconds=9540.0,
            ),
            TaskRow(
                task_id="prod-101010-1",
                attempt=1,
                op="train-static",
                run_id="run-a",
                cause="killed",
                cause_source="batch",
                exit_code=137,
                ended_at="2026-08-02T10:00:00Z",
            ),
        ],
    ),
    # The listing, not the ask: `--list` is the shape the console polls, and the
    # one where an empty answer still has to render.
    "profile": ProfilePayload(available=["run-a-task.0.1.speedscope.json"]),
    "progress": ProgressPayload(
        run_id="run-a",
        total_rows=2,
        schema_version_min=1,
        schema_version_max=1,
        coverage_plateau_iteration=2000000,
        rows=[
            # `elapsed_s`, `touched_rows`, `num_rows` and `dropped_updates` are
            # not declared on `ProgressRow` and ride through on `extra="allow"`:
            # a progress row is a RECORD off `progress.jsonl`, not something a
            # command constructs, and the model names only what a surface reads.
            ProgressRow.model_validate(
                {
                    "schema_version": 1,
                    "iteration": 1000000,
                    "elapsed_s": 900.0,
                    "iters_per_sec": 1111.0,
                    "touched_rows": 1000,
                    "num_rows": 10000,
                    "coverage": 0.1,
                    "mean_visits_per_touched": 4.2,
                    "dropped_updates": 0,
                    "checkpoint_seconds": 12.5,
                }
            ),
            # A row from before a field existed: the renderer must blank it, not
            # crash and not print a placeholder that reads as a real measurement.
            ProgressRow(schema_version=0, iteration=2000000, coverage=0.101),
        ],
    ),
    "jobs": JobsPayload(
        jobs=[
            Job(
                job="poker-20260802",
                state="BatchJobState.ACTIVE",
                # One of each half the pool has: a task OCCUPYING a node, and
                # one waiting for one. The queued shape is the one that carries
                # no node and no start time, so a sample without it would let a
                # reader assume both are always present.
                tasks=[
                    BatchTask(
                        task="train-production-to150M-090456-1",
                        job="poker-20260802",
                        state="BatchTaskState.RUNNING",
                        phase=Phase.RUNNING,
                        node="tvmps_x",
                        created="2026-08-04T09:04:56+00:00",
                        start_time="2026-08-04T09:07:12+00:00",
                    ),
                    BatchTask(
                        task="score-production-1095-150M-seed7-090501-2",
                        job="poker-20260802",
                        state="BatchTaskState.ACTIVE",
                        phase=Phase.QUEUED,
                        created="2026-08-04T09:05:01+00:00",
                    ),
                ],
            )
        ],
        total_jobs=3,
        hidden_jobs=2,
    ),
    "logs": LogsPayload(
        task="production-1",
        lines=["train-static: config=production", "publish complete"],
    ),
    # A resize error whose real cause is escaped JSON inside a value. This is
    # the shape the renderer exists to unpack -- Batch reports the generic
    # `AllocationFailed` and hides the actionable half in here.
    "pool-status": PoolPayload(
        total_nodes=2,
        burn_per_hour=1.376,
        pools=[
            PoolView(
                pool_id="train",
                hourly_cost="$0.688/hr/node",
                burn_per_hour=0.688,
                allocation_state="steady",
                current_dedicated_nodes=1,
                target_dedicated_nodes=1,
                vm_size="standard_d16als_v6",
                resize_errors=[
                    ResizeError(
                        code="AllocationFailed",
                        message="Desired number of dedicated nodes could not be allocated",
                        values={"ErrorJson": '{"code":"AllocationFailed"}', "Plain": "not json"},
                    )
                ],
            ),
            PoolView(
                pool_id="train-big",
                hourly_cost="$1.376/hr/node",
                burn_per_hour=0.688,
                allocation_state="steady",
                current_dedicated_nodes=1,
                target_dedicated_nodes=1,
                vm_size="standard_d32als_v6",
            ),
        ],
    ),
    # The probe, matching its published figure. `within_expectation=False` is
    # not a payload that reaches a renderer: `run` refuses instead, so the
    # renderer stays pure formatting.
    "benchmark": BenchmarkPayload(
        agent="check-call",
        game="HUNL 200BB",
        hands_played=2000,
        hands_failed=1,
        aivat_bb_per_100=-183.4,
        aivat_std_bb_per_100=4.7,
        raw_bb_per_100=-241.0,
        off_tree_per_hand=1.5,
        clamped_per_hand=0.4,
        truncated_hands=2,
        expected="check-call",
        within_expectation=True,
        z_score=0.1,
    ),
    # One slot given back and one that would not close: a drain that reports
    # every hand it looked at as released is the failure this guards.
    "benchmark-drain": DrainPayload(
        game="HUNL 200BB",
        open_hands=[3321480, 3326317],
        released=[3321480],
    ),
    "benchmark-board": BoardPayload(
        game="HUNL 200BB",
        version=2,
        rows=[
            BoardRow(
                bot_name="Bitcrumbs",
                organization="Individual",
                version=2,
                hands=44717,
                aivat_bb_per_100=-3.11,
                aivat_std_bb_per_100=0.99,
                raw_bb_per_100=-9.15,
            ),
            # An entry with no organisation renders as a bare name, not as
            # "name ()".
            BoardRow(
                bot_name="testbot",
                organization="",
                version=2,
                hands=100659,
                aivat_bb_per_100=-19.43,
                aivat_std_bb_per_100=0.74,
                raw_bb_per_100=-24.45,
            ),
        ],
    ),
    "configs": ConfigsPayload(
        root="/repo/config",
        kinds=[
            ConfigKind(kind="training", flag="submit --config", names=["production", "quick_test"]),
            # An empty group is a real state (a checkout without the directory)
            # and must render as one rather than as a missing section.
            ConfigKind(kind="abstraction", flag="submit-precompute --config"),
        ],
    ),
    "autoscale-check": AutoscalePayload(
        results=[
            AutoscaleView(
                pool_id="train",
                formula="$TargetDedicatedNodes = min(maxNodes, pending);",
                variables={"$TargetDedicatedNodes": "0", "pending": "0"},
            ),
            AutoscaleView(
                pool_id="train-big",
                formula="$TargetDedicatedNodes = min(maxNodes, pending);",
                variables={"$TargetDedicatedNodes": "2", "pending": "19"},
            ),
        ]
    ),
    "submit": SubmitPayload(
        target_iteration=25_000_000,
        code_snapshot="code-20260802_000000",
        job_id="poker-20260802",
        tasks=["production-000000-1"],
    ),
    "score": ScorePayload(
        run_id="run-a",
        method="exact_br",
        rungs=["10000000", "20000000"],
        code_snapshot="code-20260802_000000",
        job_id="poker-20260802",
        tasks=["run-a-000000-1", "run-a-000000-2"],
    ),
    "cancel": CancelledPayload(job_id="poker-20260802", task_id="run-a-000000-1"),
    "push-code": PushedCodePayload(code_snapshot="code-20260802_000000"),
    "submit-precompute": PrecomputeDispatchPayload(
        abstraction_config="production",
        target_name="buckets-F20T20R20-rexact-deadbeef",
        already_published=["buckets-F50T100R200-rexact-b59ef7b2"],
        force=False,
        code_snapshot="code-20260802_000000",
        job_id="poker-20260802",
        tasks=["production-000000-1"],
    ),
    # The applied-and-deleted shape, because it is the one with something to
    # report: a dry run renders a subset of these keys.
    # The APPLIED shape, for the same reason `compact-legs` uses it: a dry run
    # renders a subset. `protected` and `scored_kept` are populated because both
    # are lines a reader has to see before trusting a delete.
    "prune-checkpoints": PrunePlan(
        applied=True,
        runs_considered=272,
        runs_affected=1,
        rungs_dropped=55,
        files_deleted=8_360,
        freed_gib=77.0,
        protected=["run-pcs-production-to1-2k-river-023543-3910: still running"],
        plan=[
            {
                "run": "run-train-production-to100M-production_f4-081936-25408",
                "held": 64,
                "drop": [5_000_000],
                "dropping": 55,
                "keeping": [400_000_000],
                "scored_kept": [400_000_000],
                "gib_each": 1.4,
                "gib_freed": 77.0,
            }
        ],
    ),
}


# Composed from the panels' own fixtures, which is the point of the command:
# it renders each panel with the renderer that owns it rather than formatting
# anything itself. Built after the literal so it can reuse them by key.
#
# `tasks` is deliberately the FAILED panel here. A status screen's whole value is
# that it still shows the other two when one is unavailable, and that path only
# runs when something is already wrong -- so it is the one worth pinning.
PAYLOADS["status"] = StatusPayload(
    at="2026-08-03T00:24:48-07:00",
    elapsed_seconds=22.1,
    limit=10,
    # DUMPED, because that is what a panel actually holds: `_compose._answer`
    # serialises each part so a view can join over plain data. Embedding the
    # models here instead made this fixture agree with itself and with nothing
    # else -- `status` crashed in production on exactly this difference while
    # rendering the fixture cleanly.
    panels={
        "pool": StatusPanel(payload=PAYLOADS["pool-status"].model_dump()),
        "jobs": StatusPanel(payload=PAYLOADS["jobs"].model_dump()),
        "tasks": StatusPanel(error="Azure rejected the credential — try `az login`."),
    },
)

BY_NAME = {command.name: command for command in load_all()}


# These two renderers ARE the server: calling one blocks on uvicorn. They are the
# commands whose render has a side effect rather than being pure formatting, so
# they are excluded here by name and covered by `tests/interfaces/web/` and
# `tests/interfaces/blueprint/` instead.
# These two renderers ARE the server: `render` prints one line and then blocks
# in `uvicorn.run`. They are skipped by the loop below and driven individually
# in `TestTheServersStillFormatTheirPayload`, with the server stubbed -- which
# is the part that matters, and the part that was NOT covered when the comment
# here claimed `tests/interfaces/web/` did it. Nothing called either renderer,
# so both kept subscripting a payload that had become a model, and
# `poker-solver serve` died before uvicorn.
SIDE_EFFECTING = {"serve", "blueprint-serve"}


class TestEveryOpRenders:
    def test_every_command_renders_its_payload(self, capsys):
        for name, command in BY_NAME.items():
            if name in SIDE_EFFECTING:
                continue
            command.render(PAYLOADS[name])
            assert capsys.readouterr().out, f"'{name}' rendered nothing"

    def test_every_command_has_a_fixture(self):
        """A command with no fixture here is a command nothing pins."""
        assert set(BY_NAME) == set(PAYLOADS)

    def test_a_new_command_does_not_borrow_the_evaluate_renderer(self, capsys):
        # The original regression's shape: a command with no renderer of its own
        # fell through to the evaluate branch and died on payload["results"].
        BY_NAME["runinfo"].render(PAYLOADS["runinfo"])
        out = capsys.readouterr().out
        assert "run-a" in out
        assert "Evaluation complete" not in out

    def test_progress_blanks_fields_a_legacy_row_predates(self, capsys):
        """A resumed run appends across tasks, so one log spans code versions."""
        BY_NAME["progress"].render(PAYLOADS["progress"])
        out = capsys.readouterr().out
        assert "2,000,000" in out, "the legacy row must still be shown"
        assert "10.1%" in out, "and the fields it does carry must render"


class TestTheServersStillFormatTheirPayload:
    """The two renderers the loop above cannot call, called anyway.

    `serve` and `blueprint-serve` block in `uvicorn.run`, so they are skipped by
    `TestEveryOpRenders` -- and that skip is why both went on subscripting a
    payload that had become a model. `poker-solver serve` (and `just console`,
    the only way to start it) died with `TypeError: 'ServePayload' object is not
    subscriptable` BEFORE uvicorn, and `--json` still worked, so nothing
    noticed. `blueprint-serve` is worse: `infra/serve/main.tf` invokes it
    without `--json`, so the unit failed and the box deallocated while
    `/api/box/start` reported success.

    Stubbing the server is the whole trick -- everything before it is ordinary
    formatting, and that is the part that broke.
    """

    def test_serve_prints_where_it_will_listen(self, monkeypatch, capsys):
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        BY_NAME["serve"].render(PAYLOADS["serve"])
        assert "http://127.0.0.1:8765" in capsys.readouterr().out

    def test_blueprint_serve_reads_every_field_before_it_loads(self, monkeypatch, capsys):
        """It reads `run_dir` and `runs_dir` before touching a blueprint.

        Stopped at the load rather than the listen: building one is ~a minute
        off the share. Reaching the refusal proves the payload access above it.
        """
        import uvicorn

        from src.adapters.postgres import connect

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        monkeypatch.setattr(connect, "record_source_from_environment", lambda: None)
        with pytest.raises(Exception, match=r"run-production|No such file|not found|Errno"):
            BY_NAME["blueprint-serve"].render(PAYLOADS["blueprint-serve"])


class TestResolverMatchRenderer:
    """The resolver gate reports a chip edge, not exploitability.

    Its payload carries no `exploitability_mbb` at all, and its
    `confidence_95_mbb` is an INTERVAL rather than a half-width. Both facts
    broke this renderer on first contact with a real run -- the second one as a
    `TypeError: unsupported format string passed to tuple.__format__`, six
    minutes into a paid box. A shape example here costs nothing and catches it
    in milliseconds, which is the whole argument for `PAYLOADS`.
    """

    PAYLOAD = EvaluationPayload(
        run_id="run-a",
        method="resolver_match",
        estimator="resolver_match (duplicate-deal chip edge)",
        infosets=32_240_608,
        results={
            "resolver_mbb_per_hand": 12.5,
            "se_mbb": 4.0,
            "confidence_95_mbb": (4.7, 20.3),
            "p_value": 0.0018,
            "num_deals": 1000,
            "num_hands": 2000,
            "resolver_decisions": 5000,
            "resolver_fallbacks": 3,
            "leaf_continuation_fraction": 0.5,
            "resolver_max_iterations": 64,
        },
    )

    def test_it_renders_the_interval_without_dying(self, capsys):
        evaluate.render(self.PAYLOAD)
        out = capsys.readouterr().out
        assert "+12.50 mbb/hand" in out
        assert "+4.70..+20.30" in out

    def test_it_shows_the_arm_and_the_fallback_count(self, capsys):
        """Two numbers that decide whether the result means anything: which arm
        produced it, and whether the resolver actually resolved."""
        evaluate.render(self.PAYLOAD)
        out = capsys.readouterr().out
        assert "0.5 pot" in out
        assert "fell back" in out
        assert "5,000" in out
