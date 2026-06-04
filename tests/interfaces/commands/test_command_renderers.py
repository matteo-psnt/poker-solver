"""Every command's payload must survive human-mode rendering.

Renderers are pure formatting, so the only failure they can have is a key the
payload does not carry -- the failure that shipped once, when
a command with no renderer of its own fell through to the evaluate branch and died on
``payload["results"]``. These pin each op's payload shape against its renderer.

"Every subcommand HAS a renderer" is no longer tested: a ``Command`` cannot be
constructed without one, so it is a property of the type rather than a thing to
check.
"""

import copy
from typing import Any

from src.interfaces.cloud.cost.billing import BilledPayload, ServiceCharge, StandingCharge
from src.interfaces.cloud.cost.node_time import ConcurrencyPoint
from src.interfaces.cloud.tasks.batch import BatchTask, Job, ResizeError
from src.interfaces.commands import load_all
from src.interfaces.commands.activity import ActivityPayload, CommandActivity, Failure
from src.interfaces.commands.autoscale_check import AutoscalePayload
from src.interfaces.commands.cancel import CancelledPayload
from src.interfaces.commands.compact_legs import CompactedPayload
from src.interfaces.commands.compare import ComparePayload, PairedComparison
from src.interfaces.commands.configs import ConfigKind, ConfigsPayload
from src.interfaces.commands.cost import CostPayload
from src.interfaces.commands.curve import CurvePayload
from src.interfaces.commands.jobs import JobsPayload
from src.interfaces.commands.ledger import LedgerPayload, LedgerRow
from src.interfaces.commands.logs import LogsPayload
from src.interfaces.commands.pool_status import PoolPayload
from src.interfaces.commands.precompute import PrecomputePayload
from src.interfaces.commands.progress import ProgressPayload, ProgressRow
from src.interfaces.commands.promote import PromotedPayload
from src.interfaces.commands.push_code import PushedCodePayload
from src.interfaces.commands.push_data import PushedDataPayload
from src.interfaces.commands.report import ArmResult, ReportPayload
from src.interfaces.commands.runinfo import RunInfoPayload
from src.interfaces.commands.runs import RunsPayload, RunSummary
from src.interfaces.commands.score import ScorePayload
from src.interfaces.commands.serve_box import BoxPayload
from src.interfaces.commands.status import StatusPanel, StatusPayload
from src.interfaces.commands.submit import SubmitPayload
from src.interfaces.commands.submit_precompute import PrecomputeDispatchPayload
from src.interfaces.commands.submit_vector import SubmitVectorPayload, VectorArm
from src.interfaces.commands.tasks import TasksPayload
from src.interfaces.commands.train_static import StaticTrainingPayload
from src.interfaces.commands.train_vector import VectorBlueprintPayload
from src.interfaces.commands.vector_sweep import SweepPoint, VectorSweepPayload
from src.interfaces.commands.warm_start import WarmStartPayload
from src.pipeline.services import EvaluationPayload
from src.pipeline.services.experiments import CurveOutput, CurvePoint
from src.shared.task_history import TaskProgress, TaskRow
from src.shared.task_states import Phase

# CONSTRUCTOR CALLS, not dict literals. A literal here is a second declaration
# of a payload's shape and drifts with `contract.py` rather than against it --
# which is how renaming a REQUIRED field passed 1061 tests. A model instance is
# checked by `ty` against the same class the command constructs.
# The remaining dicts are the commands whose payload is not typed yet.
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
    "vector-sweep": VectorSweepPayload(
        abstraction="buckets-F100T300R600-rexact-a1542e88",
        buckets={"flop": 100, "turn": 300, "river": 600},
        kernel="board-free",
        derive_boards=6000,
        train_boards=8,
        score_boards=32,
        in_sample=False,
        stack=20,
        nodes=2140,
        infoset_rows=1_132_552,
        derive_seconds=457.0,
        uniform_baseline=4.1869,
        uniform_baseline_unconstrained=4.4021,
        done=2,
        total=9,
        points=[
            SweepPoint(
                iterations=400,
                train_seconds=53.1,
                exploitability=0.5392,
                unconstrained=0.8811,
            ),
            SweepPoint(
                iterations=1600,
                train_seconds=210.1,
                exploitability=0.693,
                unconstrained=0.9902,
            ),
        ],
        best_exploitability=0.5392,
        best_at_iterations=400,
    ),
    "submit-vector": SubmitVectorPayload(
        arms=[
            VectorArm(
                abstraction="buckets-F10T20R30-r200-ae5a7e66",
                kernel="board-free",
                derive_boards=6000,
            ),
            VectorArm(
                abstraction="buckets-F10T20R30-r200-ae5a7e66",
                kernel="hand-space",
                derive_boards=0,
            ),
        ],
        code_snapshot="code-20260805_000000",
        job_id="poker-20260805",
        tasks=["vector-board-free-buckets-F10T20R30-000000-1"],
    ),
    "train-vector": VectorBlueprintPayload(
        run_id="vec-a",
        runs_dir="data/runs",
        config_name="production",
        iterations=400,
        num_rows=32_240_608,
        touched_rows=32_240_608,
        coverage=1.0,
        runtime_seconds=1800.0,
        seconds_per_iteration=4.5,
        abstract_exploitability=1.16,
        universe_boards=2000,
        universe_seed=7,
        dtype="float32",
        status="completed",
    ),
    "warm-start": WarmStartPayload(
        run_id="warm-a",
        runs_dir="data/runs",
        config_name="production",
        source_run_id="vec-a",
        effective_iterations=1000,
        num_rows=32_240_608,
        seeded_rows=32_240_608,
        seeded_fraction=1.0,
        status="seeded",
    ),
    "precompute": PrecomputePayload(
        abstraction_config="production",
        output_dir="data/combo_abstraction/production",
    ),
    "promote": PromotedPayload(
        run_id="run-a",
        rationale="best so far",
        promoted_at="2026-07-30T00:00:00+00:00",
        checkpoint_iteration=25_000,
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
    "report": ReportPayload(
        experiment_id="exp-1",
        control_run_id="run-c",
        baseline_run_id="run-b",
        notes=["Tier: exact_br"],
        arms=[
            ArmResult(
                arm="control",
                run_id="run-c",
                checkpoint_iteration=1000,
                exploitability_mbb=900.0,
                std_error_mbb=1.0,
                git_branch="main",
                vs_control_mbb=None,
                vs_control_p_value=None,
                vs_control_blocked=[],
            ),
            ArmResult(
                arm="variant:pruning",
                run_id="run-v",
                checkpoint_iteration=1000,
                exploitability_mbb=880.0,
                std_error_mbb=1.0,
                git_branch="worktree-hybrid-kernels",
                vs_control_mbb=-20.0,
                vs_control_p_value=0.01,
                vs_control_blocked=["payload missing, cannot pair"],
            ),
        ],
    ),
    "ledger": LedgerPayload(
        ledger="data/eval_ledger.jsonl",
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
    "compare": ComparePayload(
        run_a="run-a",
        run_b="run-b",
        tier_warnings=["base_seed differs"],
        comparison=PairedComparison(
            # `n` and `t_statistic` were absent from this fixture for as long as
            # it was hand-written, and the real statistic has always returned
            # both.
            n=1000,
            t_statistic=-4.0,
            mean_a=900.0,
            mean_b=880.0,
            mean_diff=-20.0,
            se_diff=5.0,
            ci_lower=-30.0,
            ci_upper=-10.0,
            p_value=0.01,
            is_significant=True,
            correlation=0.8,
            se_unpaired=9.0,
        ),
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
    "serve-box": BoxPayload(
        action="status",
        vm="blueprint-server",
        resource_group="poker-solver-serve-rg",
        power="deallocated",
        usable=False,
        location="swedencentral",
    ),
    "blueprint-serve": {
        "op": "blueprint-serve",
        "run": "run-production-025433-1095",
        "run_dir": "/mnt/work/runs/run-production-025433-1095",
        "at_iteration": None,
        "idle_timeout": 1800,
        "url": "http://127.0.0.1:8790",
        "host": "127.0.0.1",
        "port": 8790,
    },
    "serve": {
        "op": "serve",
        "url": "http://127.0.0.1:8765",
        "host": "127.0.0.1",
        "port": 8765,
        "reload": False,
    },
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
        pool_id="train",
        hourly_cost="$0.80/hr/node",
        allocation_state="AllocationState.STEADY",
        current_dedicated_nodes=0,
        target_dedicated_nodes=0,
        vm_size="standard_d16als_v6",
        resize_errors=[
            ResizeError(
                code="AllocationFailed",
                message="Desired number of dedicated nodes could not be allocated",
                values={"ErrorJson": '{"code":"AllocationFailed"}', "Plain": "not json"},
            )
        ],
    ),
    "activity": ActivityPayload(
        log="/home/me/.cache/poker-solver/telemetry/invocations.jsonl",
        exists=True,
        enabled=True,
        days=7.0,
        failures_only=False,
        rows=412,
        total_rows=5031,
        first_at="2026-08-04T09:00:00+00:00",
        commands=[
            CommandActivity(
                command="tasks",
                calls=180,
                p50_seconds=2.05,
                p95_seconds=9.4,
                max_seconds=23.1,
                total_seconds=512.7,
                refusals=0,
                errors=2,
            ),
            CommandActivity(
                command="pool-status",
                calls=210,
                p50_seconds=0.4,
                p95_seconds=1.1,
                max_seconds=1.4,
                total_seconds=92.3,
                refusals=0,
                errors=0,
            ),
        ],
        failures=[
            Failure(
                at="2026-08-10T22:14:03+00:00",
                command="runinfo",
                surface="console",
                outcome="refusal",
                error_type="CommandError",
                error="'run-x' is not published.",
                asked={"run": "run-x"},
            )
        ],
        total_failures=41,
        by_surface={"cli": 96, "console": 316},
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
        pool_id="train", variables=["$TargetDedicatedNodes=0", "pending=0"]
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
    "push-data": PushedDataPayload(uploaded={"buckets-F50T100R200": 9}),
    # The applied-and-deleted shape, because it is the one with something to
    # report: a dry run renders a subset of these keys.
    "compact-legs": CompactedPayload(
        bundle="sealed.bundle.json",
        files_before=375,
        files_after=55,
        movable=321,
        carried=54,
        attempts=141,
        applied=True,
        verified=True,
        deleted=321,
        backup="/home/me/legs-backup",
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


class TestReportNamesTheWorktree:
    """`report --experiment` pairs arms; it has to say which arm each row IS.

    Arms are developed in parallel worktrees that share a commit and differ only
    in what is uncommitted, so the branch is the only thing distinguishing two
    rows whose provenance is otherwise identical.
    """

    def test_the_branch_is_printed_beside_the_arm(self, capsys):
        BY_NAME["report"].render(PAYLOADS["report"])
        out = capsys.readouterr().out
        assert "worktree-hybrid-kernels" in out
        assert "from" in out

    def test_a_report_of_legacy_arms_drops_the_column(self, capsys):
        """Not a column of em-dashes: those are wide enough to push the numbers
        off a terminal, to say nothing."""
        payload = copy.deepcopy(PAYLOADS["report"])
        for arm in payload.arms:
            arm.git_branch = ""

        BY_NAME["report"].render(payload)
        out = capsys.readouterr().out

        assert "from" not in out
        assert "900.0" in out, "the numbers must still be there"
