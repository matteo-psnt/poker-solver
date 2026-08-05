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
import dataclasses

from src.interfaces.commands import COMMANDS
from src.pipeline import services

PAYLOADS: dict[str, dict] = {
    "train-static": {
        "op": "train-static",
        "run_id": "run-a",
        "runs_dir": "data/runs",
        "config_name": "quick_test",
        "iterations": 1000,
        "num_rows": 32_240_608,
        "touched_rows": 31_970_418,
        "coverage": 0.9916,
        "mean_visits_per_touched": 32.5,
        "runtime_seconds": 1.5,
        "iterations_per_second": 666.7,
        "dropped_updates": 0,
        "status": "completed",
    },
    "precompute": {
        "op": "precompute",
        "abstraction_config": "production",
        "output_dir": "data/combo_abstraction/production",
    },
    "promote": {
        "op": "promote",
        "run_id": "run-a",
        "rationale": "best so far",
        "promoted_at": "2026-07-30T00:00:00+00:00",
        "checkpoint_iteration": 25_000,
        "baseline": "data/baseline.json",
    },
    "curve": {
        "op": "curve",
        "run_id": "run-a",
        "tier": "exact_br",
        "points": [
            {
                "iteration": 1000,
                "exploitability_mbb": 900.0,
                "std_error_mbb": 0.0,
                "num_hands": 0,
            },
            {
                "iteration": 4000,
                "exploitability_mbb": 450.0,
                "std_error_mbb": 0.0,
                "num_hands": 0,
            },
        ],
        "missing_iterations": [8000],
        "other_tiers": ["lbr/myopic"],
        "retained_iterations": [1000, 4000, 8000],
        "unplaceable_records": 1,
        "decay_ratio": 2.0,
    },
    "report": {
        "op": "report",
        "experiment_id": "exp-1",
        "control_run_id": "run-c",
        "baseline_run_id": "run-b",
        "notes": ["Tier: exact_br"],
        "arms": [
            {
                "arm": "control",
                "run_id": "run-c",
                "checkpoint_iteration": 1000,
                "exploitability_mbb": 900.0,
                "std_error_mbb": 1.0,
                "git_branch": "main",
                "vs_control_mbb": None,
                "vs_control_p_value": None,
                "vs_control_blocked": [],
            },
            {
                "arm": "variant:pruning",
                "run_id": "run-v",
                "checkpoint_iteration": 1000,
                "exploitability_mbb": 880.0,
                "std_error_mbb": 1.0,
                "git_branch": "worktree-hybrid-kernels",
                "vs_control_mbb": -20.0,
                "vs_control_p_value": 0.01,
                "vs_control_blocked": ["payload missing, cannot pair"],
            },
        ],
    },
    "ledger": {
        "op": "ledger",
        "ledger": "data/eval_ledger.jsonl",
        "rebuilt": {"recovered": 3, "preserved": 1},
        "rows": [
            {
                "run_id": "run-a",
                "eval_git_commit": "abcdef1234",
                "eval_git_dirty": True,
                "knobs": {"scorer": "lookahead", "opponent": "blueprint", "base_seed": 1},
                "results": {"exploitability_mbb": 900.0, "std_error_mbb": 12.0, "num_hands": 1000},
            }
        ],
    },
    "compare": {
        "op": "compare",
        "run_a": "run-a",
        "run_b": "run-b",
        "tier_warnings": ["base_seed differs"],
        "comparison": {
            "mean_a": 900.0,
            "mean_b": 880.0,
            "mean_diff": -20.0,
            "se_diff": 5.0,
            "ci_lower": -30.0,
            "ci_upper": -10.0,
            "p_value": 0.01,
            "is_significant": True,
            "correlation": 0.8,
            "se_unpaired": 9.0,
        },
    },
    "evaluate": {
        "op": "evaluate",
        "run_id": "run-a",
        "estimator": "local_best_response",
        "infosets": 42,
        "results": {"exploitability_mbb": 900.0, "std_error_mbb": 12.0},
    },
    "runinfo": {
        "op": "runinfo",
        "run_id": "run-a",
        "config_name": "production",
        "status": "completed",
        "experiment_id": "exp-7",
        "arm": "control",
        "parent_run_id": None,
        "git_commit": "cafebabe" * 5,
        "git_dirty": False,
        "card_abstraction_hash": "ae5a7e6648d7cd02",
        "iterations": 30_000_000,
        "runtime_seconds": 9000.0,
        "training_tasks": 4,
        "total_progress_rows": 2,
        "progress": [
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
        "coverage_flat_from": 20_000_000,
        "curve": {
            "run_id": "run-a",
            "tier": "exact_br flops=8",
            "points": [
                {
                    "iteration": 10_000_000,
                    "exploitability_mbb": 1800.0,
                    "std_error_mbb": 0.0,
                    "num_hands": 0,
                    "eval_git_commit": None,
                },
            ],
            "missing_iterations": [5_000_000, 20_000_000],
        },
        "tasks": [{"task_id": "prod-101010-1", "attempt": 1, "cause": "killed"}],
        "gaps": ["unscored ladder rungs: 5,000,000, 20,000,000"],
    },
    "blueprint-serve": {
        "op": "blueprint-serve",
        "run": "run-production-025433-1095",
        "run_dir": "/mnt/work/runs/run-production-025433-1095",
        "at_iteration": None,
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
    "cost": {
        "op": "cost",
        "hours": 0.0,
        "task_hours": 68.87,
        "tasks": 47,
        "peak_concurrency": 4,
        "first_at": "2026-08-02T22:06:18+00:00",
        "last_at": "2026-08-03T16:58:59+00:00",
        "rate_per_node_hour": 0.8,
        "dollars": 55.1,
        "series": [
            {"at": "2026-08-02T22:06:18+00:00", "running": 1},
            {"at": "2026-08-02T22:08:22+00:00", "running": 3},
            {"at": "2026-08-03T16:58:59+00:00", "running": 0},
        ],
    },
    "runs": {
        "op": "runs",
        "runs": [
            {
                "name": "run-production-025433-1095",
                "commits_ago": None,
                "git_dirty": False,
                "has_checkpoint": True,
                "loadable": True,
                "blocker": None,
                "iterations": 76_000_000,
                "num_infosets": 32_240_608,
                "config_name": "production",
                "status": "completed",
            },
            # A run that never checkpointed: still listed, with the reason, because
            # it is exactly the one someone is looking for when asking what happened.
            {
                "name": "run-20260802_203312-8c4a2c",
                "commits_ago": 3,
                "git_dirty": True,
                "has_checkpoint": False,
                "loadable": False,
                "blocker": "never checkpointed",
                "iterations": None,
                "num_infosets": None,
                "config_name": "quick_test",
                "status": "failed",
            },
        ],
    },
    "tasks": {
        "op": "tasks",
        "reconciled": 1,
        "rows": [
            {
                "task_id": "train-production-1095-to150M-090456-1",
                "attempt": 1,
                "op": "train",
                "what": "train ->150M",
                "run_id": "run-production-025433-1095",
                "cause": "running",
                "exit_code": None,
                "started_at": "2026-08-05T09:04:56+00:00",
                "ended_at": None,
                # A running task's bar. Absent on a finished one -- see below.
                "progress": {"done": 38000000.0, "total": 150000000.0, "unit": "iterations"},
                "workers": 16,
                "units": 0.0,
                # Seconds left, derived at read time from this task's own rate
                # and the history of tasks that ran at the same width.
                "eta_seconds": 9540.0,
            },
            {
                "task_id": "prod-101010-1",
                "attempt": 1,
                "op": "train-static",
                "run_id": "run-a",
                "cause": "killed",
                "exit_code": 137,
                "ended_at": "2026-08-02T10:00:00Z",
            },
        ],
    },
    "progress": {
        "op": "progress",
        "run_id": "run-a",
        "total_rows": 2,
        "schema_version_min": 1,
        "schema_version_max": 1,
        "coverage_plateau_iteration": 2000000,
        "rows": [
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
            },
            # A row from before a field existed: the renderer must blank it, not
            # crash and not print a placeholder that reads as a real measurement.
            {"schema_version": 0, "iteration": 2000000, "coverage": 0.101},
        ],
    },
    "jobs": {
        "op": "jobs",
        "jobs": [
            {
                "job": "poker-20260802",
                "state": "BatchJobState.ACTIVE",
                # One of each half the pool has: a task OCCUPYING a node, and
                # one waiting for one. The queued shape is the one that carries
                # no node and no start time, so a sample without it would let a
                # reader assume both are always present.
                "tasks": [
                    {
                        "task": "train-production-to150M-090456-1",
                        "state": "BatchTaskState.RUNNING",
                        "exit_code": None,
                        "node": "tvmps_x",
                        "created": "2026-08-04T09:04:56+00:00",
                        "start_time": "2026-08-04T09:07:12+00:00",
                    },
                    {
                        "task": "score-production-1095-150M-seed7-090501-2",
                        "state": "BatchTaskState.ACTIVE",
                        "exit_code": None,
                        "node": None,
                        "created": "2026-08-04T09:05:01+00:00",
                        "start_time": None,
                    },
                ],
            }
        ],
        "total_jobs": 3,
        "hidden_jobs": 2,
    },
    "logs": {
        "op": "logs",
        "listing": None,
        "task": "production-1",
        "lines": ["train-static: config=production", "publish complete"],
    },
    # A resize error whose real cause is escaped JSON inside a value. This is
    # the shape the renderer exists to unpack -- Batch reports the generic
    # `AllocationFailed` and hides the actionable half in here.
    "pool-status": {
        "op": "pool-status",
        "pool_id": "train",
        "hourly_cost": "$0.80/hr/node",
        "allocation_state": "AllocationState.STEADY",
        "current_dedicated_nodes": 0,
        "target_dedicated_nodes": 0,
        "vm_size": "standard_d16als_v6",
        "resize_errors": [
            {
                "code": "AllocationFailed",
                "message": "Desired number of dedicated nodes could not be allocated",
                "values": {"ErrorJson": '{"code":"AllocationFailed"}', "Plain": "not json"},
            }
        ],
    },
    "autoscale-check": {
        "op": "autoscale-check",
        "pool_id": "train",
        "variables": ["$TargetDedicatedNodes=0", "pending=0"],
        "error": None,
    },
    "submit": {
        "op": "submit",
        "target_iteration": 25_000_000,
        "code_snapshot": "code-20260802_000000",
        "job_id": "poker-20260802",
        "tasks": ["production-000000-1"],
    },
    "score": {
        "op": "score",
        "run_id": "run-a",
        "method": "exact_br",
        "rungs": ["10000000", "20000000"],
        "code_snapshot": "code-20260802_000000",
        "job_id": "poker-20260802",
        "tasks": ["run-a-000000-1", "run-a-000000-2"],
    },
    "cancel": {"op": "cancel", "job_id": "poker-20260802", "task_id": "run-a-000000-1"},
    "push-code": {"op": "push-code", "code_snapshot": "code-20260802_000000"},
    "submit-precompute": {
        "op": "submit-precompute",
        "abstraction_config": "ochs_gate_ochs",
        "target_name": "buckets-F20T20R20-rexact-deadbeef",
        "already_published": ["buckets-F50T100R200-rexact-b59ef7b2"],
        "force": False,
        "code_snapshot": "code-20260802_000000",
        "job_id": "poker-20260802",
        "tasks": ["ochs_gate_ochs-000000-1"],
    },
    "push-data": {"op": "push-data", "uploaded": {"buckets-F50T100R200": 9}},
}


class TestFixturesMatchTheRealPayloads:
    """The fixtures above are hand-written; pin them to the dataclasses they mimic.

    Without this the render tests only prove that my dict agrees with my renderer.
    Each of these ops returns ``{"op": ...} | dataclasses.asdict(<result>)``, so the
    dataclass fields are exactly the keys a renderer may rely on.
    """

    def test_fixture_keys_come_from_the_service_dataclasses(self):
        for op, cls, extra in (
            ("train-static", services.StaticTrainingOutput, set()),
            ("promote", services.Baseline, {"baseline"}),
            ("report", services.ExperimentReport, set()),
            ("curve", services.CurveOutput, {"decay_ratio"}),
        ):
            expected = {f.name for f in dataclasses.fields(cls)} | extra | {"op"}
            assert set(PAYLOADS[op]) == expected, op


# Composed from the panels' own fixtures, which is the point of the command:
# it renders each panel with the renderer that owns it rather than formatting
# anything itself. Built after the literal so it can reuse them by key.
#
# `tasks` is deliberately the FAILED panel here. A status screen's whole value is
# that it still shows the other two when one is unavailable, and that path only
# runs when something is already wrong -- so it is the one worth pinning.
PAYLOADS["status"] = {
    "op": "status",
    "at": "2026-08-03T00:24:48-07:00",
    "elapsed_seconds": 22.1,
    "watch": 0,
    "requested_watch": 0,
    "limit": 10,
    "with_tasks": True,
    "panels": {
        "pool": {"payload": PAYLOADS["pool-status"], "error": None},
        "jobs": {"payload": PAYLOADS["jobs"], "error": None},
        "tasks": {"payload": None, "error": "Azure rejected the credential — try `az login`."},
    },
}

BY_NAME = {command.name: command for command in COMMANDS}


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
        for arm in payload["arms"]:
            arm["git_branch"] = ""

        BY_NAME["report"].render(payload)
        out = capsys.readouterr().out

        assert "from" not in out
        assert "900.0" in out, "the numbers must still be there"
