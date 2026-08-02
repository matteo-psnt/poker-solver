"""Every command's payload must survive human-mode rendering.

Renderers are pure formatting, so the only failure they can have is a key the
payload does not carry -- the failure that shipped once, when
``checkpoint-profile`` fell through to the evaluate branch and died on
``payload["results"]``. These pin each op's payload shape against its renderer.

"Every subcommand HAS a renderer" is no longer tested: a ``Command`` cannot be
constructed without one, so it is a property of the type rather than a thing to
check.
"""

import dataclasses

from src.interfaces.cli.commands import COMMANDS
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
    "checkpoint-profile": {
        "op": "checkpoint-profile",
        "run": "run-a",
        "num_checkpoints": 3,
        "checkpoint_seconds": 30.0,
        "volume_commit_seconds": 10.0,
        "total_seconds": 40.0,
        "commit_share": 0.25,
        "top_level_phases": {"collect_keys": 5.0, "storage_write": 25.0},
        "write_phases": {"write_key_table": 20.0},
    },
    "jobs": {
        "op": "jobs",
        "jobs": [
            {
                "job": "poker-20260802",
                "state": "BatchJobState.ACTIVE",
                "tasks": [
                    {
                        "task": "production-1",
                        "state": "BatchTaskState.RUNNING",
                        "exit_code": None,
                        "node": "tvmps_x",
                    }
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
    "repair-ladder": {
        "op": "repair-ladder",
        "run_id": "run-a",
        "code_snapshot": "code-20260802_000000",
        "job_id": "poker-20260802",
        "tasks": ["run-a-000000-1"],
    },
    "cancel": {"op": "cancel", "job_id": "poker-20260802", "task_id": "run-a-000000-1"},
    "fetch": {
        "op": "fetch",
        "runs": ["run-a"],
        "files": 12,
        "skipped_unnamed": 3,
        "mode": "metadata",
        "ledger_rows": 40,
        "ledger_preserved": 2,
    },
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


BY_NAME = {command.name: command for command in COMMANDS}


class TestEveryOpRenders:
    def test_every_command_renders_its_payload(self, capsys):
        for name, command in BY_NAME.items():
            command.render(PAYLOADS[name])
            assert capsys.readouterr().out, f"'{name}' rendered nothing"

    def test_every_command_has_a_fixture(self):
        """A command with no fixture here is a command nothing pins."""
        assert set(BY_NAME) == set(PAYLOADS)

    def test_checkpoint_profile_does_not_borrow_the_evaluate_renderer(self, capsys):
        # The exact regression: it used to fall through and KeyError on "results".
        BY_NAME["checkpoint-profile"].render(PAYLOADS["checkpoint-profile"])
        out = capsys.readouterr().out
        assert "Checkpoint profile for run-a" in out
        assert "Evaluation complete" not in out
