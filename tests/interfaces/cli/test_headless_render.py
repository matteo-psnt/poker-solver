"""Every command's payload must survive human-mode rendering.

The renderers are pure formatting, so the only failure they can have is a key
the payload does not carry — which is exactly the failure that shipped:
``checkpoint-profile`` had no entry in the old if/elif chain and fell through to
the evaluate branch, dying on ``payload["results"]``. These tests pin the shape
of each op against its renderer so a new command cannot inherit another's.
"""

import argparse
import dataclasses

from src.interfaces.cli import headless, headless_render
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


class TestEveryOpRenders:
    def test_no_payload_kind_raises(self, capsys):
        for op, payload in PAYLOADS.items():
            headless_render.print_human(payload)
            assert capsys.readouterr().out, f"'{op}' rendered nothing"

    def test_checkpoint_profile_does_not_borrow_the_evaluate_renderer(self, capsys):
        # The exact regression: it used to fall through and KeyError on "results".
        headless_render.print_human(PAYLOADS["checkpoint-profile"])
        out = capsys.readouterr().out
        assert "Checkpoint profile for run-a" in out
        assert "Evaluation complete" not in out

    def test_unknown_op_says_so_instead_of_raising(self, capsys):
        headless_render.print_human({"op": "not-a-command"})
        assert "--json" in capsys.readouterr().out

    def test_every_subcommand_has_a_renderer(self):
        """A command reachable from the CLI but absent from RENDERERS is a gap.

        Builders are discovered rather than listed, so a new subcommand is held to
        this the moment it is added — which is what would have caught the original.
        """
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        common = argparse.ArgumentParser(add_help=False)
        builders = [
            value
            for name, value in vars(headless).items()
            if name.startswith("_add_") and name.endswith("_parser")
        ]
        assert builders, "no subcommand builders found — has headless been restructured?"
        for build in builders:
            build(sub, common)

        assert set(sub.choices) == set(headless_render.RENDERERS)
