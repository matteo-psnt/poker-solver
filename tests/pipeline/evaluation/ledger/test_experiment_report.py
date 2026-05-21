"""Experiment reporting, baseline promotion, and the tightened pairing guard.

The property that matters throughout: a variant's raw score is never presented as
evidence on its own. A fork gets extra training, and that alone moves the number,
so an arm is only attributable against its paired control — and when it cannot be,
the report says so rather than omitting the column.
"""

import json

import pytest

from src.pipeline import services
from src.pipeline.evaluation import ledger


def _row(arm, run_id, mbb, *, experiment="exp-1", seed=7, samples=None, **knobs):
    return {
        "run_id": run_id,
        "experiment_id": experiment,
        "arm": arm,
        "method": "lbr",
        "timestamp": f"2026-01-01T00:00:0{len(run_id) % 10}+00:00",
        "card_abstraction_hash": "abc",
        "action_config_hash": "def",
        "checkpoint_iteration": 1000,
        "knobs": {
            "scorer": "myopic",
            "opponent": "blueprint",
            "include_off_tree": False,
            "base_seed": seed,
            **knobs,
        },
        "results": {
            "exploitability_mbb": mbb,
            "std_error_mbb": 1.0,
            "num_hands": 3,
            "pair_samples_mbb": samples or [mbb, mbb, mbb],
        },
    }


@pytest.fixture
def store(tmp_path):
    """A ledger plus the per-run payloads its rows point at."""

    def _build(rows):
        ledger_path = tmp_path / "led.jsonl"
        written = []
        for index, row in enumerate(rows):
            run_dir = tmp_path / "runs" / row["run_id"]
            (run_dir / "evals").mkdir(parents=True, exist_ok=True)
            # One file per row, as production does — two evals of one arm must not
            # share a payload or the later silently overwrites the earlier's samples.
            payload = run_dir / "evals" / f"eval-{index}.json"
            payload.write_text(json.dumps({"results": row["results"], "run_id": row["run_id"]}))
            written.append({**row, "result_path": f"{row['run_id']}/evals/{payload.name}"})
        ledger_path.write_text("".join(json.dumps(r) + "\n" for r in written))
        return ledger_path, tmp_path / "runs", tmp_path / "baseline.json"

    return _build


class TestAttribution:
    def test_variant_is_scored_against_the_control_not_zero(self, store):
        led, runs, base = store(
            [
                _row("control", "run-c", 100.0, samples=[100.0, 100.0, 100.0]),
                _row("variant:idea", "run-v", 60.0, samples=[60.0, 60.0, 60.0]),
            ]
        )
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        variant = next(a for a in report.arms if a.arm == "variant:idea")
        assert variant.vs_control_mbb == pytest.approx(-40.0)
        assert variant.vs_control_blocked == []
        assert report.control_run_id == "run-c"

    def test_control_arm_is_not_compared_to_itself(self, store):
        led, runs, base = store([_row("control", "run-c", 100.0)])
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert report.arms[0].vs_control_mbb is None

    def test_missing_control_is_called_out(self, store):
        led, runs, base = store([_row("variant:idea", "run-v", 60.0)])
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert report.control_run_id is None
        assert any("control" in n for n in report.notes)

    def test_mismatched_tier_is_never_shown_as_a_delta(self, store):
        # Different seeds mean the deals are not paired; a delta here would be a lie.
        # The arm is kept out of the comparison table and named in the notes instead.
        led, runs, base = store(
            [
                _row("control", "run-c", 100.0, seed=1),
                _row("variant:idea", "run-v", 60.0, seed=2),
            ]
        )
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert all(a.vs_control_mbb is None for a in report.arms)
        assert any("variant:idea" in n for n in report.notes)

    def test_other_experiments_are_excluded(self, store):
        led, runs, base = store(
            [
                _row("control", "run-c", 100.0),
                _row("variant:other", "run-o", 10.0, experiment="exp-2"),
            ]
        )
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert [a.arm for a in report.arms] == ["control"]

    def test_control_tier_is_authoritative_over_a_newer_stray_eval(self, store):
        # A variant re-scored under different knobs must not displace its
        # comparable eval — otherwise a stray run makes the whole table
        # "not attributable" against a control it was never measured beside.
        rows = [
            _row("control", "run-c", 100.0, samples=[100.0] * 3),
            _row("variant:idea", "run-v", 60.0, samples=[60.0] * 3),
        ]
        stray = _row("variant:idea", "run-v", 5.0, seed=999, samples=[5.0] * 3)
        stray["timestamp"] = "2026-12-31T00:00:00+00:00"  # newest overall
        led, runs, base = store([*rows, stray])

        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        variant = next(a for a in report.arms if a.arm == "variant:idea")
        assert variant.exploitability_mbb == 60.0, "the in-tier eval must win"
        assert variant.vs_control_mbb == pytest.approx(-40.0)

    def test_arm_with_no_eval_in_the_control_tier_is_named(self, store):
        led, runs, base = store(
            [
                _row("control", "run-c", 100.0),
                _row("variant:idea", "run-v", 60.0, seed=999),
            ]
        )
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert [a.arm for a in report.arms] == ["control"]
        assert any("no evaluation in the control's tier" in n for n in report.notes)

    def test_empty_experiment_reports_a_note_not_a_crash(self, store):
        led, runs, base = store([_row("control", "run-c", 100.0)])
        report = services.experiment_report(
            "nope", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert report.arms == []
        assert report.notes


class TestBaseline:
    def test_promote_then_load_round_trips(self, tmp_path):
        path = tmp_path / "baseline.json"
        services.promote_baseline("run-a", "beat the base", path=path, checkpoint_iteration=25_000)
        loaded = services.load_baseline(path)
        assert loaded is not None
        assert loaded.run_id == "run-a"
        assert loaded.checkpoint_iteration == 25_000
        assert loaded.promoted_at.endswith("+00:00")

    def test_missing_baseline_is_none_not_an_error(self, tmp_path):
        assert services.load_baseline(tmp_path / "absent.json") is None

    def test_corrupt_baseline_is_none_not_an_error(self, tmp_path):
        path = tmp_path / "baseline.json"
        path.write_text("{not json")
        assert services.load_baseline(path) is None

    def test_promotion_replaces_the_previous_pointer(self, tmp_path):
        path = tmp_path / "baseline.json"
        services.promote_baseline("run-a", "first", path=path)
        services.promote_baseline("run-b", "second", path=path)
        loaded = services.load_baseline(path)
        assert loaded is not None
        assert loaded.run_id == "run-b"

    def test_report_surfaces_the_current_baseline(self, store, tmp_path):
        led, runs, base = store([_row("control", "run-c", 100.0)])
        services.promote_baseline("run-base", "why", path=base)
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert report.baseline_run_id == "run-base"


class TestTierGuardGaps:
    """Gaps that let materially different evals pair silently before this change."""

    def _pair(self, **over_b):
        a = _row("control", "run-a", 100.0)
        b = _row("variant", "run-b", 100.0)
        for key, value in over_b.items():
            if key in ("method", "card_abstraction_hash", "action_config_hash"):
                b[key] = value
            else:
                b["knobs"][key] = value
                a["knobs"].setdefault(key, None)
        return a, b

    def test_lookahead_depth_now_blocks(self):
        a, b = self._pair(lookahead_depth=4)
        a["knobs"]["lookahead_depth"] = 2
        assert any("lookahead_depth" in r for r in ledger.tier_mismatches(a, b))

    def test_runouts_now_block(self):
        a, b = self._pair(runouts=32)
        a["knobs"]["runouts"] = 8
        assert any("runouts" in r for r in ledger.tier_mismatches(a, b))

    def test_differing_method_blocks(self):
        a, b = self._pair(method="rollout")
        assert any("method differs" in r for r in ledger.tier_mismatches(a, b))

    def test_two_exact_br_rows_are_refused_with_a_reason(self):
        # Previously passed every check vacuously, then died on a bare KeyError
        # deep in compare because exact_br payloads carry no per-hand samples.
        a = _row("control", "run-a", 100.0)
        b = _row("variant", "run-b", 90.0)
        a["method"] = b["method"] = "exact_br"
        reasons = ledger.tier_mismatches(a, b)
        assert any("no per-hand samples" in r for r in reasons)

    def test_differing_abstraction_hash_blocks(self):
        a, b = self._pair(card_abstraction_hash="other")
        assert any("card_abstraction_hash" in r for r in ledger.tier_mismatches(a, b))

    def test_a_matched_pair_still_passes(self):
        a = _row("control", "run-a", 100.0)
        b = _row("variant", "run-b", 90.0)
        assert ledger.tier_mismatches(a, b) == []


class TestStrayControlEval:
    """A stray re-score of the CONTROL must not empty the report.

    The control is the row that sets the tier, so it is the row whose strays do the
    most damage: picking its newest eval would blame every variant for "no
    evaluation in the control's tier" while a fully matched set sat in the older one.
    """

    def test_stray_control_rescore_does_not_hide_matched_arms(self, store):
        matched = [
            _row("control", "run-c", 100.0, samples=[100.0] * 3),
            _row("variant:a", "run-a", 60.0, samples=[60.0] * 3),
            _row("variant:b", "run-b", 80.0, samples=[80.0] * 3),
        ]
        # Operator sanity-checks the control at a deeper lookahead, much later.
        stray = _row("control", "run-c", 42.0, seed=999, samples=[42.0] * 3)
        stray["timestamp"] = "2026-12-31T00:00:00+00:00"
        led, runs, base = store([*matched, stray])

        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert {a.arm for a in report.arms} == {"control", "variant:a", "variant:b"}
        assert all(a.vs_control_mbb is not None for a in report.arms if a.arm != "control"), (
            "the matched set must still be attributable"
        )

    def test_tier_with_the_most_arms_wins(self, store):
        # Two complete tiers; the one comparing more arms is the useful one.
        small = [
            _row("control", "run-c", 100.0, seed=1, samples=[100.0] * 3),
            _row("variant:a", "run-a", 60.0, seed=1, samples=[60.0] * 3),
        ]
        big = [
            _row("control", "run-c", 90.0, seed=2, samples=[90.0] * 3),
            _row("variant:a", "run-a", 50.0, seed=2, samples=[50.0] * 3),
            _row("variant:b", "run-b", 70.0, seed=2, samples=[70.0] * 3),
        ]
        led, runs, base = store([*big, *small])
        report = services.experiment_report(
            "exp-1", ledger_path=led, runs_dir=runs, baseline_path=base
        )
        assert len(report.arms) == 3


class TestTierLabelNamesEverythingTierKeySplitsOn:
    def test_distinct_tiers_never_render_identically(self):
        def row(depth):
            return {
                "method": "lbr",
                "knobs": {
                    "scorer": "lookahead",
                    "opponent": "blueprint",
                    "include_off_tree": False,
                    "base_seed": 7,
                    "lookahead_depth": depth,
                },
            }

        a, b = row(2), row(4)
        assert ledger.tier_key(a) != ledger.tier_key(b)
        assert ledger.tier_label(a) != ledger.tier_label(b), (
            "two different tiers printing the same string leaves the operator "
            "unable to tell what --tier 1 selects"
        )

    def test_inapplicable_knobs_are_not_padded_into_the_label(self):
        myopic = {
            "method": "lbr",
            "knobs": {
                "scorer": "myopic",
                "opponent": "blueprint",
                "include_off_tree": False,
                "base_seed": 7,
            },
        }
        assert "lookahead_depth" not in ledger.tier_label(myopic)
