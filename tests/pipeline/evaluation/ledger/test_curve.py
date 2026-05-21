"""Tests for within-run convergence curves built from ledger rows.

The property under test throughout is that a curve never mixes comparison tiers:
two scorers measure different things, so plotting them on one axis produces a
shape that means nothing.
"""

import json

import pytest

from src.pipeline.evaluation import ledger
from src.shared.records import STATIC_CHECKPOINT


def _row(run_id="run-a", iteration=1000, mbb=100.0, scorer="myopic", seed=7, **knobs):
    return {
        "run_id": run_id,
        "method": "lbr",
        "checkpoint_iteration": iteration,
        "knobs": {
            "scorer": scorer,
            "opponent": "blueprint",
            "include_off_tree": False,
            "base_seed": seed,
            **knobs,
        },
        "results": {"exploitability_mbb": mbb, "std_error_mbb": 1.0, "num_hands": 10},
    }


class TestCurveSeries:
    def test_orders_points_by_iteration_regardless_of_record_order(self):
        rows = [_row(iteration=3000), _row(iteration=500), _row(iteration=2000)]
        [(_, points)] = ledger.curve_series(rows, "run-a")
        assert sorted(points) == [500, 2000, 3000]

    def test_never_merges_distinct_tiers(self):
        rows = [_row(scorer="myopic"), _row(scorer="lookahead", iteration=2000)]
        series = ledger.curve_series(rows, "run-a")
        assert len(series) == 2, "a myopic and a lookahead eval are different instruments"

    def test_differing_base_seed_is_a_different_tier(self):
        # Paired CRN requires identical deals; two seeds are not one curve.
        rows = [_row(seed=1), _row(seed=2, iteration=2000)]
        assert len(ledger.curve_series(rows, "run-a")) == 2

    def test_best_covered_tier_comes_first(self):
        rows = [
            _row(scorer="lookahead", iteration=1000),
            _row(scorer="myopic", iteration=1000),
            _row(scorer="myopic", iteration=2000),
        ]
        label, points = ledger.curve_series(rows, "run-a")[0]
        assert "myopic" in label
        assert len(points) == 2

    def test_later_record_supersedes_earlier_for_same_checkpoint(self):
        rows = [_row(iteration=1000, mbb=500.0), _row(iteration=1000, mbb=300.0)]
        [(_, points)] = ledger.curve_series(rows, "run-a")
        assert points[1000]["results"]["exploitability_mbb"] == 300.0

    def test_ignores_other_runs_and_unplaceable_rows(self):
        rows = [_row(), _row(run_id="run-b"), {**_row(), "checkpoint_iteration": None}]
        [(_, points)] = ledger.curve_series(rows, "run-a")
        assert len(points) == 1


class TestExploitabilityCurve:
    """The services-layer join of the on-disk ladder to recorded evaluations."""

    @pytest.fixture
    def run_dir(self, tmp_path):
        # Deliberately no .run.json: the curve is a report over the ledger and the
        # manifest, and must not depend on metadata a legacy run may not have.
        run = tmp_path / "run-a"
        run.mkdir()
        return run

    def _ledger(self, tmp_path, rows):
        path = tmp_path / "led.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def _manifest(self, run_dir, current, retained):
        (run_dir / STATIC_CHECKPOINT).write_text(
            json.dumps(
                {
                    "iteration": current,
                    "zarr": f"static-{current}.zarr",
                    "fingerprint": "fp",
                    "retained": [{"iteration": i, "zarr": f"static-{i}.zarr"} for i in retained],
                }
            )
        )

    def test_reports_ladder_rungs_that_have_no_evaluation(self, run_dir, tmp_path):
        from src.pipeline import services

        self._manifest(run_dir, 3000, [1000, 2000])
        path = self._ledger(tmp_path, [_row(iteration=1000)])
        out = services.exploitability_curve(run_dir, ledger_path=path)
        assert [p.iteration for p in out.points] == [1000]
        assert out.missing_iterations == [2000, 3000], "a curve with holes must say so"

    def test_counts_rows_that_predate_checkpoint_iteration(self, run_dir, tmp_path):
        from src.pipeline import services

        self._manifest(run_dir, 1000, [])
        rows = [{**_row(), "checkpoint_iteration": None} for _ in range(3)]
        out = services.exploitability_curve(run_dir, ledger_path=self._ledger(tmp_path, rows))
        assert out.points == []
        assert out.unplaceable_records == 3, "an empty curve beside a full ledger reads as a bug"

    def test_survives_a_legacy_or_torn_manifest(self, run_dir, tmp_path):
        from src.pipeline import services

        # Missing 'key_table' — the shape a real pre-v3 run on disk has.
        (run_dir / "CHECKPOINT.json").write_text(json.dumps({"iteration": 13000}))
        path = self._ledger(tmp_path, [_row(iteration=13000)])
        out = services.exploitability_curve(run_dir, ledger_path=path)
        assert [p.iteration for p in out.points] == [13000], "a report must not die on the ladder"
        assert out.retained_iterations == []

    def test_lists_unplotted_tiers_rather_than_dropping_them(self, run_dir, tmp_path):
        from src.pipeline import services

        self._manifest(run_dir, 2000, [1000])
        rows = [
            _row(iteration=1000),
            _row(iteration=2000),
            _row(iteration=1000, scorer="lookahead"),
        ]
        out = services.exploitability_curve(run_dir, ledger_path=self._ledger(tmp_path, rows))
        assert len(out.points) == 2
        assert any("lookahead" in t for t in out.other_tiers)

    def test_decay_ratio_is_first_over_last(self, run_dir, tmp_path):
        from src.pipeline import services

        self._manifest(run_dir, 2000, [1000])
        rows = [_row(iteration=1000, mbb=400.0), _row(iteration=2000, mbb=100.0)]
        out = services.exploitability_curve(run_dir, ledger_path=self._ledger(tmp_path, rows))
        assert out.decay_ratio == pytest.approx(4.0)

    def test_decay_ratio_is_none_with_a_single_point(self, run_dir, tmp_path):
        from src.pipeline import services

        self._manifest(run_dir, 1000, [])
        out = services.exploitability_curve(
            run_dir, ledger_path=self._ledger(tmp_path, [_row(iteration=1000)])
        )
        assert out.decay_ratio is None


class TestTierSelectorBounds:
    """`--tier` must not silently plot a tier the caller did not ask for."""

    def _setup(self, tmp_path, tiers=1):
        run = tmp_path / "run-a"
        run.mkdir()
        rows = [_row(iteration=1000, scorer=f"s{i}") for i in range(tiers)]
        path = tmp_path / "led.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return run, path

    def test_out_of_range_tier_raises_a_named_error(self, tmp_path):
        from src.pipeline import services

        run, path = self._setup(tmp_path, tiers=1)
        with pytest.raises(IndexError, match="out of range"):
            services.exploitability_curve(run, ledger_path=path, tier_index=5)

    def test_negative_tier_is_rejected_not_wrapped(self, tmp_path):
        # Python would index from the end and quietly return a different tier.
        from src.pipeline import services

        run, path = self._setup(tmp_path, tiers=2)
        with pytest.raises(IndexError):
            services.exploitability_curve(run, ledger_path=path, tier_index=-1)
