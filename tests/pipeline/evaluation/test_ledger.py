"""Tests for the append-only evaluation ledger and its comparison guard."""

from dataclasses import replace

import pytest

from src.pipeline.evaluation import ledger
from src.pipeline.evaluation.hunl_local_best_response import LBRConfig


def _fake_provenance(run_id="run-x"):
    return ledger.RunProvenance(
        run_id=run_id,
        git_commit="cafebabe" * 5,
        git_dirty=False,
        config_name="quick_test",
        card_abstraction_hash="deadbeef",
        action_config_hash="beefcafe",
        representation_version=1,
    )


def _lbr_config(**over):
    # LBRConfig's own defaults (myopic/blueprint/on-tree, lookahead 2/3) are the
    # baseline tier; tests override per-case.
    return replace(LBRConfig(num_hands=100, equity_runouts=12), **over)


def _results(base_seed=7, mbb=100.0, n=100):
    return {
        "exploitability_mbb": mbb,
        "std_error_mbb": 5.0,
        "num_hands": n,
        "base_seed": base_seed,
        "pair_samples_mbb": [float(i) for i in range(n)],
    }


class TestKnobs:
    def test_lbr_knobs_take_seed_from_results(self):
        knobs = ledger.build_lbr_knobs(_lbr_config(), _results(base_seed=42))
        assert knobs["base_seed"] == 42
        assert knobs["scorer"] == "myopic"
        # deployed-only / lookahead-only knobs omitted for the blueprint+myopic tier
        assert "resolver_iterations" not in knobs
        assert "lookahead_depth" not in knobs

    def test_lbr_knobs_include_tier_specific(self):
        """Deployed/lookahead tiers pick up the resolver pin (from results — it is
        resolved during the eval) and the lookahead knobs (from the config)."""
        knobs = ledger.build_lbr_knobs(
            _lbr_config(opponent="deployed", scorer="lookahead"),
            _results() | {"resolver_iterations": 64},
        )
        assert knobs["resolver_iterations"] == 64
        assert knobs["lookahead_depth"] == 2

    def test_config_and_params_builders_agree(self):
        """The LBRConfig wrapper and the explicit-params core must produce identical
        tiers so every transport records pairable rows."""
        results = _results(base_seed=42) | {"resolver_iterations": 64}
        config = _lbr_config(opponent="deployed", scorer="lookahead")
        from_config = ledger.build_lbr_knobs(config, results)
        from_params = ledger.build_lbr_knobs_from_params(
            scorer="lookahead",
            opponent="deployed",
            hands=100,
            runouts=12,
            include_off_tree=False,
            base_seed=42,
            resolver_iterations=64,
            lookahead_depth=2,
            lookahead_top_k=3,
        )
        assert from_config == from_params


class TestWriteAndAppend:
    def test_write_payload_never_clobbers(self, tmp_path):
        knobs = {"scorer": "myopic", "base_seed": 7}
        p1 = ledger.write_payload(tmp_path, {"op": "evaluate", "n": 1}, knobs)
        p2 = ledger.write_payload(tmp_path, {"op": "evaluate", "n": 2}, knobs)
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_append_and_read_roundtrip(self, tmp_path):
        led = tmp_path / "eval_ledger.jsonl"
        ledger.append_record({"run_id": "r1", "x": 1}, led)
        ledger.append_record({"run_id": "r2", "x": 2}, led)
        rows = ledger.read_records(led)
        assert [r["run_id"] for r in rows] == ["r1", "r2"]

    def test_read_missing_ledger_is_empty(self, tmp_path):
        assert ledger.read_records(tmp_path / "absent.jsonl") == []

    def test_latest_record_for_run_returns_last(self, tmp_path):
        led = tmp_path / "l.jsonl"
        ledger.append_record({"run_id": "r1", "v": 1}, led)
        ledger.append_record({"run_id": "r1", "v": 2}, led)
        ledger.append_record({"run_id": "r2", "v": 9}, led)
        latest = ledger.latest_record_for_run("r1", led)
        assert latest is not None
        assert latest["v"] == 2
        assert ledger.latest_record_for_run("missing", led) is None

    def test_build_record_shape_and_provenance(self, tmp_path):
        results = _results()
        knobs = ledger.build_lbr_knobs(_lbr_config(), results)
        payload_path = ledger.write_payload(tmp_path, {"results": results}, knobs)
        record = ledger.build_record(
            provenance=_fake_provenance("run-x"),
            method="lbr",
            estimator="lbr",
            infosets=10,
            knobs=knobs,
            results=results,
            result_path=payload_path,
            timestamp="2026-07-17T00:00:00",
        )
        assert record["run_id"] == "run-x"
        assert record["train_git_commit"] == _fake_provenance("run-x").git_commit
        assert record["results"]["n"] == 100
        assert record["results"]["exploitability_mbb"] == 100.0
        # eval_git_* are stamped from the current checkout (str/None, bool/None)
        assert "eval_git_commit" in record


class TestRecordEvaluation:
    def test_writes_payload_and_appends_row(self, tmp_path):
        run_dir = tmp_path / "run-x"
        run_dir.mkdir()
        results = _results(base_seed=7)
        knobs = ledger.build_lbr_knobs(_lbr_config(), results)
        payload = {"op": "evaluate", "infosets": 10, "results": results}
        led = tmp_path / "eval_ledger.jsonl"

        result_path, record = ledger.record_evaluation(
            run_dir=run_dir,
            payload=payload,
            provenance=_fake_provenance("run-x"),
            method="lbr",
            estimator="lbr",
            knobs=knobs,
            ledger_path=led,
        )

        assert result_path.exists()
        assert result_path.parent == run_dir / "evals"
        rows = ledger.read_records(led)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-x"
        assert rows[0]["result_path"] == str(result_path)
        # The appended row round-trips to its full payload, including per-hand samples.
        assert ledger.load_payload(record)["results"]["base_seed"] == 7


class TestTierMismatches:
    def _row(self, **knobs):
        base = dict(scorer="myopic", opponent="blueprint", include_off_tree=False, base_seed=7)
        base.update(knobs)
        return {"knobs": base, "results": {"num_hands": 100}}

    def test_matching_rows_pass(self):
        assert ledger.tier_mismatches(self._row(), self._row()) == []

    def test_seed_mismatch_refused(self):
        reasons = ledger.tier_mismatches(self._row(base_seed=7), self._row(base_seed=8))
        assert any("base_seed" in r for r in reasons)

    def test_missing_seed_refused(self):
        reasons = ledger.tier_mismatches(self._row(base_seed=None), self._row())
        assert any("base_seed missing" in r for r in reasons)

    def test_scorer_tier_mismatch_refused(self):
        reasons = ledger.tier_mismatches(self._row(scorer="myopic"), self._row(scorer="lookahead"))
        assert any("scorer" in r for r in reasons)

    def test_hand_count_mismatch_refused(self):
        a = self._row()
        b = {"knobs": a["knobs"], "results": {"num_hands": 50}}
        reasons = ledger.tier_mismatches(a, b)
        assert any("num_hands" in r for r in reasons)


class TestLoadPayload:
    def test_missing_payload_raises(self):
        with pytest.raises(FileNotFoundError):
            ledger.load_payload({"run_id": "r", "result_path": "/no/such/file.json"})
