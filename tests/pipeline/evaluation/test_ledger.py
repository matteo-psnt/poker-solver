"""Tests for the append-only evaluation ledger and its comparison guard."""

import json
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
    def test_write_eval_never_clobbers(self, tmp_path):
        knobs = {"scorer": "myopic", "base_seed": 7}
        p1 = ledger.write_eval(tmp_path, {"op": "evaluate", "n": 1}, ledger.eval_slug(knobs))
        p2 = ledger.write_eval(tmp_path, {"op": "evaluate", "n": 2}, ledger.eval_slug(knobs))
        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    def test_one_file_per_eval(self, tmp_path):
        """Was three shapes: a payload, a record summarising it, and a ledger row."""
        knobs = {"scorer": "myopic", "base_seed": 7}
        ledger.write_eval(tmp_path, {"op": "evaluate"}, ledger.eval_slug(knobs))
        assert len(list((tmp_path / "evals").glob("*.json"))) == 1

    def test_the_ledger_row_is_derived_from_the_document(self, tmp_path):
        """Derivability is what makes `ledger --rebuild` able to regenerate."""
        document = {
            "run_id": "run-a",
            "timestamp": "2026-08-02T00:00:00",
            "results": {
                "exploitability_mbb": 1200.0,
                "std_error_mbb": 12.0,
                "num_hands": 3,
                "pair_samples_mbb": [1.0, 2.0, 3.0],
                "hand_records": ["bulk"] * 100,
            },
        }
        row = ledger.ledger_row(document)
        assert row["run_id"] == "run-a"
        assert row["results"]["n"] == 3
        assert "pair_samples_mbb" not in row["results"], "bulk stays in the document"
        assert "hand_records" not in row["results"]

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

    def test_the_document_carries_provenance_and_full_results(self, tmp_path):
        results = _results()
        knobs = ledger.build_lbr_knobs(_lbr_config(), results)
        payload_path = ledger.write_eval(tmp_path, {"results": results}, ledger.eval_slug(knobs))
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
        assert record["results"]["exploitability_mbb"] == 100.0
        # The FULL results, samples and all: provenance and samples used to live
        # in different files, so neither could be rebuilt from the other.
        assert len(record["results"]["pair_samples_mbb"]) == 100
        # eval_git_* are stamped from the current checkout (str/None, bool/None)
        assert "eval_git_commit" in record
        assert ledger.ledger_row(record)["results"]["n"] == 100


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
        # Stored run-relative, not CWD-relative: the pointer must mean the same
        # thing on a machine that mounts its runs directory somewhere else.
        assert rows[0]["result_path"] == f"run-x/evals/{result_path.name}"
        # The appended row round-trips to its full payload, including per-hand samples.
        assert ledger.load_payload(record, tmp_path)["results"]["base_seed"] == 7


class TestTierMismatches:
    def _row(self, **knobs):
        base = {
            "scorer": "myopic",
            "opponent": "blueprint",
            "include_off_tree": False,
            "base_seed": 7,
        }
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


class TestEvalConsolidationMigration:
    """The old layout: a payload, a record summarising it, and a ledger row.

    Provenance lived only in the record and samples only in the payload, so
    neither could rebuild the other. On the real tree 59 of 78 payloads had no
    record at all.
    """

    def _legacy(self, tmp_path, run_id, slug, *, record=True, ledger_row=True):
        run_dir = tmp_path / run_id
        evals = run_dir / "evals"
        evals.mkdir(parents=True, exist_ok=True)
        (evals / f"eval-{slug}.json").write_text(
            json.dumps({"run_id": run_id, "results": {"exploitability_mbb": 12.0, "n": 3}})
        )
        if record:
            (evals / f"record-{slug}.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "timestamp": "2026-07-18T00:00:00",
                        "knobs": {"scorer": "myopic"},
                        "train_git_commit": "abc",
                        "results": {"exploitability_mbb": 12.0},
                        "result_path": f"{run_id}/evals/eval-{slug}.json",
                    }
                )
            )
        led = tmp_path / "led.jsonl"
        if ledger_row:
            with led.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "timestamp": "2026-07-18T00:00:00",
                            "knobs": {"scorer": "myopic"},
                            "result_path": f"{run_id}/evals/eval-{slug}.json",
                        }
                    )
                    + "\n"
                )
        return run_dir, led

    def test_a_pair_becomes_one_document_carrying_both_halves(self, tmp_path):
        run_dir, led = self._legacy(tmp_path, "run-a", "s1")
        counts = ledger.migrate_eval_files(tmp_path, led)

        assert counts["merged"] == 1
        document = json.loads((run_dir / "evals" / "s1.json").read_text())
        assert document["train_git_commit"] == "abc", "provenance from the record"
        assert document["results"]["n"] == 3, "samples from the payload"

    def test_a_payload_with_no_record_is_recovered_from_the_ledger(self, tmp_path):
        """The ledger is the fullest history, so it -- not record-*.json -- is the key."""
        run_dir, led = self._legacy(tmp_path, "run-a", "s1", record=False)
        counts = ledger.migrate_eval_files(tmp_path, led)

        assert counts["merged"] == 1
        assert json.loads((run_dir / "evals" / "s1.json").read_text())["knobs"]

    def test_a_payload_with_no_provenance_anywhere_is_still_kept(self, tmp_path):
        """The measurement is real; a document that says less beats one that invents."""
        run_dir, led = self._legacy(tmp_path, "run-a", "s1", record=False, ledger_row=False)
        counts = ledger.migrate_eval_files(tmp_path, led)

        assert counts["payload_only"] == 1
        assert (run_dir / "evals" / "s1.json").exists()

    def test_it_is_non_destructive_and_idempotent(self, tmp_path):
        run_dir, led = self._legacy(tmp_path, "run-a", "s1")
        ledger.migrate_eval_files(tmp_path, led)
        again = ledger.migrate_eval_files(tmp_path, led)

        assert again["skipped"] == 1
        assert again["merged"] == 0
        assert (run_dir / "evals" / "eval-s1.json").exists(), "originals stay for the operator"

    def test_ledger_rows_are_repointed_so_nothing_is_indexed_twice(self, tmp_path):
        """The move renames the file a row names. Measured before this: 63 rows
        became 110, the same evaluation under two pointers."""
        _, led = self._legacy(tmp_path, "run-a", "s1")
        ledger.migrate_eval_files(tmp_path, led)
        ledger.rebuild_ledger(tmp_path, led)

        rows = ledger.read_records(led)
        assert len(rows) == 1
        assert rows[0]["result_path"] == "run-a/evals/s1.json"

    def test_legacy_files_are_not_read_as_documents(self, tmp_path):
        """Both layouts coexist during the migration window."""
        _, led = self._legacy(tmp_path, "run-a", "s1")
        ledger.migrate_eval_files(tmp_path, led)
        recovered, _ = ledger.rebuild_ledger(tmp_path, led)
        assert recovered == 1

    def test_an_untierable_document_is_not_indexed(self, tmp_path):
        """It would hash into a tier of (method, None, None, ...) and sort to year
        1 AD -- and since tiers rank by coverage, a pile of them would become the
        DEFAULT curve. The document stays on disk; only the index is withheld."""
        run_dir, led = self._legacy(tmp_path, "run-a", "s1", record=False, ledger_row=False)
        ledger.migrate_eval_files(tmp_path, led)
        recovered, _ = ledger.rebuild_ledger(tmp_path, led)

        assert (run_dir / "evals" / "s1.json").exists()
        assert recovered == 0


class TestRepointDoesNotLoseRows:
    """`--migrate` rewrites the ledger, so anything it drops is gone for good."""

    def _legacy(self, tmp_path, run_id, slug):
        evals = tmp_path / run_id / "evals"
        evals.mkdir(parents=True, exist_ok=True)
        (evals / f"eval-{slug}.json").write_text(
            json.dumps({"run_id": run_id, "results": {"exploitability_mbb": 12.0, "n": 3}})
        )
        return evals

    def test_a_torn_line_survives_the_rewrite(self, tmp_path):
        """`--rebuild` preserves rows it cannot regenerate; `--migrate` rewriting
        from the PARSED rows deleted them instead."""
        self._legacy(tmp_path, "run-a", "s1")
        led = tmp_path / "led.jsonl"
        led.write_text(
            json.dumps(
                {
                    "run_id": "run-a",
                    "timestamp": "2026-07-18T00:00:00",
                    "knobs": {"scorer": "myopic"},
                    "result_path": "run-a/evals/eval-s1.json",
                }
            )
            + "\n"
            + '{"run_id": "run-torn", "resu\n'
        )

        ledger.migrate_eval_files(tmp_path, led)

        lines = [line for line in led.read_text().splitlines() if line.strip()]
        assert any("run-torn" in line for line in lines), "the torn line was deleted"
        assert len(lines) == 2

    def test_a_row_is_not_repointed_at_a_document_that_was_never_written(self, tmp_path):
        """Migration skips a payload it cannot read; repointing it anyway leaves
        the row naming a file that does not exist."""
        evals = self._legacy(tmp_path, "run-a", "s1")
        (evals / "eval-s1.json").write_text("{ not json")
        led = tmp_path / "led.jsonl"
        led.write_text(
            json.dumps(
                {
                    "run_id": "run-a",
                    "timestamp": "2026-07-18T00:00:00",
                    "knobs": {"scorer": "myopic"},
                    "result_path": "run-a/evals/eval-s1.json",
                }
            )
            + "\n"
        )

        ledger.migrate_eval_files(tmp_path, led)

        pointer = json.loads(led.read_text().strip())["result_path"]
        assert (tmp_path / pointer).exists(), f"row points at a missing file: {pointer}"
