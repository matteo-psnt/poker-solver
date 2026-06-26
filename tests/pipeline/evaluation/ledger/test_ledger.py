"""Tests for the derived evaluation index and its comparison guard."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.pipeline.evaluation import ledger
from src.pipeline.evaluation.estimators.lbr.config import LBRConfig
from src.pipeline.evaluation.ledger import records as eval_records
from src.pipeline.evaluation.ledger import records as ledger_records
from src.shared import gitinfo
from src.shared.cloudtask import task_log
from tests.test_helpers import seed_ledger


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

    def test_read_roundtrip(self, tmp_path):
        led = tmp_path / "eval_ledger.jsonl"
        seed_ledger(led, {"run_id": "r1", "x": 1}, {"run_id": "r2", "x": 2})
        rows = ledger.read_records(led)
        assert [r["run_id"] for r in rows] == ["r1", "r2"]

    def test_read_missing_ledger_is_empty(self, tmp_path):
        assert ledger.read_records(tmp_path / "absent.jsonl") == []

    def test_latest_record_for_run_returns_last(self, tmp_path):
        led = tmp_path / "l.jsonl"
        seed_ledger(
            led, {"run_id": "r1", "v": 1}, {"run_id": "r1", "v": 2}, {"run_id": "r2", "v": 9}
        )
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
    def test_writes_one_document_and_no_index(self, tmp_path):
        run_dir = tmp_path / "run-x"
        run_dir.mkdir()
        results = _results(base_seed=7)
        knobs = ledger.build_lbr_knobs(_lbr_config(), results)
        payload = {"op": "evaluate", "infosets": 10, "results": results}

        result_path, record = ledger.record_evaluation(
            run_dir=run_dir,
            payload=payload,
            provenance=_fake_provenance("run-x"),
            method="lbr",
            estimator="lbr",
            knobs=knobs,
        )

        assert result_path.exists()
        assert result_path.parent == run_dir / "evals"
        # The document is the ONLY thing written. Recording used to also append
        # to a module-default `data/eval_ledger.jsonl`, so every cloud eval wrote
        # a stored index the architecture says is derived on read.
        assert list(tmp_path.rglob("*.jsonl")) == []
        assert record["run_id"] == "run-x"
        # Stored run-relative, not CWD-relative: the pointer must mean the same
        # thing on a machine that mounts its runs directory somewhere else.
        assert record["result_path"] == f"run-x/evals/{result_path.name}"
        # The document round-trips to its full payload, including per-hand samples.
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


class TestWhichGameTheNumberDescribes:
    """Two exact_br rows at identical knobs from different GAMES used to hash
    into one tier: `tier_mismatches` checked the abstraction and action hashes
    pairwise while `tier_key` grouped without them, and neither knew about a
    rules change under a fixed action config (the limp fix)."""

    def _row(self, **over):
        row = {
            "method": "exact_br",
            "card_abstraction_hash": "a1542e88be59da97",
            "action_config_hash": "eb598d79",
            "eval_tree_fingerprint": "b0367ae018a58a2f",
            "knobs": {"num_flops": 4, "num_turns": 2, "num_rivers": 2, "base_seed": 7},
        }
        row.update(over)
        return row

    def test_a_different_tree_is_a_different_tier(self):
        other = self._row(eval_tree_fingerprint="0000000000000000")
        assert ledger.tier_key(self._row()) != ledger.tier_key(other)

    def test_a_different_abstraction_is_a_different_tier(self):
        other = self._row(card_abstraction_hash="deadbeef")
        assert ledger.tier_key(self._row()) != ledger.tier_key(other)

    def test_the_label_names_every_knob_the_key_splits_on(self):
        """Otherwise two genuinely different tiers render as one string and
        `--tier 1` selects something the operator cannot see."""
        label = ledger.tier_label(self._row())
        for value in ("a1542e88", "eb598d79", "b0367ae0"):
            assert value in label, label

    def test_a_matched_pair_still_pairs(self):
        assert ledger.tier_key(self._row()) == ledger.tier_key(self._row())


RESOLVER_RESULTS = {
    "num_deals": 1000,
    "seed": 7,
    "leaf_continuation_fraction": 0.5,
    "resolver_max_iterations": 64,
    "allin_runouts": 1,
    "root_prior_weight": 100.0,
    "leaf_rollouts": 8,
    "resolver_blend_alpha": 0.35,
}


def _resolver_row(**over):
    return {
        "method": "resolver_match",
        "knobs": ledger.build_resolver_match_knobs(RESOLVER_RESULTS | over),
        "results": {"num_hands": 2000},
    }


class TestEveryRecordedKnobIsActuallyTiered:
    """A knob a builder records but neither guard lists is worse than no knob:
    the row LOOKS identified while `tier_key` groups it with its own control and
    `tier_mismatches` pairs them. Every resolver knob was in that state -- the
    A/B those knobs exist to express was the one comparison silently mixed.
    """

    def _covered(self, knobs):
        listed = set(ledger.TIER_KNOBS) | set(ledger.CONDITIONAL_TIER_KNOBS) | {"base_seed"}
        # `hands`/`num_deals` are precision, not instrument, and `num_hands` is
        # checked separately by tier_mismatches.
        return {k for k in knobs if k not in listed and k not in ("hands", "num_deals")}

    def test_resolver_match_knobs_are_all_tiered(self):
        knobs = ledger.build_resolver_match_knobs(RESOLVER_RESULTS)
        assert self._covered(knobs) == set()

    def test_deployed_lbr_knobs_are_all_tiered(self):
        knobs = ledger.build_lbr_knobs(
            _lbr_config(opponent="deployed"),
            _results()
            | {
                "resolver_iterations": 64,
                "resolver_blend_alpha": 0.35,
                "resolver_root_prior_weight": 100.0,
            },
        )
        assert self._covered(knobs) == set()

    def test_a_different_blend_alpha_is_a_different_tier(self):
        """alpha=0 scored +2140 where the shipped alpha scored -781.6, so these
        two rows are not two measurements of one thing."""
        shipped, bare = _resolver_row(), _resolver_row(resolver_blend_alpha=0.0)
        assert ledger.tier_key(shipped) != ledger.tier_key(bare)
        assert any("resolver_blend_alpha" in r for r in ledger.tier_mismatches(shipped, bare))

    def test_a_different_leaf_valuation_is_a_different_tier(self):
        a, b = _resolver_row(), _resolver_row(leaf_continuation_fraction=1.0)
        assert ledger.tier_key(a) != ledger.tier_key(b)
        assert any("leaf_continuation_fraction" in r for r in ledger.tier_mismatches(a, b))

    def test_the_label_names_the_resolver_knobs(self):
        label = ledger.tier_label(_resolver_row())
        assert "resolver_blend_alpha=0.35" in label, label
        assert "leaf_continuation_fraction=0.5" in label, label


class TestLoadPayload:
    def test_missing_payload_raises(self):
        with pytest.raises(FileNotFoundError):
            ledger.load_payload({"run_id": "r", "result_path": "/no/such/file.json"})


class TestRebuildSkipsLegacyShapes:
    """Two pre-substrate shapes still sit on the share: `eval-*.json` (a payload
    with almost no provenance) and `record-*.json` (provenance plus a four-key
    summary). A legacy record points at the OLD filename, so reading both enters
    one evaluation twice under two pointers -- measured: 63 rows became 110.
    """

    def _document(self, tmp_path, run_id, slug, **overrides):
        run_dir = tmp_path / run_id
        document = {
            "run_id": run_id,
            "timestamp": "2026-07-18T00:00:00",
            "knobs": {"scorer": "myopic"},
            "results": {"exploitability_mbb": 12.0},
            "result_path": f"{run_id}/evals/{slug}.json",
            **overrides,
        }
        ledger.write_eval(run_dir, document, slug)
        return run_dir / "evals"

    def test_a_document_beside_its_legacy_halves_is_indexed_once(self, tmp_path):
        evals = self._document(tmp_path, "run-a", "s1")
        (evals / "eval-s1.json").write_text(
            json.dumps({"run_id": "run-a", "results": {"exploitability_mbb": 12.0}})
        )
        (evals / "record-s1.json").write_text(
            json.dumps(
                {
                    "run_id": "run-a",
                    "timestamp": "2026-07-18T00:00:00",
                    "knobs": {"scorer": "myopic"},
                    "result_path": "run-a/evals/eval-s1.json",
                }
            )
        )
        led = tmp_path / "led.jsonl"

        recovered, _ = ledger.rebuild_ledger(tmp_path, led)
        assert recovered == 1
        assert [r["result_path"] for r in ledger.read_records(led)] == ["run-a/evals/s1.json"]

    def test_an_untierable_document_is_not_indexed(self, tmp_path):
        """It would hash into a tier of (method, None, None, ...) and sort to year
        1 AD -- and since tiers rank by coverage, a pile of them would become the
        DEFAULT curve. The document stays on disk; only the index is withheld."""
        evals = self._document(tmp_path, "run-a", "s1", knobs={}, timestamp="")
        led = tmp_path / "led.jsonl"

        recovered, _ = ledger.rebuild_ledger(tmp_path, led)
        assert (evals / "s1.json").exists()
        assert recovered == 0


class TestEvalDocumentNamesItsTask:
    """Without a task id there is no key joining a number to the task that made it.

    Correlating by timestamp is what fails here specifically: concurrent
    evaluations of ONE run have completely overlapping intervals, so every
    document falls inside every task's window.
    """

    def _build(self, tmp_path):
        return ledger.build_record(
            provenance=_fake_provenance(),
            method="exact_br",
            estimator="public_tree_exact_br",
            checkpoint_iteration=150_000_000,
            infosets=1,
            knobs={"base_seed": 7},
            results={},
            result_path=tmp_path / "evals" / "x.json",
            timestamp="2026-08-04T09:04:56+00:00",
        )

    def test_it_records_the_batch_task_that_produced_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv(task_log.TASK_ID_ENV, "score-production-1095-150M-seed7-090456-18475")
        assert self._build(tmp_path)["task_id"].endswith("seed7-090456-18475")

    def test_off_a_node_it_is_empty_rather_than_a_placeholder(self, tmp_path, monkeypatch):
        """An evaluation run anywhere else genuinely has no task to point at."""
        monkeypatch.delenv(task_log.TASK_ID_ENV, raising=False)
        assert self._build(tmp_path)["task_id"] == ""

    def test_the_ledger_row_carries_it_through(self, tmp_path, monkeypatch):
        """The index is derived from the document; a field it drops is unqueryable."""
        monkeypatch.setenv(task_log.TASK_ID_ENV, "t-1")
        assert ledger.ledger_row(self._build(tmp_path))["task_id"] == "t-1"


class TestWhichWorktreeProducedTheNumber:
    """Two commits are recorded because two matter; two branches, for the same reason.

    The train and eval commits are routinely IDENTICAL across arms that differ
    entirely — experiments live in parallel git worktrees, and a worktree carries
    its change uncommitted for as long as it is being iterated on. Without the
    branch, `report --experiment` pairs arms it cannot tell apart.
    """

    def test_both_branches_are_recorded(self, monkeypatch):
        monkeypatch.setattr(ledger_records, "get_git_branch", lambda: "main")
        provenance = ledger.RunProvenance(
            run_id="run-x",
            git_commit="cafebabe" * 5,
            git_dirty=True,
            config_name="quick_test",
            card_abstraction_hash="deadbeef",
            action_config_hash="beefcafe",
            git_branch="worktree-hybrid-kernels",
        )

        record = ledger_records.build_record(
            provenance=provenance,
            method="exact_br",
            estimator="exact",
            infosets=1,
            knobs={},
            results={},
            result_path=Path("evals/x.json"),
            timestamp="2026-08-06T00:00:00+00:00",
        )

        assert record["train_git_branch"] == "worktree-hybrid-kernels"
        assert record["eval_git_branch"] == "main", "measured somewhere else — a separate fact"

    def test_a_legacy_provenance_still_builds(self):
        """Every row already on the share has no branch; none of them is unloadable."""
        record = ledger_records.build_record(
            provenance=_fake_provenance(),
            method="exact_br",
            estimator="exact",
            infosets=1,
            knobs={},
            results={},
            result_path=Path("evals/x.json"),
            timestamp="2026-08-06T00:00:00+00:00",
        )
        assert record["train_git_branch"] is None


class TestTheRecordNamesTheCodeItRan:
    """A commit plus a dirty bit is not a complete answer; the snapshot is.

    Measured 2026-08-10: of 56 code snapshots published to the share, SEVEN were
    named by any record at all, and all seven by task records written after
    2026-08-02. The other 49 were indistinguishable from garbage while being the
    only copy of code states that included uncommitted work — because the run
    that used one never wrote down which one. Deleting them was considered and
    refused on exactly that ground.
    """

    def test_an_eval_document_carries_both_snapshots(self, monkeypatch):
        monkeypatch.setenv(gitinfo.SNAPSHOT_ENV, "code-20260810_120000")
        record = eval_records.build_record(
            provenance=eval_records.RunProvenance(
                run_id="run-a",
                git_commit="abc1234",
                git_dirty=False,
                config_name="production",
                card_abstraction_hash="h",
                action_config_hash="a",
                code_snapshot="code-20260801_090000",
            ),
            method="exact_br",
            estimator="exact_br",
            infosets=10,
            knobs={"scorer": "exact_br"},
            results={"exploitability_mbb": 1.0},
            result_path=Path("x.json"),
            timestamp="2026-08-10T00:00:00",
        )

        assert record["train_code_snapshot"] == "code-20260801_090000", (
            "the code that produced the CHECKPOINT"
        )
        assert record["eval_code_snapshot"] == "code-20260810_120000", (
            "the code that MEASURED it — a different tarball, and the reason both exist"
        )

    def test_absent_off_a_node_rather_than_invented(self, monkeypatch):
        """Run anywhere else there is no snapshot: the working tree is the code,
        and the commit already describes it as well as anything can."""
        monkeypatch.delenv(gitinfo.SNAPSHOT_ENV, raising=False)
        assert gitinfo.get_code_snapshot() is None
