"""Concurrency safety of evaluation records.

The eval ledger is the one piece of shared mutable state in the system, and the
one piece that has actually lost data: 12 of 14 rows vanished in a parallel sweep
because every writer did read-append-publish on one shared file. The fix is that
an evaluation IS a uniquely-named document under the run directory, and the index
is derived from those documents on every read -- recording writes no index at
all. These tests pin that property.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from pathlib import Path

from src.pipeline.evaluation import ledger


def _provenance(run_id="run-a"):
    return ledger.RunProvenance(
        run_id=run_id,
        git_commit="c" * 40,
        git_dirty=False,
        config_name="quick_test",
        card_abstraction_hash="deadbeef",
        action_config_hash="beefcafe",
    )


def _record_one(run_dir: Path, index: int):
    knobs = {
        "scorer": "myopic",
        "opponent": "blueprint",
        "include_off_tree": False,
        "base_seed": index,
    }
    payload = {
        "op": "evaluate",
        "run_id": run_dir.name,
        "infosets": 10,
        "checkpoint_iteration": 1000 + index,
        "results": {
            "exploitability_mbb": float(index),
            "std_error_mbb": 1.0,
            "num_hands": 2,
            "pair_samples_mbb": [1.0, 2.0],
        },
    }
    return ledger.record_evaluation(
        run_dir=run_dir,
        payload=payload,
        provenance=_provenance(run_dir.name),
        method="lbr",
        estimator="lbr",
        knobs=knobs,
    )


class TestUniqueNaming:
    def test_same_knobs_same_instant_do_not_collide(self, tmp_path):
        # Two boxes evaluating one run with identical knobs is the fan-out shape a
        # noise-floor sweep produces; a timestamp+knobhash name alone can collide.
        slugs = {ledger.eval_slug({"a": 1}) for _ in range(200)}
        assert len(slugs) == 200

    def test_an_evaluation_is_one_file(self, tmp_path):
        """Was two -- a payload and a record summarising it -- plus a ledger row."""
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        result_path, document = _record_one(run_dir, 0)
        assert result_path.exists()
        written = list((run_dir / "evals").glob("*.json"))
        assert len(written) == 1
        assert json.loads(written[0].read_text())["result_path"] == document["result_path"]


class TestConcurrentWriters:
    def test_no_record_is_lost_under_concurrent_evaluation(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: _record_one(run_dir, i), range(24)))

        durable = list((run_dir / "evals").glob("*.json"))
        assert len(durable) == 24, "every writer's durable record must survive"

    def test_rebuild_recovers_rows_a_clobbered_cache_lost(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        ledger_path = tmp_path / "led.jsonl"
        for i in range(5):
            _record_one(run_dir, i)
        ledger.rebuild_ledger(tmp_path, ledger_path)

        # Simulate the clobber: a last-writer-wins publish leaves only one row.
        surviving = ledger.read_records(ledger_path)[-1]
        ledger_path.write_text(json.dumps(surviving) + "\n")
        assert len(ledger.read_records(ledger_path)) == 1

        recovered, preserved = ledger.rebuild_ledger(tmp_path, ledger_path)
        assert recovered == 5
        assert preserved == 0
        assert len(ledger.read_records(ledger_path)) == 5


class TestRebuildIsNonDestructive:
    def test_rows_without_a_record_file_are_preserved(self, tmp_path):
        # Rows written before per-run records existed cannot be regenerated:
        # eval_git_commit, knobs and timestamp exist nowhere else. Rebuild is
        # forward-only, so it must never drop them.
        ledger_path = tmp_path / "led.jsonl"
        legacy = {
            "run_id": "run-old",
            "timestamp": "2020-01-01T00:00:00+00:00",
            "result_path": "data/runs/run-old/evals/eval-gone.json",
            "knobs": {"scorer": "myopic"},
        }
        ledger_path.write_text(json.dumps(legacy) + "\n")

        recovered, preserved = ledger.rebuild_ledger(tmp_path, ledger_path)
        assert (recovered, preserved) == (0, 1)
        assert ledger.read_records(ledger_path)[0]["run_id"] == "run-old"

    def test_legacy_rows_sharing_a_result_path_are_both_kept(self, tmp_path):
        # The real ledger contains four such pairs, from the 07-18 clobber recovery.
        # Deduping them would make the anti-row-loss command lose rows.
        ledger_path = tmp_path / "led.jsonl"
        dup = {
            "run_id": "run-old",
            "timestamp": "2020-01-01T00:00:00+00:00",
            "result_path": "data/runs/run-old/evals/eval-same.json",
        }
        ledger_path.write_text(json.dumps(dup) + "\n" + json.dumps(dup) + "\n")

        recovered, preserved = ledger.rebuild_ledger(tmp_path, ledger_path)
        assert (recovered, preserved) == (0, 2)
        assert len(ledger.read_records(ledger_path)) == 2

    def test_rebuild_is_idempotent(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        ledger_path = tmp_path / "led.jsonl"
        for i in range(3):
            _record_one(run_dir, i)

        ledger.rebuild_ledger(tmp_path, ledger_path)
        first = ledger_path.read_text()
        ledger.rebuild_ledger(tmp_path, ledger_path)
        assert ledger_path.read_text() == first


class TestOrdering:
    def test_records_sort_by_timestamp_not_file_position(self, tmp_path):
        # Interleaved writers append out of order; "latest" must mean latest eval.
        ledger_path = tmp_path / "led.jsonl"
        rows = [
            {"run_id": "r", "timestamp": "2026-01-03T00:00:00+00:00", "n": 3},
            {"run_id": "r", "timestamp": "2026-01-01T00:00:00+00:00", "n": 1},
            {"run_id": "r", "timestamp": "2026-01-02T00:00:00+00:00", "n": 2},
        ]
        ledger_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert [r["n"] for r in ledger.read_records(ledger_path)] == [1, 2, 3]

    def test_latest_record_uses_timestamp_order(self, tmp_path):
        ledger_path = tmp_path / "led.jsonl"
        rows = [
            {"run_id": "r", "timestamp": "2026-01-09T00:00:00+00:00", "n": "newest"},
            {"run_id": "r", "timestamp": "2026-01-01T00:00:00+00:00", "n": "oldest"},
        ]
        ledger_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        latest = ledger.latest_record_for_run("r", ledger_path)
        assert latest is not None
        assert latest["n"] == "newest"

    def test_recorded_timestamps_are_utc_aware(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        _, record = _record_one(run_dir, 0)
        # Boxes in different timezones must produce comparable timestamps.
        assert record["timestamp"].endswith("+00:00")


class TestPayloadResolution:
    def test_falls_back_to_run_dir_when_the_recorded_path_moved(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        _, record = _record_one(run_dir, 0)

        # A ledger pulled from a box that mounted its data elsewhere.
        record["result_path"] = "/nonexistent/elsewhere/" + Path(record["result_path"]).name
        assert ledger.load_payload(record, runs_dir=tmp_path)["run_id"] == "run-a"


class TestReviewFixes:
    """Regressions for the 2026-07-28 review findings."""

    def test_torn_line_does_not_break_the_whole_ledger(self, tmp_path):
        # A half-written final line must not make the ledger unreadable -- that
        # would also disable `--rebuild`, the one command able to repair it.
        path = tmp_path / "led.jsonl"
        good = {"run_id": "r", "timestamp": "2026-01-01T00:00:00+00:00"}
        path.write_text(json.dumps(good) + "\n" + '{"run_id": "torn"')
        rows = ledger.read_records(path)
        assert [r["run_id"] for r in rows] == ["r"]

    def test_rebuild_survives_a_torn_ledger(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        path = tmp_path / "led.jsonl"
        _record_one(run_dir, 0)
        with path.open("a") as fh:
            fh.write('{"half')
        recovered, _ = ledger.rebuild_ledger(tmp_path, path)
        assert recovered == 1

    def test_rebuild_leaves_no_temp_file_behind(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        path = tmp_path / "led.jsonl"
        _record_one(run_dir, 0)
        ledger.rebuild_ledger(tmp_path, path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_naive_legacy_timestamps_are_read_as_local_not_utc(self, tmp_path):
        # Legacy rows were written by datetime.now() -- naive LOCAL time. Reading
        # them as UTC would skew every legacy row by the writer's offset.
        from datetime import datetime

        naive = {"run_id": "old", "timestamp": "2026-07-17T21:43:51.412258"}
        expected = datetime.fromisoformat(naive["timestamp"]).astimezone()
        assert ledger.record_instant(naive) == expected

    def test_unparseable_timestamp_sorts_first_rather_than_raising(self):
        assert ledger.record_instant({"timestamp": "not-a-date"}) < ledger.record_instant(
            {"timestamp": "2020-01-01T00:00:00+00:00"}
        )

    def test_ordering_is_correct_across_both_timestamp_vintages(self, tmp_path):
        from datetime import datetime, timedelta

        # One naive-local row and one UTC-aware row an hour later in real time.
        local = datetime.now().astimezone()
        rows = [
            {
                "run_id": "utc",
                "timestamp": (local + timedelta(hours=1)).astimezone(UTC).isoformat(),
            },
            {"run_id": "naive", "timestamp": local.replace(tzinfo=None).isoformat()},
        ]
        path = tmp_path / "led.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert [r["run_id"] for r in ledger.read_records(path)] == ["naive", "utc"]


class TestTierKeyCoversConditionalKnobs:
    """tier_key and tier_mismatches must encode ONE rule, not two."""

    def _row(self, **knobs):
        return {
            "run_id": "r",
            "method": "lbr",
            "checkpoint_iteration": 1,
            "knobs": {
                "scorer": "lookahead",
                "opponent": "blueprint",
                "include_off_tree": False,
                "base_seed": 7,
                **knobs,
            },
            "results": {"exploitability_mbb": 1.0},
        }

    def test_lookahead_depth_splits_the_tier(self):
        # Previously these hashed into one tier and got plotted on one axis.
        a, b = self._row(lookahead_depth=2), self._row(lookahead_depth=4)
        assert ledger.tier_key(a) != ledger.tier_key(b)

    def test_exact_br_board_budget_splits_the_tier(self):
        a = self._row(num_flops=10)
        b = self._row(num_flops=50)
        assert ledger.tier_key(a) != ledger.tier_key(b)

    def test_tier_key_agrees_with_tier_mismatches(self):
        # The invariant: same key <=> no mismatch reasons.
        a, b = self._row(lookahead_depth=2), self._row(lookahead_depth=4)
        assert ledger.tier_mismatches(a, b) != []
        assert ledger.tier_key(a) != ledger.tier_key(b)

        same = self._row(lookahead_depth=2)
        assert ledger.tier_mismatches(a, same) == []
        assert ledger.tier_key(a) == ledger.tier_key(same)
