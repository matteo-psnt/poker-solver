"""Tests for per-run metrics history: the JSONL writer and array-based quality."""

import json

import numpy as np

from src.pipeline.training.metrics import MetricsTracker, compute_quality_from_arrays
from src.pipeline.training.metrics_history import MetricsHistoryWriter


class TestMetricsHistoryWriter:
    def test_append_writes_one_line_per_row(self, tmp_path):
        path = tmp_path / "nested" / "metrics.jsonl"
        writer = MetricsHistoryWriter(path)
        writer.append({"iteration": 100, "avg_utility": 1.5})
        writer.append({"iteration": 200, "avg_utility": -0.5})

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["iteration"] == 100
        assert json.loads(lines[1])["avg_utility"] == -0.5

    def test_resume_appends_rather_than_truncates(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        MetricsHistoryWriter(path).append({"iteration": 1})
        # A second writer (as on resume) must continue the same file.
        MetricsHistoryWriter(path).append({"iteration": 2})
        assert len(path.read_text().strip().splitlines()) == 2

    def test_write_error_disables_without_raising(self, tmp_path):
        # Point the writer at a path whose parent is a file → mkdir/open fails.
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        writer = MetricsHistoryWriter(blocker / "sub" / "metrics.jsonl")
        writer.append({"iteration": 1})  # must not raise
        assert writer._disabled is True


class TestComputeQualityFromArrays:
    def test_matches_expected_semantics(self):
        # capacity 4, max_actions 3. Row 0 unallocated (action_count 0).
        regrets = np.zeros((4, 3), dtype=np.float64)
        action_counts = np.array([0, 2, 2, 3], dtype=np.int32)
        regrets[1, :2] = [10.0, 5.0]  # positive regrets, non-uniform strategy
        # rows 2 and 3 stay all-zero → zero-regret and uniform strategy

        q = compute_quality_from_arrays(regrets, action_counts, np.array([1, 2, 3]))

        assert q["mean_pos_regret"] == 7.5  # mean of [10, 5]
        assert q["max_pos_regret"] == 10.0
        assert q["zero_regret_pct"] == 200.0 / 3  # rows 2,3 of 3 sampled
        assert q["uniform_strategy_pct"] == 200.0 / 3  # the two all-zero rows
        assert 0.0 < q["avg_entropy"] <= 1.0

    def test_skips_unallocated_rows(self):
        regrets = np.zeros((2, 2), dtype=np.float64)
        action_counts = np.array([0, 0], dtype=np.int32)
        q = compute_quality_from_arrays(regrets, action_counts, np.array([0, 1]))
        assert q["mean_pos_regret"] == 0.0
        assert q["zero_regret_pct"] == 0.0

    def test_empty_sample_returns_zeros(self):
        q = compute_quality_from_arrays(
            np.zeros((1, 2)), np.array([2], dtype=np.int32), np.array([], dtype=int)
        )
        assert q["avg_entropy"] == 0.0


class TestRecordQuality:
    def test_populates_windows_and_summary(self):
        tracker = MetricsTracker()
        tracker.log_iteration(1, utility=1.0, num_infosets=10)
        tracker.record_quality(
            {
                "mean_pos_regret": 3.0,
                "max_pos_regret": 9.0,
                "zero_regret_pct": 20.0,
                "avg_entropy": 0.5,
                "uniform_strategy_pct": 10.0,
            }
        )
        assert tracker.get_mean_pos_regret() == 3.0
        assert "mean_pos_regret" in tracker.get_summary()
        assert "R+" in tracker.get_compact_summary() or "H=" in tracker.get_compact_summary()
