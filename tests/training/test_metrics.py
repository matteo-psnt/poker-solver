"""Tests for metrics tracking."""

import time

import pytest

from src.training.metrics import MetricsTracker


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_create_tracker(self):
        tracker = MetricsTracker(window_size=10)

        assert tracker.iteration == 0
        assert tracker.window_size == 10
        assert len(tracker.utilities) == 0

    def test_log_iteration(self):
        tracker = MetricsTracker()

        tracker.log_iteration(iteration=1, utility=10.0, num_infosets=100)

        assert tracker.iteration == 1
        assert len(tracker.utilities) == 1
        assert len(tracker.infoset_counts) == 1

    def test_get_avg_utility(self):
        tracker = MetricsTracker(window_size=3)

        tracker.log_iteration(1, 10.0, 100)
        tracker.log_iteration(2, 20.0, 200)
        tracker.log_iteration(3, 30.0, 300)

        avg = tracker.get_avg_utility()
        assert avg == pytest.approx(20.0)

    def test_get_utility_std(self):
        tracker = MetricsTracker(window_size=3)

        tracker.log_iteration(1, 10.0, 100)
        tracker.log_iteration(2, 20.0, 200)
        tracker.log_iteration(3, 30.0, 300)

        std = tracker.get_utility_std()
        assert std > 0  # Should have non-zero std

    def test_get_avg_infosets(self):
        tracker = MetricsTracker(window_size=3)

        tracker.log_iteration(1, 10.0, 100)
        tracker.log_iteration(2, 20.0, 200)
        tracker.log_iteration(3, 30.0, 300)

        avg = tracker.get_avg_infosets()
        assert avg == pytest.approx(200.0)

    def test_window_size_limits(self):
        tracker = MetricsTracker(window_size=2)

        tracker.log_iteration(1, 10.0, 100)
        tracker.log_iteration(2, 20.0, 200)
        tracker.log_iteration(3, 30.0, 300)

        # Should only average last 2
        avg = tracker.get_avg_utility()
        assert avg == pytest.approx(25.0)  # (20 + 30) / 2

    def test_get_elapsed_time(self):
        tracker = MetricsTracker()
        time.sleep(0.01)  # Wait a bit

        elapsed = tracker.get_elapsed_time()
        assert elapsed > 0

    def test_get_iterations_per_second(self):
        tracker = MetricsTracker()

        tracker.log_iteration(1, 10.0, 100)
        time.sleep(0.01)
        tracker.log_iteration(2, 20.0, 200)

        iter_per_sec = tracker.get_iterations_per_second()
        assert iter_per_sec > 0

    def test_get_summary(self):
        tracker = MetricsTracker()

        tracker.log_iteration(1, 10.0, 100)

        summary = tracker.get_summary()

        assert "iteration" in summary
        assert "avg_utility" in summary
        assert "avg_infosets" in summary
        assert "iter_per_sec" in summary
        assert "elapsed_time" in summary

    def test_print_summary(self):
        tracker = MetricsTracker()

        tracker.log_iteration(1, 10.0, 100)

        # Should not raise
        tracker.print_summary()

    def test_get_progress_string(self):
        tracker = MetricsTracker()

        tracker.log_iteration(50, 10.0, 100)

        progress = tracker.get_progress_string(target_iterations=100)

        assert "50.0%" in progress or "50%" in progress
        assert "ETA" in progress

    def test_str_representation(self):
        tracker = MetricsTracker()

        tracker.log_iteration(1, 10.0, 100)

        s = str(tracker)
        assert "MetricsTracker" in s
        assert "iter=1" in s
