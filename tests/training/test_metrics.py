"""Tests for metrics tracking."""

import time

import numpy as np
import pytest

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action, ActionType
from src.game.state import Street
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

    def test_solver_quality_metrics_with_sampler(self):
        """Test solver-quality metrics with infoset sampler."""
        tracker = MetricsTracker(sample_size=10)

        # Create mock infosets with different characteristics
        def create_mock_sampler():
            # Create some sample infosets
            infosets = []

            # Infoset with positive regrets
            key1 = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="c",
                preflop_hand=None,
                postflop_bucket=0,
                spr_bucket=1,
            )
            actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]
            infoset1 = InfoSet(key1, actions)
            infoset1.regrets = np.array([10.0, 5.0], dtype=np.float32)
            infosets.append(infoset1)

            # Infoset with all zero regrets
            key2 = InfoSetKey(
                player_position=1,
                street=Street.FLOP,
                betting_sequence="b0.5",
                preflop_hand=None,
                postflop_bucket=1,
                spr_bucket=1,
            )
            infoset2 = InfoSet(key2, actions)
            infoset2.regrets = np.array([0.0, 0.0], dtype=np.float32)
            infosets.append(infoset2)

            # Infoset with uniform strategy (3 actions)
            key3 = InfoSetKey(
                player_position=0,
                street=Street.TURN,
                betting_sequence="c-b0.5",
                preflop_hand=None,
                postflop_bucket=2,
                spr_bucket=1,
            )
            actions3 = [
                Action(ActionType.FOLD),
                Action(ActionType.CALL),
                Action(ActionType.RAISE, 10),
            ]
            infoset3 = InfoSet(key3, actions3)
            infoset3.regrets = np.array(
                [0.0, 0.0, 0.0], dtype=np.float32
            )  # All zero -> uniform strategy
            infosets.append(infoset3)

            return lambda n: infosets

        sampler = create_mock_sampler()

        # Log iteration with sampler
        tracker.log_iteration(iteration=1, utility=10.0, num_infosets=100, infoset_sampler=sampler)

        # Verify quality metrics were computed
        assert tracker.get_mean_pos_regret() > 0
        assert tracker.get_max_pos_regret() > 0
        assert tracker.get_zero_regret_pct() > 0  # Should have some zero-regret infosets
        assert tracker.get_avg_entropy() >= 0

        # Verify summary includes quality metrics
        summary = tracker.get_summary()
        assert "mean_pos_regret" in summary
        assert "max_pos_regret" in summary
        assert "zero_regret_pct" in summary
        assert "avg_entropy" in summary
        assert "uniform_strategy_pct" in summary

    def test_entropy_computation(self):
        """Test normalized entropy computation for strategies."""
        # Uniform distribution should have entropy = 1.0 (maximum)
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        uniform_entropy = MetricsTracker._compute_normalized_entropy(uniform_probs)
        assert uniform_entropy == pytest.approx(1.0, rel=0.01)

        # Deterministic distribution should have entropy = 0.0 (minimum)
        deterministic_probs = np.array([1.0, 0.0, 0.0, 0.0])
        deterministic_entropy = MetricsTracker._compute_normalized_entropy(deterministic_probs)
        assert deterministic_entropy == pytest.approx(0.0, abs=1e-6)

        # Mixed distribution should have intermediate entropy in (0, 1)
        mixed_probs = np.array([0.7, 0.2, 0.1, 0.0])
        mixed_entropy = MetricsTracker._compute_normalized_entropy(mixed_probs)
        assert 0 < mixed_entropy < 1.0

        # Test that normalized entropy is comparable across action counts
        # 2-action uniform should also have entropy = 1.0
        uniform_2action = np.array([0.5, 0.5])
        entropy_2action = MetricsTracker._compute_normalized_entropy(uniform_2action)
        assert entropy_2action == pytest.approx(1.0, rel=0.01)

    def test_fallback_rate_tracking(self):
        """Test fallback rate tracking."""
        tracker = MetricsTracker()

        # Log iterations with fallback rate
        tracker.log_iteration(1, 10.0, 100, fallback_rate=0.05)
        tracker.log_iteration(2, 15.0, 150, fallback_rate=0.10)

        # Verify fallback rate is tracked
        assert tracker.get_fallback_rate() > 0
        avg_fallback = tracker.get_fallback_rate()
        assert avg_fallback == pytest.approx(0.075)  # Average of 0.05 and 0.10

        # Verify summary includes fallback rate
        summary = tracker.get_summary()
        assert "fallback_rate" in summary

    def test_compact_summary_format(self):
        """Test compact summary format for progress bars."""
        tracker = MetricsTracker()

        # Create mock sampler
        def mock_sampler(n):
            key = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="c",
                preflop_hand=None,
                postflop_bucket=0,
                spr_bucket=1,
            )
            actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]
            infoset = InfoSet(key, actions)
            infoset.regrets = np.array([5.0, 3.0], dtype=np.float32)
            return [infoset]

        # Log with quality metrics
        tracker.log_iteration(1, 10.0, 100, infoset_sampler=mock_sampler, fallback_rate=0.02)

        # Get compact summary
        compact = tracker.get_compact_summary()

        # Should include key metrics in compact format
        assert "it/s" in compact
        assert "IS" in compact
        assert "R+" in compact or "H=" in compact  # Regret or entropy
