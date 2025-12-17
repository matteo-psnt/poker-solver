"""
Training metrics tracking and logging.

Tracks key metrics during CFR training: utilities, iteration count,
timing, convergence indicators, etc.
"""

import time
from collections import deque
from typing import Deque, Dict, List

import numpy as np


class MetricsTracker:
    """
    Tracks and logs training metrics.

    Maintains running averages, convergence indicators, and timing information
    for MCCFR training runs.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of rolling window for averages
        """
        self.window_size = window_size

        # Iteration tracking
        self.iteration = 0
        self.start_time = time.time()
        self.last_log_time = time.time()

        # Utility tracking
        self.utilities: List[float] = []
        self.utility_window: Deque[float] = deque(maxlen=window_size)

        # Infoset tracking
        self.infoset_counts: List[int] = []
        self.infoset_window: Deque[int] = deque(maxlen=window_size)

        # Timing tracking
        self.iteration_times: Deque[float] = deque(maxlen=window_size)

    def log_iteration(
        self,
        iteration: int,
        utility: float,
        num_infosets: int,
    ):
        """
        Log metrics for a training iteration.

        Args:
            iteration: Iteration number
            utility: Player 0 utility for this iteration
            num_infosets: Total number of infosets discovered
        """
        current_time = time.time()
        iter_time = current_time - self.last_log_time

        # Update tracking
        self.iteration = iteration
        self.utilities.append(utility)
        self.utility_window.append(utility)
        self.infoset_counts.append(num_infosets)
        self.infoset_window.append(num_infosets)
        self.iteration_times.append(iter_time)

        self.last_log_time = current_time

    def get_avg_utility(self) -> float:
        """
        Get average utility over window.

        Returns:
            Average utility (recent window)
        """
        if not self.utility_window:
            return 0.0
        return float(np.mean(list(self.utility_window)))

    def get_utility_std(self) -> float:
        """
        Get utility standard deviation over window.

        Returns:
            Utility standard deviation
        """
        if len(self.utility_window) < 2:
            return 0.0
        return float(np.std(list(self.utility_window)))

    def get_avg_infosets(self) -> float:
        """
        Get average infoset count over window.

        Returns:
            Average infoset count
        """
        if not self.infoset_window:
            return 0.0
        return float(np.mean(list(self.infoset_window)))

    def get_iterations_per_second(self) -> float:
        """
        Get iterations per second (recent window).

        Returns:
            Iterations/second
        """
        if not self.iteration_times:
            return 0.0

        avg_time = np.mean(list(self.iteration_times))
        if avg_time == 0:
            return 0.0

        return 1.0 / avg_time

    def get_elapsed_time(self) -> float:
        """
        Get total elapsed training time in seconds.

        Returns:
            Elapsed time (seconds)
        """
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary of metric names to values
        """
        return {
            "iteration": self.iteration,
            "avg_utility": self.get_avg_utility(),
            "utility_std": self.get_utility_std(),
            "avg_infosets": self.get_avg_infosets(),
            "iter_per_sec": self.get_iterations_per_second(),
            "elapsed_time": self.get_elapsed_time(),
        }

    def print_summary(self):
        """Print formatted metrics summary."""
        summary = self.get_summary()

        print(f"\n{'=' * 60}")
        print(f"Iteration: {summary['iteration']:,}")
        print(
            f"Avg Utility (last {self.window_size}): {summary['avg_utility']:+.2f} "
            f"(±{summary['utility_std']:.2f})"
        )
        print(f"Avg Infosets: {summary['avg_infosets']:.0f}")
        print(f"Speed: {summary['iter_per_sec']:.1f} iter/s")
        print(f"Elapsed: {_format_time(summary['elapsed_time'])}")
        print(f"{'=' * 60}\n")

    def get_progress_string(self, target_iterations: int) -> str:
        """
        Get progress bar string.

        Args:
            target_iterations: Total iterations target

        Returns:
            Progress string with ETA
        """
        progress = self.iteration / target_iterations
        elapsed = self.get_elapsed_time()

        if progress > 0:
            eta = (elapsed / progress) - elapsed
        else:
            eta = 0

        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        return (
            f"[{bar}] {progress * 100:.1f}% "
            f"({self.iteration:,}/{target_iterations:,}) "
            f"ETA: {_format_time(eta)}"
        )

    def __str__(self) -> str:
        """String representation."""
        return f"MetricsTracker(iter={self.iteration}, avg_util={self.get_avg_utility():+.2f})"


def _format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
