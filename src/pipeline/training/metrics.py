"""
Training metrics tracking and logging.

Tracks key metrics during CFR training: utilities, iteration count,
timing, convergence indicators, solver-quality metrics, etc.
"""

import time
from collections import deque
from collections.abc import Callable

import numpy as np


class MetricsTracker:
    """
    Tracks and logs training metrics.

    Maintains running averages, convergence indicators, timing information,
    and solver-quality metrics for MCCFR training runs.
    """

    def __init__(self, window_size: int = 100, sample_size: int = 1000):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of rolling window for averages
            sample_size: Number of infosets to sample for solver-quality metrics
        """
        self.window_size = window_size
        self.sample_size = sample_size

        # Iteration tracking
        self.iteration = 0
        self.start_time = time.time()
        self.last_log_time = time.time()

        # For accurate iterations/second calculation
        self.window_start_time = time.time()
        self.window_start_iteration = 0

        # Utility tracking
        self.utility_window: deque[float] = deque(maxlen=window_size)

        # Infoset tracking
        self.infoset_window: deque[int] = deque(maxlen=window_size)

        # Solver-quality metrics (CFR health indicators)
        self.mean_pos_regret_window: deque[float] = deque(maxlen=window_size)
        self.max_pos_regret_window: deque[float] = deque(maxlen=window_size)
        self.zero_regret_pct_window: deque[float] = deque(maxlen=window_size)
        self.avg_entropy_window: deque[float] = deque(maxlen=window_size)
        self.uniform_strategy_pct_window: deque[float] = deque(maxlen=window_size)

    def log_iteration(
        self,
        iteration: int,
        utility: float,
        num_infosets: int,
        infoset_sampler: Callable | None = None,
    ):
        """
        Log metrics for a training iteration.

        Args:
            iteration: Iteration number
            utility: Player 0 utility for this iteration
            num_infosets: Total number of infosets discovered
            infoset_sampler: Optional callable that returns sampled infosets for quality metrics
        """
        current_time = time.time()

        # Update basic tracking
        self.iteration = iteration
        self.utility_window.append(utility)
        self.infoset_window.append(num_infosets)

        # Compute solver-quality metrics if sampler provided
        if infoset_sampler is not None:
            sampled_infosets = infoset_sampler(self.sample_size)
            if sampled_infosets:
                quality_metrics = self._compute_quality_metrics(sampled_infosets)
                self.mean_pos_regret_window.append(quality_metrics["mean_pos_regret"])
                self.max_pos_regret_window.append(quality_metrics["max_pos_regret"])
                self.zero_regret_pct_window.append(quality_metrics["zero_regret_pct"])
                self.avg_entropy_window.append(quality_metrics["avg_entropy"])
                self.uniform_strategy_pct_window.append(quality_metrics["uniform_strategy_pct"])

        self.last_log_time = current_time

    def record_quality(self, quality: dict[str, float]) -> None:
        """Push an externally-computed quality-metrics dict into the rolling windows.

        The parallel training loop cannot cheaply hand ``log_iteration`` a list of
        ``InfoSet`` objects (the state lives in shared arrays owned by workers), so it
        computes the same metrics directly from those arrays via
        :func:`compute_quality_from_arrays` and feeds the result here. This keeps
        ``get_summary``/``get_compact_summary`` as the single source of truth for the
        health indicators regardless of how they were produced.
        """
        self.mean_pos_regret_window.append(quality["mean_pos_regret"])
        self.max_pos_regret_window.append(quality["max_pos_regret"])
        self.zero_regret_pct_window.append(quality["zero_regret_pct"])
        self.avg_entropy_window.append(quality["avg_entropy"])
        self.uniform_strategy_pct_window.append(quality["uniform_strategy_pct"])

    def _compute_quality_metrics(self, infosets) -> dict[str, float]:
        """
        Compute solver-quality metrics from sampled infosets.

        Args:
            infosets: List of sampled InfoSet objects

        Returns:
            Dictionary of quality metrics
        """
        if not infosets:
            return {
                "mean_pos_regret": 0.0,
                "max_pos_regret": 0.0,
                "zero_regret_pct": 0.0,
                "avg_entropy": 0.0,
                "uniform_strategy_pct": 0.0,
            }

        # Regret metrics
        all_positive_regrets = []
        zero_regret_count = 0

        # Strategy metrics
        entropies = []
        uniform_count = 0

        for infoset in infosets:
            # Positive regrets (learning signal)
            pos_regrets = np.maximum(infoset.regrets, 0)
            if np.any(pos_regrets > 0):
                all_positive_regrets.extend(pos_regrets[pos_regrets > 0])

            # Check if all regrets are zero (no learning signal)
            if np.all(infoset.regrets == 0):
                zero_regret_count += 1

            # Strategy entropy (diversity measure)
            # Normalized entropy is in [0, 1] where 1 = uniform distribution
            strategy = infoset.get_strategy()
            normalized_entropy = self._compute_normalized_entropy(strategy)
            entropies.append(normalized_entropy)

            # Check if strategy is approximately uniform (entropy close to 1.0)
            if normalized_entropy > 0.99:  # Within 1% of max entropy
                uniform_count += 1

        return {
            "mean_pos_regret": float(np.mean(all_positive_regrets))
            if all_positive_regrets
            else 0.0,
            "max_pos_regret": float(np.max(all_positive_regrets)) if all_positive_regrets else 0.0,
            "zero_regret_pct": 100.0 * zero_regret_count / len(infosets),
            "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "uniform_strategy_pct": 100.0 * uniform_count / len(infosets),
        }

    @staticmethod
    def _compute_normalized_entropy(probabilities: np.ndarray) -> float:
        """Deprecated thin wrapper; use :func:`normalized_entropy` directly."""
        return normalized_entropy(probabilities)

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

        Calculates based on total iterations in window divided by elapsed time,
        which is more accurate for parallel training than averaging per-log times.

        Returns:
            Iterations/second
        """
        if self.iteration == 0:
            return 0.0

        # Calculate based on iterations completed in the current window
        current_time = time.time()
        elapsed = current_time - self.window_start_time

        if elapsed <= 0:
            return 0.0

        iterations_in_window = self.iteration - self.window_start_iteration

        # Reset window if we've accumulated enough iterations
        if iterations_in_window >= self.window_size:
            self.window_start_time = current_time
            self.window_start_iteration = self.iteration

        return float(iterations_in_window / elapsed) if elapsed > 0 else 0.0

    def get_elapsed_time(self) -> float:
        """
        Get total elapsed training time in seconds.

        Returns:
            Elapsed time (seconds)
        """
        return time.time() - self.start_time

    def get_mean_pos_regret(self) -> float:
        """Get average mean positive regret over window."""
        if not self.mean_pos_regret_window:
            return 0.0
        return float(np.mean(list(self.mean_pos_regret_window)))

    def get_max_pos_regret(self) -> float:
        """Get average max positive regret over window."""
        if not self.max_pos_regret_window:
            return 0.0
        return float(np.mean(list(self.max_pos_regret_window)))

    def get_zero_regret_pct(self) -> float:
        """Get average percent of zero-regret infosets over window."""
        if not self.zero_regret_pct_window:
            return 0.0
        return float(np.mean(list(self.zero_regret_pct_window)))

    def get_avg_entropy(self) -> float:
        """Get average strategy entropy over window."""
        if not self.avg_entropy_window:
            return 0.0
        return float(np.mean(list(self.avg_entropy_window)))

    def get_uniform_strategy_pct(self) -> float:
        """Get average percent of uniform strategies over window."""
        if not self.uniform_strategy_pct_window:
            return 0.0
        return float(np.mean(list(self.uniform_strategy_pct_window)))

    def get_summary(self) -> dict[str, float | int]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary of metric names to values
        """
        summary = {
            "iteration": self.iteration,
            "avg_utility": self.get_avg_utility(),
            "utility_std": self.get_utility_std(),
            "avg_infosets": self.get_avg_infosets(),
            "iter_per_sec": self.get_iterations_per_second(),
            "elapsed_time": self.get_elapsed_time(),
        }

        # Add solver-quality metrics if available
        if self.mean_pos_regret_window:
            summary.update(
                {
                    "mean_pos_regret": self.get_mean_pos_regret(),
                    "max_pos_regret": self.get_max_pos_regret(),
                    "zero_regret_pct": self.get_zero_regret_pct(),
                    "avg_entropy": self.get_avg_entropy(),
                    "uniform_strategy_pct": self.get_uniform_strategy_pct(),
                }
            )

        return summary

    def print_summary(self):
        """Print formatted metrics summary with solver-quality indicators."""
        summary = self.get_summary()

        print(f"\n{'=' * 80}")
        print(f"Iteration: {summary['iteration']:,}")
        print(
            f"Avg Utility (last {self.window_size}): {summary['avg_utility']:+.2f} "
            f"(±{summary['utility_std']:.2f})"
        )
        print(f"Avg Infosets: {summary['avg_infosets']:.0f}")
        print(f"Speed: {summary['iter_per_sec']:.1f} iter/s")
        print(f"Elapsed: {_format_time(summary['elapsed_time'])}")

        # Print solver-quality metrics if available
        if "mean_pos_regret" in summary:
            print("\nSolver Quality Metrics:")
            print(
                f"  Regrets: mean={summary['mean_pos_regret']:.2e}, "
                f"max={summary['max_pos_regret']:.2e}, "
                f"zero={summary['zero_regret_pct']:.1f}%"
            )
            print(
                f"  Strategy: entropy={summary['avg_entropy']:.3f}, "
                f"uniform={summary['uniform_strategy_pct']:.1f}%"
            )

        print(f"{'=' * 80}\n")

    def get_compact_summary(self) -> str:
        """
        Get compact one-line summary for progress displays.

        Returns:
            Formatted summary string
        """
        summary = self.get_summary()

        parts = [f"{summary['iter_per_sec']:.1f}it/s"]

        # Add solver-quality metrics if available
        if "mean_pos_regret" in summary:
            parts.extend(
                [
                    f"R+={summary['mean_pos_regret']:.1e}",
                    f"H={summary['avg_entropy']:.2f}",
                ]
            )

        return " | ".join(parts)

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


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Normalized Shannon entropy of a probability distribution, in ``[0, 1]``.

    ``0`` = deterministic (all mass on one action), ``1`` = uniform (maximum entropy).
    Normalizing by ``log(num_actions)`` makes the value comparable across infosets
    with different action counts.
    """
    # Filter out zero probabilities to avoid log(0)
    probs = probabilities[probabilities > 0]
    if len(probs) == 0:
        return 0.0

    # Compute raw entropy in nats
    raw_entropy = float(-np.sum(probs * np.log(probs)))

    # Normalize by max possible entropy (log of number of actions)
    num_actions = len(probabilities)
    if num_actions <= 1:
        return 0.0  # No choice available

    max_entropy = np.log(num_actions)
    normalized = raw_entropy / max_entropy

    # Clamp to [0, 1] to handle numerical errors
    return float(np.clip(normalized, 0.0, 1.0))


def compute_quality_from_arrays(
    regrets: np.ndarray,
    action_counts: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, float]:
    """Compute solver-quality metrics directly from the shared storage arrays.

    Mirrors :meth:`MetricsTracker._compute_quality_metrics` (same output schema and
    the same regret-matching strategy definition) but reads ragged rows straight out
    of the flat ``shared_regrets`` / ``shared_action_counts`` arrays, so the parallel
    coordinator can sample health metrics without reconstructing ``InfoSet`` objects.

    Args:
        regrets: ``(capacity, max_actions)`` regret array (may be read live).
        action_counts: ``(capacity,)`` legal-action count per row; ``0`` marks an
            unallocated slot.
        sample_ids: Row indices to sample (already filtered to allocated rows).

    Returns:
        Dict with the same keys as ``_compute_quality_metrics``.
    """
    empty = {
        "mean_pos_regret": 0.0,
        "max_pos_regret": 0.0,
        "zero_regret_pct": 0.0,
        "avg_entropy": 0.0,
        "uniform_strategy_pct": 0.0,
    }
    if len(sample_ids) == 0:
        return empty

    all_positive_regrets: list[float] = []
    zero_regret_count = 0
    entropies: list[float] = []
    uniform_count = 0
    sampled = 0

    for row_id in sample_ids:
        k = int(action_counts[row_id])
        if k <= 0:
            continue
        sampled += 1
        r = np.asarray(regrets[row_id, :k], dtype=np.float64)

        pos = np.maximum(r, 0.0)
        positive = pos[pos > 0]
        if positive.size:
            all_positive_regrets.extend(positive.tolist())

        if np.all(r == 0):
            zero_regret_count += 1

        total = float(pos.sum())
        strategy = pos / total if total > 0 else np.full(k, 1.0 / k)
        entropy = normalized_entropy(strategy)
        entropies.append(entropy)
        if entropy > 0.99:
            uniform_count += 1

    if sampled == 0:
        return empty

    return {
        "mean_pos_regret": float(np.mean(all_positive_regrets)) if all_positive_regrets else 0.0,
        "max_pos_regret": float(np.max(all_positive_regrets)) if all_positive_regrets else 0.0,
        "zero_regret_pct": 100.0 * zero_regret_count / sampled,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "uniform_strategy_pct": 100.0 * uniform_count / sampled,
    }


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
