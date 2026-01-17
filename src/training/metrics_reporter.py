"""
Metrics reporting for training runs.

Handles high-level metrics formatting, progress display, and training headers.
Wraps MetricsTracker with presentation logic.
"""

from tqdm import tqdm

from src.training.metrics import MetricsTracker


class MetricsReporter:
    """
    High-level metrics reporting for training.

    Responsibilities:
    - Format and display training headers
    - Update progress bars with compact summaries
    - Log batch completion with metrics
    - Print final training summaries

    Extracted from TrainingSession to separate presentation from orchestration.
    """

    def __init__(
        self,
        metrics: MetricsTracker,
        verbose: bool = True,
    ):
        """
        Initialize metrics reporter.

        Args:
            metrics: Underlying metrics tracker
            verbose: Whether to print progress messages
        """
        self._metrics = metrics
        self._verbose = verbose

    @property
    def metrics(self) -> MetricsTracker:
        """Access underlying metrics tracker."""
        return self._metrics

    def print_training_header(
        self,
        num_workers: int,
        num_iterations: int,
        batch_size: int,
        initial_capacity: int,
        max_actions: int,
    ) -> None:
        """
        Display training configuration header.

        Args:
            num_workers: Number of parallel workers
            num_iterations: Total iterations to run
            batch_size: Batch size for parallel training
            initial_capacity: Initial storage capacity
            max_actions: Maximum actions per infoset
        """
        if not self._verbose:
            return

        print("\n🚀 Shared Array Parallel Training")
        print(f"   Workers: {num_workers}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Batch size: {batch_size}")
        print(f"   Initial capacity: {initial_capacity:,}")
        print(f"   Max actions: {max_actions}")
        print("   Mode: Live shared memory arrays")

    def log_batch_utilities(
        self,
        utilities: list[float],
        start_iteration: int,
        total_infosets: int,
    ) -> None:
        """
        Log metrics for each iteration in a batch.

        Args:
            utilities: List of utility values from the batch
            start_iteration: Starting iteration number for this batch
            total_infosets: Total number of infosets discovered
        """
        for i, util in enumerate(utilities):
            iter_num = start_iteration + i + 1
            self._metrics.log_iteration(
                iteration=iter_num,
                utility=util,
                num_infosets=total_infosets,
                infoset_sampler=None,
            )

    def update_progress_bar(
        self,
        progress_bar: tqdm,
        iteration: int,
        total_infosets: int,
        capacity_usage: float,
    ) -> None:
        """
        Update tqdm progress bar with compact metrics summary.

        Args:
            progress_bar: tqdm progress bar instance
            iteration: Current iteration number
            total_infosets: Total number of infosets
            capacity_usage: Storage capacity usage (0.0 to 1.0)
        """
        if not self._verbose or not isinstance(progress_bar, tqdm):
            return

        compact_summary = self._metrics.get_compact_summary()
        progress_bar.set_postfix_str(
            f"iter={iteration} infosets={total_infosets} "
            f"cap={capacity_usage:.0%} | {compact_summary}"
        )

    def print_final_summary(
        self,
        total_iterations: int,
        total_infosets: int,
        elapsed_time: float,
        interrupted: bool,
        fallback_stats: dict[str, float] | None = None,
    ) -> None:
        """
        Print final training summary.

        Args:
            total_iterations: Total iterations completed
            total_infosets: Total infosets discovered
            elapsed_time: Total elapsed time in seconds
            interrupted: Whether training was interrupted
            fallback_stats: Optional fallback statistics from card abstraction
        """
        if not self._verbose:
            return

        if interrupted:
            print("🟡 Training interrupted")
        else:
            print("✅ Shared Array Training complete!")

        print(f"   Iterations: {total_iterations}")
        print(f"   Infosets: {total_infosets:,}")
        print(f"   Time: {elapsed_time:.1f}s")

        if total_iterations > 0:
            print(f"   Speed: {total_iterations / elapsed_time:.2f} iter/s")

        if fallback_stats:
            total_lookups = int(fallback_stats.get("total_lookups", 0))
            fallback_count = int(fallback_stats.get("fallback_count", 0))
            if total_lookups > 0:
                fallback_rate = fallback_stats.get("fallback_rate", 0.0) * 100
                print(
                    f"   Abstraction fallbacks: {fallback_count:,}/{total_lookups:,} "
                    f"({fallback_rate:.2f}%)"
                )
