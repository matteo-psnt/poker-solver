"""Memory telemetry, and keeping progress bars out of redirected logs.

Both exist because of the same incident: three legs stalled with every worker
ALIVE and none producing -- memory pressure rather than a crash -- and neither
question could be answered afterwards. Nothing recorded how much memory the run
used, and the progress-bar repaints had made the log too large to download.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

from src.shared.procinfo import rss_mb


class TestRssReading:
    def test_returns_none_rather_than_zero_when_unreadable(self):
        # None and 0 must not be confused: one is "could not measure", the other
        # would average into telemetry as a real reading of no memory used.
        assert rss_mb(pid=999_999_999) is None

    def test_reads_own_rss_on_linux(self):
        if sys.platform == "darwin":
            assert rss_mb() is None  # documented: no /proc, local-dev only
        else:
            value = rss_mb()
            assert value is not None and value > 0


class TestBatchResultCarriesMemory:
    """The aggregate must be a MAX, never a sum."""

    def _aggregate(self, worker_rss: list[int]) -> int | None:
        # Mirrors batch_ops: shared pages (mmapped bucket matrices, shared
        # training arrays) are counted in every process that maps them, so a sum
        # multiplies them by the worker count and is meaningless.
        collected = [r for r in worker_rss if isinstance(r, int)]
        return max(collected) if collected else None

    def test_uses_max_not_sum(self):
        assert self._aggregate([1000, 1200, 1100]) == 1200

    def test_none_when_no_worker_could_measure(self):
        assert self._aggregate([]) is None

    def test_ignores_unmeasurable_workers_without_poisoning_the_max(self):
        assert self._aggregate([1000, None, 1500]) == 1500  # type: ignore[list-item]


class TestProgressBarSuppression:
    """A redirected log must not receive progress-bar repaints."""

    def _disabled(self, *, verbose: bool, isatty: bool) -> bool:
        return not verbose or not isatty

    def test_disabled_when_stderr_is_redirected(self):
        # The cloud case: this is what made a leg's stderr too big to download.
        assert self._disabled(verbose=True, isatty=False) is True

    def test_enabled_on_an_interactive_terminal(self):
        assert self._disabled(verbose=True, isatty=True) is False

    def test_quiet_mode_still_wins_over_a_terminal(self):
        assert self._disabled(verbose=False, isatty=True) is True

    def test_partitioned_consults_stderr_isatty(self):
        """Pin the mechanism, not just the arithmetic."""
        from src.pipeline.training.trainer import partitioned

        source = partitioned.__file__
        assert source is not None
        text = pathlib.Path(source).read_text()
        assert "sys.stderr.isatty()" in text, (
            "progress-bar suppression must key on whether stderr is a terminal"
        )


class TestMetricsRowShape:
    def test_recorder_accepts_and_stores_the_memory_fields(self):
        from src.pipeline.training.trainer.batch_coordinator import TrainingBatchCoordinator

        rows: list[dict] = []
        coordinator = mock.Mock(spec=TrainingBatchCoordinator)
        coordinator.history_writer = SimpleNamespace(append=rows.append)
        coordinator.session = SimpleNamespace(
            metrics=SimpleNamespace(
                get_elapsed_time=lambda: 1.0,
                get_iterations_per_second=lambda: 100.0,
                get_avg_utility=lambda: 0.0,
                get_utility_std=lambda: 0.0,
                record_quality=lambda _q: None,
            )
        )
        coordinator._sample_quality = lambda: None
        coordinator._policy_delta = lambda _n: None

        TrainingBatchCoordinator._record_history(
            coordinator,
            1000,
            5000,
            capacity_pct=12.5,
            max_worker_rss_mb=1234,
            master_rss_mb=4185,
        )
        assert rows and rows[0]["max_worker_rss_mb"] == 1234
        assert rows[0]["master_rss_mb"] == 4185
