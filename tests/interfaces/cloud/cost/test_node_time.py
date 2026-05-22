"""Node time from task intervals.

This replaced a sampler that only recorded while the console's server ran, and
so covered ~3% of any window. These numbers are derived from the task log, which
is complete by construction — which makes the arithmetic worth pinning, because
nothing else will contradict it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.interfaces.cloud.cost import node_time
from src.shared import task_history
from src.shared.cloudtask import task_log

NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)


def _task(
    start_min: float, end_min: float | None, cause: str = task_log.CAUSE_COMPLETED
) -> dict[str, object]:
    """One task row. An open row needs a `cause`, because that is the only thing
    that says whether the clock is still running -- see `_live`."""
    base = NOW - timedelta(hours=6)
    row: dict[str, object] = {
        "started_at": (base + timedelta(minutes=start_min)).isoformat(),
        "cause": cause,
    }
    if end_min is not None:
        row["ended_at"] = (base + timedelta(minutes=end_min)).isoformat()
    return row


def _live(start_min: float) -> dict[str, object]:
    """A task Batch says is running right now."""
    return _task(start_min, None, cause=task_history.CAUSE_RUNNING)


class TestNodeTime:
    def test_one_task_is_its_own_duration(self):
        result = node_time.summarise([_task(0, 60)], now=NOW)
        assert result["task_hours"] == pytest.approx(1.0)
        assert result["peak_concurrency"] == 1

    def test_concurrent_tasks_add_up(self):
        """Four tasks for an hour each, all at once, is four node-hours — which
        is the whole point of integrating concurrency rather than wall clock."""
        result = node_time.summarise([_task(0, 60) for _ in range(4)], now=NOW)
        assert result["task_hours"] == pytest.approx(4.0)
        assert result["peak_concurrency"] == 4

    def test_sequential_tasks_do_not_stack(self):
        result = node_time.summarise([_task(0, 60), _task(60, 120)], now=NOW)
        assert result["task_hours"] == pytest.approx(2.0)
        assert result["peak_concurrency"] == 1

    def test_a_live_task_is_credited_up_to_now(self):
        """Dropping in-flight work would make the total DIP exactly when the
        pool is busiest, which is when someone is most likely to look."""
        result = node_time.summarise([_live(300)], now=NOW)
        assert result["task_hours"] == pytest.approx(1.0)
        assert result["unended"] == 0

    def test_preparing_counts_too(self):
        """The node is committed and billing before the task is running on it."""
        result = node_time.summarise(
            [_task(300, None, cause=task_history.CAUSE_PREPARING)], now=NOW
        )
        assert result["task_hours"] == pytest.approx(1.0)

    def test_an_open_task_with_no_live_cause_is_excluded_and_counted(self):
        """THE BUG THIS MODULE SHIPPED WITH. `unresolved` means the node wrote a
        start, never wrote an end, and Batch could not explain it either -- it
        covers a running task AND one that died silently, and the record cannot
        tell them apart. Running it to `now` invented 455 of the 718 node-hours
        the cost screen reported, from four attempts abandoned days earlier, and
        the total grew by four hours per elapsed hour while the pool sat idle.

        Excluded rather than estimated, and counted so the exclusion is visible:
        their node time is unknown, which is not the same as zero."""
        result = node_time.summarise([_task(300, None, cause="unresolved")], now=NOW)
        assert result["task_hours"] == 0.0
        assert result["tasks"] == 0
        assert result["unended"] == 1

    def test_an_open_task_does_not_grow_with_the_clock(self):
        """The property that made the old number absurd: read it an hour later
        and it was an hour bigger, with nothing having run."""
        row = _task(300, None, cause="unresolved")
        later = node_time.summarise([row], now=NOW + timedelta(days=7))
        assert later["task_hours"] == 0.0

    def test_a_closed_task_ignores_its_cause(self):
        """A failed task still burned the node it failed on."""
        result = node_time.summarise([_task(0, 60, cause=task_log.CAUSE_FAILED)], now=NOW)
        assert result["task_hours"] == pytest.approx(1.0)

    def test_a_task_with_no_start_is_skipped_not_fatal(self):
        result = node_time.summarise([{"ended_at": NOW.isoformat()}, _task(0, 60)], now=NOW)
        assert result["tasks"] == 1

    def test_a_task_with_no_start_is_not_counted_as_unended(self):
        """`unended` means "ran, and we do not know for how long". A task that
        never started has no missing interval to report."""
        result = node_time.summarise([{"ended_at": NOW.isoformat()}], now=NOW)
        assert result["unended"] == 0

    def test_a_malformed_timestamp_is_skipped(self):
        result = node_time.summarise([{"started_at": "not-a-date"}, _task(0, 60)], now=NOW)
        assert result["tasks"] == 1

    def test_no_tasks_is_zero_not_an_error(self):
        result = node_time.summarise([], now=NOW)
        assert result["task_hours"] == 0.0
        assert result["peak_concurrency"] == 0
        assert result["series"] == []


class TestWindowing:
    def test_a_window_clips_rather_than_excludes(self):
        """A task that started before the window but ran into it contributes the
        part inside it. Excluding it would erase work that was genuinely running,
        and counting it whole would attribute time from outside the window."""
        result = node_time.summarise([_task(0, 360)], now=NOW, since=NOW - timedelta(hours=1))
        assert result["task_hours"] == pytest.approx(1.0)

    def test_a_task_entirely_before_the_window_is_dropped(self):
        result = node_time.summarise([_task(0, 60)], now=NOW, since=NOW - timedelta(hours=1))
        assert result["tasks"] == 0


class TestTimeline:
    def test_it_tracks_concurrency_up_and_down(self):
        spans, _ = node_time.intervals([_task(0, 60), _task(30, 90)], now=NOW)
        counts = [count for _, count in node_time.timeline(spans)]
        assert counts == [1, 2, 1, 0]

    def test_simultaneous_events_collapse_to_one_point(self):
        """Three tasks starting in the same second is ONE change in concurrency;
        emitting three points would draw a staircase that never happened."""
        spans, _ = node_time.intervals([_task(0, 60) for _ in range(3)], now=NOW)
        series = node_time.timeline(spans)
        assert [count for _, count in series] == [3, 0]
