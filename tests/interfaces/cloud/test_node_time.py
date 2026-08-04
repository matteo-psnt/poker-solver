"""Node time from leg intervals.

This replaced a sampler that only recorded while the console's server ran, and
so covered ~3% of any window. These numbers are derived from the leg log, which
is complete by construction — which makes the arithmetic worth pinning, because
nothing else will contradict it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.interfaces.cloud import node_time

NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)


def _leg(start_min: float, end_min: float | None) -> dict[str, object]:
    base = NOW - timedelta(hours=6)
    row: dict[str, object] = {"started_at": (base + timedelta(minutes=start_min)).isoformat()}
    if end_min is not None:
        row["ended_at"] = (base + timedelta(minutes=end_min)).isoformat()
    return row


class TestNodeTime:
    def test_one_leg_is_its_own_duration(self):
        result = node_time.summarise([_leg(0, 60)], now=NOW)
        assert result["task_hours"] == pytest.approx(1.0)
        assert result["peak_concurrency"] == 1

    def test_concurrent_legs_add_up(self):
        """Four legs for an hour each, all at once, is four node-hours — which
        is the whole point of integrating concurrency rather than wall clock."""
        result = node_time.summarise([_leg(0, 60) for _ in range(4)], now=NOW)
        assert result["task_hours"] == pytest.approx(4.0)
        assert result["peak_concurrency"] == 4

    def test_sequential_legs_do_not_stack(self):
        result = node_time.summarise([_leg(0, 60), _leg(60, 120)], now=NOW)
        assert result["task_hours"] == pytest.approx(2.0)
        assert result["peak_concurrency"] == 1

    def test_an_unfinished_leg_is_credited_up_to_now(self):
        """Dropping in-flight work would make the total DIP exactly when the
        pool is busiest, which is when someone is most likely to look."""
        result = node_time.summarise([_leg(300, None)], now=NOW)
        assert result["task_hours"] == pytest.approx(1.0)

    def test_a_leg_with_no_start_is_skipped_not_fatal(self):
        result = node_time.summarise([{"ended_at": NOW.isoformat()}, _leg(0, 60)], now=NOW)
        assert result["legs"] == 1

    def test_a_malformed_timestamp_is_skipped(self):
        result = node_time.summarise([{"started_at": "not-a-date"}, _leg(0, 60)], now=NOW)
        assert result["legs"] == 1

    def test_no_legs_is_zero_not_an_error(self):
        result = node_time.summarise([], now=NOW)
        assert result["task_hours"] == 0.0
        assert result["peak_concurrency"] == 0
        assert result["series"] == []


class TestWindowing:
    def test_a_window_clips_rather_than_excludes(self):
        """A leg that started before the window but ran into it contributes the
        part inside it. Excluding it would erase work that was genuinely running,
        and counting it whole would attribute time from outside the window."""
        result = node_time.summarise([_leg(0, 360)], now=NOW, since=NOW - timedelta(hours=1))
        assert result["task_hours"] == pytest.approx(1.0)

    def test_a_leg_entirely_before_the_window_is_dropped(self):
        result = node_time.summarise([_leg(0, 60)], now=NOW, since=NOW - timedelta(hours=1))
        assert result["legs"] == 0


class TestTimeline:
    def test_it_tracks_concurrency_up_and_down(self):
        spans = node_time.intervals([_leg(0, 60), _leg(30, 90)], now=NOW)
        counts = [count for _, count in node_time.timeline(spans)]
        assert counts == [1, 2, 1, 0]

    def test_simultaneous_events_collapse_to_one_point(self):
        """Three legs starting in the same second is ONE change in concurrency;
        emitting three points would draw a staircase that never happened."""
        spans = node_time.intervals([_leg(0, 60) for _ in range(3)], now=NOW)
        series = node_time.timeline(spans)
        assert [count for _, count in series] == [3, 0]
