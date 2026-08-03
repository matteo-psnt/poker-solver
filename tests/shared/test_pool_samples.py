"""Node-hours, and the gap that must not be billed.

The series cannot be reconstructed — Batch keeps no node history — so a number
derived from it wrongly cannot be checked against anything later. The arithmetic
is therefore pinned against hand-computed values rather than against itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.shared import pool_samples


def _at(offset_seconds: float) -> str:
    base = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=offset_seconds)).isoformat()


def _rows(*pairs: tuple[float, int]) -> list[dict[str, object]]:
    return [{"at": _at(offset), "nodes": nodes, "vm_size": "d16"} for offset, nodes in pairs]


class TestIntegrate:
    def test_a_flat_series_is_nodes_times_elapsed(self):
        """4 nodes held across two 30s intervals = 4 * 60s = 1/15 node-hour."""
        result = pool_samples.integrate(_rows((0, 4), (30, 4), (60, 4)))
        assert result["node_hours"] == pytest.approx(4 * 60 / 3600)

    def test_each_sample_is_credited_until_the_next_one(self):
        """The interval after a sample carries that sample's count, which is
        what 'the pool held N nodes' means for a series of observations."""
        # 4 nodes for 60s, then 0 for 60s -> only the first interval counts.
        result = pool_samples.integrate(_rows((0, 4), (60, 0), (120, 0)), max_gap=120)
        assert result["node_hours"] == pytest.approx(4 * 60 / 3600)

    def test_an_idle_pool_costs_nothing(self):
        assert pool_samples.integrate(_rows((0, 0), (30, 0)))["node_hours"] == 0.0

    def test_a_gap_is_excluded_and_reported(self):
        """THE correctness case. If the server was off for six hours, the last
        sample before the gap must not bill six hours at whatever was running.
        A total that silently swallowed it would read as a complete accounting."""
        rows = _rows((0, 4), (30, 4), (30 + 6 * 3600, 4), (30 + 6 * 3600 + 30, 4))
        result = pool_samples.integrate(rows)

        assert result["node_hours"] == pytest.approx(2 * 4 * 30 / 3600)
        assert result["unobserved_seconds"] == pytest.approx(6 * 3600)

    def test_out_of_order_samples_are_sorted_not_negative(self):
        """Appends are ordered in practice; arithmetic that assumes it would go
        silently wrong rather than fail if they ever were not."""
        ordered = pool_samples.integrate(_rows((0, 4), (30, 4)))
        shuffled = pool_samples.integrate(_rows((30, 4), (0, 4)))
        assert shuffled["node_hours"] == pytest.approx(ordered["node_hours"])

    def test_an_empty_or_single_series_yields_zero_not_an_error(self):
        """One observation bounds no interval, so it is zero — never a crash on
        the page that would explain why nothing has been recorded."""
        for rows in ([], _rows((0, 4))):
            result = pool_samples.integrate(rows)
            assert result["node_hours"] == 0.0


class TestReadingIsForgiving:
    def test_a_half_written_line_is_skipped_not_fatal(self, tmp_path):
        """The expected residue of a process killed mid-append."""
        path = tmp_path / "s.jsonl"
        path.write_text('{"at": "2026-08-03T12:00:00+00:00", "nodes": 4}\n{"at": "2026-')
        assert len(pool_samples.read(path)) == 1

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert pool_samples.read(tmp_path / "never-written.jsonl") == []

    def test_append_then_read_round_trips(self, tmp_path):
        path = tmp_path / "s.jsonl"
        pool_samples.append(4, "d16", path=path)
        pool_samples.append(0, "d16", path=path)
        rows = pool_samples.read(path)
        assert [row["nodes"] for row in rows] == [4, 0]


class TestPrune:
    def test_it_drops_only_what_is_past_retention(self, tmp_path):
        path = tmp_path / "s.jsonl"
        now = datetime.now(UTC)
        for age_days, nodes in ((40, 1), (31, 2), (1, 3)):
            pool_samples.append(
                nodes, "d16", path=path, at=(now - timedelta(days=age_days)).isoformat()
            )

        removed = pool_samples.prune(path)

        assert removed == 2
        assert [row["nodes"] for row in pool_samples.read(path)] == [3]

    def test_it_leaves_the_file_alone_when_nothing_expired(self, tmp_path):
        path = tmp_path / "s.jsonl"
        pool_samples.append(4, "d16", path=path)
        before = path.read_text()
        assert pool_samples.prune(path) == 0
        assert path.read_text() == before


class TestHourlyRate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("$0.80/hr/node", 0.80), ("0.8", 0.8), ("$1.2345/hr", 1.2345)],
    )
    def test_it_reads_terraforms_human_string(self, raw, expected):
        assert pool_samples.hourly_rate(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "unknown"])
    def test_an_unreadable_rate_is_none_not_zero(self, raw):
        """Zero would render as a confident '$0.00'. A wrong dollar figure is
        worse than none, so the page shows node-hours alone instead."""
        assert pool_samples.hourly_rate(raw) is None
