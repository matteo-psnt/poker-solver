"""Billed spend, parsed out of Cost Management's row-and-column shape.

The point of this module is that it is the AUTHORITY -- the number a human
compares against `just credit-check`. So the parsing is pinned, and so is the
contract that matters more than the parsing: every failure reads as "unknown",
never as "$0.00". A cost screen that says zero because a token expired is worse
than one that says it does not know.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import httpx
import pytest

from src.interfaces.cloud import billing

SINCE = dt.date(2026, 7, 26)
UNTIL = dt.date(2026, 8, 9)


def _response(rows: list[list[object]]) -> dict[str, object]:
    """Cost Management's shape: named columns, positional rows."""
    return {
        "properties": {
            "columns": [
                {"name": "Cost"},
                {"name": "UsageQuantity"},
                {"name": "UsageDate"},
                {"name": "ServiceName"},
                {"name": "ResourceGroupName"},
                {"name": "Currency"},
            ],
            "rows": rows,
        }
    }


"""Two days of the real bill, reduced. Numbers taken from the 2026-08 audit so
the arithmetic below is checkable against something that happened."""
REAL_ROWS: list[list[object]] = [
    [100.00, 145.35, 20260803, "Virtual Machines", "azurebatch-abc-c", "USD"],
    [114.12, 165.87, 20260806, "Virtual Machines", "azurebatch-def-c", "USD"],
    # The blueprint server: a VM that is simply ON, in its own resource group.
    [23.33, 67.83, 20260806, "Virtual Machines", "poker-solver-serve-rg", "USD"],
    [54.00, 3600.24, 20260806, "Storage", "poker-solver-store-rg", "USD"],
    [0.00, 0.00, 20260809, "Storage", "poker-solver-store-rg", "USD"],
]


@pytest.fixture
def stub(monkeypatch):
    """Replace the network call, keeping the parsing under test."""

    def install(result):
        calls: list[int] = []

        def fake(subscription_id, since, until):
            calls.append(1)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(billing, "_query", fake)
        return calls

    return install


class TestParsing:
    def test_it_splits_compute_from_everything_else(self, stub):
        """The split IS the finding: 28% of the real bill was not compute, and a
        node-hours estimate cannot see any of it."""
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert result.total == pytest.approx(291.45)
        assert result.other == pytest.approx(54.00)

    def test_a_standing_vm_is_not_counted_as_pool_compute(self, stub):
        """THE SECOND WRONG NUMBER. `blueprint-server` sits in its own resource
        group billing 24 hours a day whether or not anything trains. Folded into
        compute it made 313.5 pool node-hours read as 381.5, so node time looked
        like 1.45x of allocation overhead when the real figure is 1.19x -- and
        the caveat text confidently blamed overhead for a machine nobody had
        switched off."""
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert result.pool_node_hours == pytest.approx(311.22)
        assert result.pool_cost == pytest.approx(214.12)
        assert result.standing_hours == pytest.approx(67.83)
        assert result.standing_cost == pytest.approx(23.33)

    def test_standing_machines_are_named(self, stub):
        """An unattributed number invites the reader to assume it is training."""
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert [name for name, _, _ in result.standing] == ["poker-solver-serve-rg"]

    def test_node_hours_come_only_from_compute(self, stub):
        """3600 storage operations are a UsageQuantity too. Summing the column
        blind would report 3,911 node-hours."""
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert result.pool_node_hours == pytest.approx(311.22)

    def test_services_are_ranked_by_spend(self, stub):
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert [name for name, _ in result.by_service] == ["Virtual Machines", "Storage"]

    def test_it_dates_the_data_by_charged_days_only(self, stub):
        """A zero-cost row for today would claim coverage the biller has not
        given. `as_of` is the freshness caveat -- cost data lags hours and the
        most recent day always reads low -- so it must not be inflated by a row
        that carries no money."""
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert result.as_of == dt.date(2026, 8, 6)
        assert result.first_at == dt.date(2026, 8, 3)

    def test_first_at_is_not_the_query_floor(self, stub):
        """`since` is asked for; `first_at` is what was charged. For the
        all-history window they differ by almost a year."""
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=dt.date(2025, 8, 10), until=UNTIL)
        assert result is not None
        assert result.since == dt.date(2025, 8, 10)
        assert result.first_at == dt.date(2026, 8, 3)

    def test_an_empty_window_is_a_real_zero(self, stub):
        """Distinct from unavailable: the biller answered and there were no
        charges."""
        stub(_response([]))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert result.total == 0.0
        assert result.as_of is None


class TestUnavailableIsNotZero:
    @pytest.mark.parametrize(
        "failure",
        [
            httpx.ConnectError("no route"),
            httpx.ReadTimeout("slow"),
            RuntimeError("az login required"),
        ],
        ids=["unreachable", "timeout", "no-credential"],
    )
    def test_a_failed_read_is_none(self, stub, failure):
        """The exception TYPE is not the point -- `_summarise_uncached` catches
        anything on purpose. What is pinned is that none of them reach a caller
        as a traceback or as a zero."""
        stub(failure)
        assert billing.summarise("sub", since=SINCE, until=UNTIL) is None

    def test_a_throttled_read_is_none(self, stub):
        stub(billing.ThrottledError(retry_after=30.0))
        assert billing.summarise("sub", since=SINCE, until=UNTIL) is None


class TestThrottlingIsNotAnAuthProblem:
    """429 is the failure that ACTUALLY happens -- Cost Management is metered
    far more tightly than the rest of ARM, and three probes inside a minute
    tripped it while this was being written. It is also the only failure that is
    not a fault: the query was fine and the figures are unchanged.

    Reporting it as "check `az login`" sends someone to fix an identity that was
    never broken, which is the same class of error as the $0.80 rate this change
    removed -- a confident explanation that happens to be wrong.
    """

    def test_it_is_reported_as_waiting_not_as_broken(self, stub):
        stub(billing.ThrottledError(retry_after=30.0))
        _, reason = billing.summarise_with_reason("sub", since=SINCE, until=UNTIL)
        assert reason is not None
        assert "rate-limiting" in reason
        assert "az login" not in reason

    def test_any_other_failure_still_points_at_the_usual_suspects(self, stub):
        stub(httpx.ConnectError("no route"))
        _, reason = billing.summarise_with_reason("sub", since=SINCE, until=UNTIL)
        assert reason == billing.UNAVAILABLE

    def test_a_successful_read_has_no_reason(self, stub):
        stub(_response(REAL_ROWS))
        result, reason = billing.summarise_with_reason("sub", since=SINCE, until=UNTIL)
        assert result is not None
        assert reason is None

    def test_a_429_response_becomes_throttled_not_a_status_error(self, monkeypatch):
        """Pinned at the HTTP boundary, because the whole distinction is lost if
        429 falls into the generic `raise_for_status` path."""
        request = httpx.Request("POST", "https://management.azure.com/")

        monkeypatch.setattr(billing, "AzureCliCredential", lambda: _FakeCredential())
        monkeypatch.setattr(
            billing.httpx,
            "post",
            lambda *a, **k: httpx.Response(429, request=request, headers={"Retry-After": "300"}),
        )

        with pytest.raises(billing.ThrottledError) as raised:
            billing._query("sub", SINCE, UNTIL)
        assert raised.value.retry_after == 300.0

    def test_the_backoff_is_clamped_at_both_ends(self):
        """Azure's number is honoured BETWEEN the bounds and ignored outside
        them. The floor matters as much as the ceiling: retrying sooner than the
        ordinary failure TTL is how one 429 becomes a sustained one."""
        request = httpx.Request("POST", "https://management.azure.com/")

        def wait(header: str | None) -> float:
            headers = {"Retry-After": header} if header is not None else {}
            return billing._retry_after(httpx.Response(429, request=request, headers=headers))

        assert wait("300") == 300.0
        assert wait("5") == billing.FAILURE_TTL_SECONDS
        assert wait("99999") == billing.CACHE_TTL_SECONDS
        assert wait(None) == billing.FAILURE_TTL_SECONDS
        assert wait("soon") == billing.FAILURE_TTL_SECONDS


class _FakeCredential:
    def get_token(self, scope):
        """Signature mirrors the SDK's; the scope is not used."""
        del scope
        return type("Token", (), {"token": "fake"})()

    def test_a_changed_response_shape_is_none_not_zero(self, stub):
        """Silently reporting $0.00 because a column was renamed is the failure
        mode this whole module exists to avoid."""
        stub({"properties": {"columns": [{"name": "Cost"}], "rows": [[1.0]]}})
        assert billing.summarise("sub", since=SINCE, until=UNTIL) is None

    def test_a_garbage_payload_is_none_not_a_traceback(self, stub):
        stub({"nothing": "useful"})
        assert billing.summarise("sub", since=SINCE, until=UNTIL) is None


class TestDiskCache:
    """The in-process memo is useless to the CLI, which is a fresh process every
    time -- `poker-solver cost` run three times was three queries, which is
    exactly how the 429s that prompted this were earned. Cost data lags hours,
    so serving a 15-minute-old figure to a new process is not a compromise.
    """

    def test_an_answer_survives_the_process_that_fetched_it(self, stub):
        calls = stub(_response(REAL_ROWS))
        first = billing.summarise("sub", since=SINCE, until=UNTIL)

        billing._MEMO.clear()  # a new process has no memo
        second = billing.summarise("sub", since=SINCE, until=UNTIL)

        assert len(calls) == 1
        assert first is not None
        assert second is not None
        assert second.total == first.total
        assert second.as_of == first.as_of

    def test_a_stale_file_is_a_miss_not_a_stale_answer(self, stub, tmp_path):
        """Backdated in the file rather than by patching the clock: `time` is a
        shared module, so patching `billing.time.time` patches the test's own
        `time` too and the lambda calls itself."""
        calls = stub(_response(REAL_ROWS))
        billing.summarise("sub", since=SINCE, until=UNTIL)
        billing._MEMO.clear()

        for path in (tmp_path / "billing").iterdir():
            stored = json.loads(path.read_text())
            stored["at"] = time.time() - billing.CACHE_TTL_SECONDS - 1
            path.write_text(json.dumps(stored))

        billing.summarise("sub", since=SINCE, until=UNTIL)

        assert len(calls) == 2

    def test_a_failure_is_not_written_to_disk(self, stub):
        """A cross-process failure cache would make one bad minute persist into
        the next command, and the CLI is where someone is actively fixing it."""
        stub(httpx.ConnectError("no route"))
        billing.summarise("sub", since=SINCE, until=UNTIL)
        billing._MEMO.clear()

        calls = stub(_response(REAL_ROWS))
        assert billing.summarise("sub", since=SINCE, until=UNTIL) is not None
        assert len(calls) == 1

    def test_a_corrupt_cache_file_is_a_miss_not_a_crash(self, stub, tmp_path):
        calls = stub(_response(REAL_ROWS))
        billing.summarise("sub", since=SINCE, until=UNTIL)
        billing._MEMO.clear()

        for path in (tmp_path / "billing").iterdir():
            path.write_text("{not json")

        assert billing.summarise("sub", since=SINCE, until=UNTIL) is not None
        assert len(calls) == 2

    def test_an_unwritable_cache_root_still_returns_the_figures(self, stub, monkeypatch):
        """Best-effort in both directions: a cache must not break the thing it
        accelerates."""
        stub(_response(REAL_ROWS))
        monkeypatch.setattr(billing.cache, "cache_root", lambda: Path("/nonexistent-root/x"))
        assert billing.summarise("sub", since=SINCE, until=UNTIL) is not None


class TestMemo:
    def test_a_second_read_does_not_hit_the_api(self, stub):
        """Cost Management answers 429 to a handful of queries in quick
        succession, and the console polls every 60s against data that moves
        hourly."""
        calls = stub(_response(REAL_ROWS))
        billing.summarise("sub", since=SINCE, until=UNTIL)
        billing.summarise("sub", since=SINCE, until=UNTIL)
        assert len(calls) == 1

    def test_a_different_window_is_a_different_answer(self, stub):
        calls = stub(_response(REAL_ROWS))
        billing.summarise("sub", since=SINCE, until=UNTIL)
        billing.summarise("sub", since=SINCE, until=dt.date(2026, 8, 8))
        assert len(calls) == 2

    def test_a_failure_is_memoised_briefly_not_for_the_full_ttl(self, stub):
        """Long enough that a 429 is not answered with more traffic, short
        enough that `az login` fixes the screen on the next refresh."""
        assert billing.FAILURE_TTL_SECONDS < billing.CACHE_TTL_SECONDS
        calls = stub(httpx.ConnectError("no route"))
        billing.summarise("sub", since=SINCE, until=UNTIL)
        billing.summarise("sub", since=SINCE, until=UNTIL)
        assert len(calls) == 1

    def test_a_throttled_read_backs_off_for_as_long_as_azure_asked(self, stub):
        """Answering a 429 with more traffic is how one becomes sustained."""
        calls = stub(billing.ThrottledError(retry_after=300.0))
        billing.summarise("sub", since=SINCE, until=UNTIL)
        billing.summarise("sub", since=SINCE, until=UNTIL)
        assert len(calls) == 1
        assert billing._MEMO[("sub", SINCE.isoformat(), UNTIL.isoformat())][2] is not None


class TestPayload:
    def test_dates_cross_the_wire_as_strings(self, stub):
        """The payload is consumed by TypeScript, which has no `date`."""
        stub(_response(REAL_ROWS))
        result = billing.summarise("sub", since=SINCE, until=UNTIL)
        assert result is not None
        payload = result.as_payload()
        assert payload["since"] == "2026-07-26"
        assert payload["as_of"] == "2026-08-06"
        assert payload["by_service"][0] == {"service": "Virtual Machines", "cost": 237.45}
        assert payload["standing"][0]["resource_group"] == "poker-solver-serve-rg"
