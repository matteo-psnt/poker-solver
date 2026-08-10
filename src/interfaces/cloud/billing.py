"""What Azure actually charged, as opposed to what node time implies.

:mod:`~src.interfaces.cloud.node_time` derives an estimate from the task log:
complete, granular, attributable to a run -- and structurally unable to be the
bill. Three reasons, all measured over 2026-07-26..2026-08-09:

* **A node is not only a task.** 351.3 VM-hours were billed against 263.0 hours
  of task execution -- allocation before a task starts and release after it
  ends, at 1.34x.
* **Compute is not the whole invoice.** $88.71 of $316.71 (28%) was storage,
  disks, load balancer and public IPs. The largest single non-compute line was
  $54.00 of LRS write operations -- 3.6M of them, the share being written a
  record at a time. No multiple of node-hours finds that.
* **One rate cannot span the history.** Billing shows three SKUs (D16als_v6,
  D8als_v6, and 0.2h of D8alds_v6), while ``hourly_cost`` is a single scalar
  for whatever ``vm_size`` is deployed *now*. The task log carries ``node_id``
  but no SKU, so the estimator cannot fix this even in principle.

So this asks the biller. Cost Management is the same source ``just
credit-check`` reads, which is what makes the two agree.

Additive, and independently failing
-----------------------------------
Every entry point here returns ``None`` rather than raising when Azure cannot be
reached. Node time is a property of the task log and must stay reportable on a
machine with no cloud credentials configured -- that promise predates this
module and outranks it. A cost screen that dies because the billing API is slow
is worse than one that says "billed spend unavailable" above numbers it still
has.

Freshness
---------
Cost data LAGS, by hours, and is restated. The most recent day is always
partial: 2026-08-08 read $7.22 against 21.00 VM-hours ($0.34/hr) while the
deployed SKU bills $0.688 -- not a cheap day, an incomplete one. ``as_of`` is
therefore reported and every surface is expected to show it, because a number
that silently means "up to some point yesterday" invites exactly the wrong
conclusion about a pool that is running right now.

Rate limits, and why the cache lives here
-----------------------------------------
Cost Management is metered far more tightly than the rest of ARM and answers
429 to a handful of queries in quick succession -- observed repeatedly while
developing this, and the penalty outlasts the burst by many minutes. The
console's own cache is 15s, which for this endpoint is both a way to get
throttled and pointless: the underlying data does not move for hours.

So the cache is here rather than at a caller. The constraint belongs to the API,
and putting it at the seam means every surface inherits it -- the CLI, the web
server, and any future poller -- instead of each rediscovering the 429.

It is TWO layers, because they solve different halves:

in-process
    A dict, for the long-lived server. Several endpoints and several open tabs
    share one sweep.
on disk
    Under the shared cache root, for the CLI -- which is a fresh process every
    time, so the memo above never hits and `poker-solver cost` run three times
    was three queries. That is exactly how the 429s above were earned. Cost data
    lags hours, so serving a 15-minute-old figure to a new process is not a
    compromise; re-asking would be.

Failures are cached too, briefly -- a throttled miss that immediately retries is
what turns one 429 into a sustained one -- but only in memory. A cross-process
failure cache would make one bad minute persist into the next command, and the
CLI is where someone is actively trying to fix things.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
from azure.identity import AzureCliCredential

from src.shared import cache

COST_MANAGEMENT_API = "2023-03-01"
ARM = "https://management.azure.com"
SCOPE = "https://management.azure.com/.default"

"""Which service lines are the pool doing its job.

Grouped rather than itemised because the question a cost screen answers is
"where did the money go", and 25 meters is not an answer. Everything not named
here is 'other', which is the bucket that was invisible entirely before.
"""
COMPUTE_SERVICES = frozenset({"Virtual Machines"})

"""How a pool node is told apart from a machine that is simply switched on.

Batch in UserSubscription mode creates one `azurebatch-<guid>-c` resource group
per pool and puts its nodes there, so the prefix identifies compute that a task
log can account for. Everything else billing VM hours is a STANDING machine --
here, `blueprint-server` in `poker-solver-serve-rg`.

The distinction is not cosmetic. Lumping them together produced a genuinely
wrong screen: 381.5 "billed node-hours" against 263.0 hours of task execution
looked like 1.45x of allocation overhead, and the caveat said so. 68 of those
hours were the serve box, which is not a pool node, is not allocated for a task,
and will never appear in the task log however long it runs. Against pool nodes
alone the ratio is 1.19x -- which is what allocation overhead actually costs.
"""
BATCH_RESOURCE_GROUP_PREFIX = "azurebatch"

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

"""How long an answer -- or the absence of one -- is reused.

The success TTL is generous because the data genuinely does not move faster
than that; a cost screen refreshed every 15s would show the same figures for
hours while spending a query each time. The failure TTL is short enough that
`az login` fixes the screen on the next refresh rather than at the end of a long
TTL, and long enough that a 429 is not answered with more traffic.
"""
CACHE_TTL_SECONDS = 900.0
FAILURE_TTL_SECONDS = 60.0


@dataclass(frozen=True)
class Billed:
    """Actual charges over a window, as Cost Management reports them."""

    total: float
    other: float
    currency: str

    """Pool nodes: the compute a task log can account for. ``pool_node_hours``
    is the figure that node time should be compared against, and nothing
    else."""
    pool_cost: float
    pool_node_hours: float

    """Machines that are simply ON, outside the pool -- the blueprint server.
    Reported separately because no amount of training explains them, and
    because an idle one bills 24 hours a day in total silence."""
    standing_cost: float
    standing_hours: float
    standing: list[tuple[str, float, float]]
    since: dt.date
    as_of: dt.date | None
    by_service: list[tuple[str, float]]

    """``since`` is what was ASKED for; ``first_at`` is the earliest day that
    actually carried a charge. For the all-history window the two differ by
    almost a year -- the query floor is a 364-day cap, while the subscription
    only started spending in 2026-07 -- and printing the floor would date the
    record to a period nothing happened in."""
    first_at: dt.date | None

    def as_payload(self) -> dict[str, Any]:
        """The wire shape. Dates as ISO strings, because a payload crosses to
        TypeScript and a ``date`` does not."""
        return {
            "total": self.total,
            "other": self.other,
            "currency": self.currency,
            "pool_cost": self.pool_cost,
            "pool_node_hours": self.pool_node_hours,
            "standing_cost": self.standing_cost,
            "standing_hours": self.standing_hours,
            "standing": [
                {"resource_group": name, "hours": hours, "cost": cost}
                for name, hours, cost in self.standing
            ],
            "since": self.since.isoformat(),
            "first_at": self.first_at.isoformat() if self.first_at else None,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "by_service": [{"service": name, "cost": cost} for name, cost in self.by_service],
        }


class ThrottledError(Exception):
    """Cost Management answered 429.

    Its own class because it is the failure that actually happens and the only
    one that is not a problem: the query was fine, it was asked too often, and
    the answer is to wait. Reporting it as "billing unavailable, check az login"
    -- which is what a single failure path does -- sends someone to fix an
    identity that was never broken.
    """

    def __init__(self, retry_after: float) -> None:
        super().__init__("Cost Management is rate-limiting; the figures are unchanged")
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> float:
    """How long Azure asked us to wait, clamped to something a screen can live
    with. Absent or unparseable falls back to the standard failure TTL."""
    raw = response.headers.get("Retry-After", "")
    try:
        return max(FAILURE_TTL_SECONDS, min(float(raw), CACHE_TTL_SECONDS))
    except (TypeError, ValueError):
        return FAILURE_TTL_SECONDS


def _query(subscription_id: str, since: dt.date, until: dt.date) -> dict[str, Any]:
    """One Cost Management query, daily and grouped by service.

    Daily granularity rather than ``None`` for one reason: it is the only way to
    learn how far the data actually reaches. A totals-only response cannot
    distinguish "complete through today" from "complete through Tuesday", and
    the difference is the whole freshness caveat.
    """
    token = AzureCliCredential().get_token(SCOPE).token
    url = (
        f"{ARM}/subscriptions/{subscription_id}"
        f"/providers/Microsoft.CostManagement/query?api-version={COST_MANAGEMENT_API}"
    )
    body = {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {
            "from": f"{since:%Y-%m-%d}T00:00:00Z",
            "to": f"{until:%Y-%m-%d}T23:59:59Z",
        },
        "dataset": {
            "granularity": "Daily",
            "aggregation": {
                "totalCost": {"name": "Cost", "function": "Sum"},
                "qty": {"name": "UsageQuantity", "function": "Sum"},
            },
            "grouping": [
                {"type": "Dimension", "name": "ServiceName"},
                # The resource group is what separates a pool node from a
                # standing machine; see BATCH_RESOURCE_GROUP_PREFIX.
                {"type": "Dimension", "name": "ResourceGroupName"},
            ],
        },
    }
    response = httpx.post(
        url, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT
    )
    if response.status_code == 429:
        raise ThrottledError(_retry_after(response))
    response.raise_for_status()
    return response.json()


def _usage_date(raw: Any) -> dt.date | None:
    """Cost Management returns ``UsageDate`` as the integer 20260808.

    Built as a date rather than parsed as a datetime: there is no time and no
    zone in the value, so producing a naive datetime and taking ``.date()`` off
    it only invites someone to compare it against an aware one later.
    """
    try:
        digits = str(int(raw))
        return dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except (TypeError, ValueError):
        return None


"""Why there is no figure, in the words a surface should show.

A single "unavailable" covers two situations that call for opposite responses:
throttling means wait and the numbers are fine, everything else means something
is actually broken. Telling someone to check `az login` when Cost Management is
merely busy sends them to fix an identity that was never wrong.
"""
UNAVAILABLE = "Billing unavailable — check `az login` and that Terraform state is readable."

_LOCK = threading.Lock()
_MEMO: dict[tuple[str, str, str], tuple[float, Billed | None, str | None]] = {}


def summarise(subscription_id: str, *, since: dt.date, until: dt.date) -> Billed | None:
    """Billed spend over the window, or ``None`` if Azure could not be asked."""
    return summarise_with_reason(subscription_id, since=since, until=until)[0]


def summarise_with_reason(
    subscription_id: str, *, since: dt.date, until: dt.date
) -> tuple[Billed | None, str | None]:
    """Billed spend, and -- when there is none -- why, in one sentence.

    Memoised, including failures; see the header on rate limits. The query runs
    OUTSIDE the lock: holding it across a network call would serialise every
    caller behind the slowest one, and two concurrent misses costing two queries
    is a far smaller problem than a blocked console.
    """
    key = (subscription_id, since.isoformat(), until.isoformat())
    with _LOCK:
        entry = _MEMO.get(key)
        if entry is not None and time.monotonic() < entry[0]:
            return entry[1], entry[2]

    stored = _read_disk(key)
    if stored is not None:
        with _LOCK:
            _MEMO[key] = (time.monotonic() + CACHE_TTL_SECONDS, stored, None)
        return stored, None

    result, reason, ttl = _summarise_uncached(subscription_id, since=since, until=until)

    with _LOCK:
        _MEMO[key] = (time.monotonic() + ttl, result, reason)
    if result is not None:
        _write_disk(key, result)
    return result, reason


CACHE_NAME = "billing"


def _read_disk(key: tuple[str, str, str]) -> Billed | None:
    """A recent answer from a previous process, rebuilt from its payload."""
    payload = cache.cached_json(CACHE_NAME, "|".join(key), CACHE_TTL_SECONDS)
    if payload is None:
        return None
    try:
        return Billed(
            total=float(payload["total"]),
            pool_cost=float(payload["pool_cost"]),
            pool_node_hours=float(payload["pool_node_hours"]),
            standing_cost=float(payload["standing_cost"]),
            standing_hours=float(payload["standing_hours"]),
            standing=[
                (row["resource_group"], float(row["hours"]), float(row["cost"]))
                for row in payload["standing"]
            ],
            other=float(payload["other"]),
            currency=str(payload["currency"]),
            since=dt.date.fromisoformat(payload["since"]),
            first_at=dt.date.fromisoformat(payload["first_at"]) if payload["first_at"] else None,
            as_of=dt.date.fromisoformat(payload["as_of"]) if payload["as_of"] else None,
            by_service=[(row["service"], float(row["cost"])) for row in payload["by_service"]],
        )
    except (KeyError, TypeError, ValueError):
        # A cache written by an older shape. A miss, like any other.
        return None


def _write_disk(key: tuple[str, str, str], result: Billed) -> None:
    """Store an answer for the next process."""
    cache.store_json(CACHE_NAME, "|".join(key), result.as_payload())


def _summarise_uncached(
    subscription_id: str, *, since: dt.date, until: dt.date
) -> tuple[Billed | None, str | None, float]:
    """The uncached read: the figures, why not, and how long to remember it.

    The broad except is the point of the module: see the header. Only throttling
    is singled out, because it is the one failure that is not a fault.
    """
    try:
        payload = _query(subscription_id, since, until)
    except ThrottledError as throttled:
        return None, str(throttled), throttled.retry_after
    except Exception:  # noqa: BLE001 -- billing is additive; unavailable reads as unknown
        return None, UNAVAILABLE, FAILURE_TTL_SECONDS

    try:
        properties = payload["properties"]
        columns = [column["name"] for column in properties["columns"]]
        index = {name: position for position, name in enumerate(columns)}
        for needed in ("Cost", "UsageQuantity", "UsageDate", "ServiceName", "ResourceGroupName"):
            if needed not in index:
                return None, UNAVAILABLE, FAILURE_TTL_SECONDS

        total = pool_cost = pool_hours = standing_cost = standing_hours = 0.0
        per_standing: dict[str, tuple[float, float]] = {}
        per_service: dict[str, float] = {}
        earliest: dt.date | None = None
        latest: dt.date | None = None
        currency = "USD"

        for row in properties["rows"]:
            cost = float(row[index["Cost"]])
            quantity = float(row[index["UsageQuantity"]])
            service = str(row[index["ServiceName"]] or "other")
            if "Currency" in index:
                currency = str(row[index["Currency"]] or currency)

            total += cost
            per_service[service] = per_service.get(service, 0.0) + cost
            if service in COMPUTE_SERVICES:
                group = str(row[index["ResourceGroupName"]] or "")
                if group.startswith(BATCH_RESOURCE_GROUP_PREFIX):
                    pool_cost += cost
                    pool_hours += quantity
                else:
                    standing_cost += cost
                    standing_hours += quantity
                    was = per_standing.get(group, (0.0, 0.0))
                    per_standing[group] = (was[0] + quantity, was[1] + cost)

            day = _usage_date(row[index["UsageDate"]])
            # Only days that actually carry a charge date the data. A zero-cost
            # row for today would otherwise claim coverage the biller has not
            # given, which is the freshness lie this exists to avoid.
            if day is not None and cost:
                latest = day if latest is None else max(latest, day)
                earliest = day if earliest is None else min(earliest, day)
    except (KeyError, IndexError, TypeError, ValueError):
        # A shape change is not a zero bill. Same contract as an unreachable
        # API: unknown, so the surface says so rather than reporting $0.00.
        return None, UNAVAILABLE, FAILURE_TTL_SECONDS

    return (
        Billed(
            total=total,
            other=total - pool_cost - standing_cost,
            pool_cost=pool_cost,
            pool_node_hours=pool_hours,
            standing_cost=standing_cost,
            standing_hours=standing_hours,
            standing=sorted(
                ((name, hours, cost) for name, (hours, cost) in per_standing.items()),
                key=lambda row: -row[2],
            ),
            currency=currency,
            since=since,
            first_at=earliest,
            as_of=latest,
            by_service=sorted(per_service.items(), key=lambda item: -item[1]),
        ),
        None,
        CACHE_TTL_SECONDS,
    )
