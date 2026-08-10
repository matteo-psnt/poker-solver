"""The `cost` subcommand: what was billed, and what node time was spent.

Two numbers, from two sources, on purpose.

**Billed** comes from Cost Management -- the actual invoice line, the same
source `just credit-check` reads. It is the answer to "what has this cost".

**Node time** comes from the task log -- every attempt, with its run, its op and
its exit cause. It is the answer to "where did it go", and it is the only one of
the two that can be attributed to a run.

Neither replaces the other, and an earlier version had only the second one
multiplied by a rate. That is worth spelling out, because it read as authority
and was wrong three ways at once: it counted 455 phantom node-hours from four
attempts whose end was never recorded, it multiplied by a $0.80 rate the biller
has never charged ($0.688), and it had no way to see the 28% of the bill that
is not compute. It reported $574.61 against an actual $316.71 and rose hourly
while the pool sat at zero nodes.

Derived, not sampled. An earlier version than THAT sampled the pool's node count
every 15s from inside the console's server -- which meant it only recorded while
that server happened to be running, so a 24h window was typically 3% observed.
The task log has no such hole: every attempt is written to the share by the node
wrapper with its own start and end, whether or not anything is watching.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from typing import Any

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.cost import billing, node_time
from src.interfaces.commands import tasks as tasks_command
from src.interfaces.commands._base import Command

"""How far back to ask the biller when the window is 'everything'.

Cost Management caps a query at one year and rejects the whole request with a
400 above it. The range is INCLUSIVE of both ends, so 365 days back is 366 days
of data and fails; 364 is the largest that does not. A year covers the whole
life of this subscription with room to spare -- the credit lot was granted in
2026-07 -- so the cap costs nothing here, but it must not be tripped, because
this module answers a rejected query and an unreachable Azure identically.
"""
_ALL_HISTORY_DAYS = 364

"""The shortest window the biller can honestly answer.

Cost Management's finest granularity is a DAY, and its data reaches only to
yesterday. So a windowed billing figure is never the window that was asked for:
`--hours 6` at 21:00 UTC becomes "everything charged since 00:00 today", and
`cost --hours 6` printed `$1.76 billed` directly above `No tasks have run in the
last 6h` -- two numbers that look comparable, describing different intervals,
which is the same class of error as the rate this change fixes.

Below this, billing is SUPPRESSED and the renderer says why. Reporting a wrong
window is worse than reporting none, and the node-time half answers short
windows exactly, which is what a short window is for.
"""
MIN_BILLED_WINDOW_HOURS = 48.0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver cost`."""
    parser.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="Window to report on. 0 (the default) is ALL recorded history, "
        "which is the point of deriving this rather than sampling it.",
    )
    parser.add_argument(
        "--rate",
        default="",
        help="Dollars per node-hour for the DERIVED estimate. Read from "
        "Terraform when omitted; an unreadable rate yields node-hours alone, "
        "because a wrong currency figure is worse than none.",
    )
    parser.add_argument(
        "--skip-billing",
        action="store_true",
        help="Do not ask Cost Management what was actually billed. The derived "
        "node-time half needs no cloud credentials and still answers.",
    )


def _parse_rate(raw: str | None) -> float | None:
    """The number out of `$0.688/hr/node`, or nothing."""
    if not raw:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
    return float(match.group(1)) if match else None


def _rate(explicit: str) -> float | None:
    """The explicit rate, else Terraform's, else nothing.

    Terraform's failure is swallowed: node time is a property of the task log,
    so it should still be reportable on a machine with no cloud credentials
    configured.
    """
    if explicit:
        return _parse_rate(explicit)
    try:
        return _parse_rate(CloudConfig.load().hourly_cost)
    except Exception:  # noqa: BLE001 -- cost is a display value; unavailable reads as unknown
        return None


def _billed(hours: float, now: dt.datetime) -> tuple[dict[str, Any] | None, str | None]:
    """Actual charges over the same window, and why there are none if there are.

    Independently failing, like a `status` panel: the subscription id comes from
    Terraform and the figures from Azure, and neither being reachable may stop
    the task log from being read.
    """
    if 0 < hours < MIN_BILLED_WINDOW_HOURS:
        return None, (
            f"Billed spend is not reported for a {hours:g}h window — Azure bills by whole "
            "days and its data reaches only to yesterday. Node time answers this exactly."
        )
    try:
        subscription = CloudConfig.load().subscription_id
    except Exception:  # noqa: BLE001 -- see the module header: billing is additive
        return None, billing.UNAVAILABLE

    since = (
        (now - dt.timedelta(hours=hours)).date()
        if hours > 0
        else (now - dt.timedelta(days=_ALL_HISTORY_DAYS)).date()
    )
    result, reason = billing.summarise_with_reason(subscription, since=since, until=now.date())
    return (result.as_payload() if result else None), reason


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Summarise node time over the window, and what it was billed as."""
    now = dt.datetime.now(dt.UTC)
    since = now - dt.timedelta(hours=args.hours) if args.hours > 0 else None
    rows = tasks_command.COMMAND.invoke(limit=0, skip_reconcile=False, tasks_dir=None)["rows"]

    totals = node_time.summarise(rows, now=now, since=since)
    rate = _rate(args.rate)
    billed, billed_reason = (None, None) if args.skip_billing else _billed(args.hours, now)
    return {
        "op": "cost",
        "hours": args.hours,
        "rate_per_node_hour": rate,
        "dollars": None if rate is None else totals["task_hours"] * rate,
        "billed": billed,
        # Why there is no billed figure, when there is none. A surface that says
        # "check az login" at a throttled API sends someone to fix an identity
        # that was never broken -- and throttling is the failure that actually
        # happens here, Cost Management being metered far tighter than ARM.
        "billed_reason": billed_reason,
        **totals,
    }


def _render_billed(billed: dict[str, Any], hours: float) -> None:
    """The invoice half. Printed first: it is the one that is actually true."""
    currency = "$" if billed["currency"] == "USD" else f"{billed['currency']} "
    start = billed["first_at"] or billed["since"]
    print(f"Billed since {start} — Azure Cost Management, the authority")
    if hours:
        # Whole days, so a windowed billing figure covers a little MORE than the
        # window asked for. Said out loud rather than left for someone to notice
        # that two adjacent totals do not divide.
        print(f"  (whole days — not exactly the last {hours:g}h; the biller has no finer grain)")
    print(f"  total:       {currency}{billed['total']:,.2f}")
    print(
        f"    pool:      {currency}{billed['pool_cost']:,.2f}"
        f"   over {billed['pool_node_hours']:,.1f} billed node-hours"
    )
    if billed["standing_hours"]:
        # Named, not folded into compute. A machine that is simply switched on
        # bills 24 hours a day and no task log will ever mention it -- which is
        # exactly why it has to be said out loud rather than left to look like
        # the pool's allocation overhead.
        print(
            f"    standing:  {currency}{billed['standing_cost']:,.2f}"
            f"   over {billed['standing_hours']:,.1f} hours — machines left ON:"
        )
        for box in billed["standing"][:3]:
            print(
                f"                 {currency}{box['cost']:>9,.2f}  {box['resource_group']} "
                f"({box['hours']:,.1f} h)"
            )
    print(f"    other:     {currency}{billed['other']:,.2f}   storage, disks, network")
    for line in billed["by_service"][:4]:
        print(f"                 {currency}{line['cost']:>9,.2f}  {line['service']}")
    if billed["as_of"]:
        print(f"  complete to: {billed['as_of']} — cost data lags hours and is restated,")
        print("               so the most recent day always reads low.")


def render(payload: dict[str, Any]) -> None:
    hours = payload["hours"]
    billed = payload.get("billed")
    if billed:
        _render_billed(billed, hours)
        print()
    elif payload.get("billed_reason"):
        print(f"{payload['billed_reason']}\n")

    if not payload["tasks"]:
        window = f" in the last {payload['hours']:g}h" if payload["hours"] else ""
        print(f"No tasks have run{window}.")
        return

    scope = f"last {payload['hours']:g}h" if payload["hours"] else "all recorded history"
    print(f"Node time over {scope} — from the task log, attributable to a run")
    print(f"  node-hours:  {payload['task_hours']:.2f}", end="")
    if payload["dollars"] is not None:
        print(f"   (~${payload['dollars']:.2f} at ${payload['rate_per_node_hour']:.3f}/node-hr)")
    else:
        print("   (rate unknown)")
    print(f"  tasks:        {payload['tasks']:,}, peak {payload['peak_concurrency']} at once")
    print(f"  spanning:    {payload['first_at']} → {payload['last_at']}")
    if payload.get("unended"):
        print(
            f"  EXCLUDED:    {payload['unended']} attempt(s) started and never recorded an end. "
            "Their\n               node time is unknown, not zero — `poker-solver tasks` "
            "lists them."
        )
    # Compared against POOL node-hours specifically. Comparing it against total
    # compute is what made the old caveat wrong: it folded in a standing VM and
    # then blamed the whole gap on allocation overhead.
    pool_hours = (billed or {}).get("pool_node_hours")
    if pool_hours:
        print(
            f"  Counts time tasks were EXECUTING. Against {pool_hours:,.1f} billed POOL "
            f"node-hours\n  that is {pool_hours / payload['task_hours']:.2f}x — a node is "
            "allocated before its task starts\n  and released after it ends, and the record "
            "begins after the first charges.\n  Machines left ON outside the pool are listed "
            "above and are not in this number."
        )
    else:
        print("  Counts time tasks were EXECUTING, so it runs BELOW what the pool was")
        print("  billed: a node is allocated before its task starts and released after.")


COMMAND = Command(
    name="cost",
    help="Billed spend from Azure, and node time derived from the task log.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
