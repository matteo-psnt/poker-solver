"""The `cost` subcommand: node-hours from the recorded pool series.

Reads only what the sampler wrote. It cannot be backfilled -- Batch keeps no
node history -- so an empty answer means "nothing has been recording", which is
said in those words rather than shown as a zero.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands._base import Command
from src.shared import pool_samples


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run cost`."""
    parser.add_argument(
        "--hours", type=float, default=24.0, help="Window to report on (0 = everything recorded)."
    )
    parser.add_argument(
        "--samples-path",
        default=str(pool_samples.DEFAULT_PATH),
        help="Where the sampler writes. Only the console's server records this.",
    )
    parser.add_argument(
        "--rate",
        default="",
        help="Dollars per node-hour. Read from Terraform when omitted; a bad "
        "figure is worse than none, so an unreadable rate yields node-hours alone.",
    )


def _within(rows: list[dict[str, Any]], hours: float) -> list[dict[str, Any]]:
    if hours <= 0:
        return rows
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    kept = []
    for row in rows:
        instant = pool_samples.instant(row)
        if instant is not None and instant >= cutoff:
            kept.append(row)
    return kept


def _rate(args: argparse.Namespace) -> float | None:
    """The explicit rate, else Terraform's, else nothing.

    Terraform is asked lazily and its failure is swallowed: a console with no
    cloud credentials should still be able to show node-hours, which are a
    property of the recording rather than of the account.
    """
    if args.rate:
        return pool_samples.hourly_rate(args.rate)
    try:
        from src.interfaces.cloud.config import CloudConfig

        return pool_samples.hourly_rate(CloudConfig.load().hourly_cost)
    except Exception:
        return None


def _series_with_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The samples, with a null inserted wherever nothing was recording.

    Without this a chart joins the two sides of a ten-hour outage with a flat
    line at the last-seen node count -- which reads as "the pool held 4 nodes
    all night". That is precisely the claim :func:`pool_samples.integrate`
    refuses to make, so a plot that made it would contradict the total printed
    beside it. The break is the honest shape: nothing was known here.
    """
    out: list[dict[str, Any]] = []
    previous: datetime | None = None
    for row in rows:
        instant = pool_samples.instant(row)
        if (
            previous is not None
            and instant is not None
            and (instant - previous).total_seconds() > pool_samples.MAX_GAP_SECONDS
        ):
            out.append({"at": previous.isoformat(), "nodes": None})
        out.append({"at": row["at"], "nodes": row.get("nodes")})
        previous = instant or previous
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Summarise the recorded node series over a window."""
    rows = _within(pool_samples.read(Path(args.samples_path)), args.hours)
    totals = pool_samples.integrate(rows)
    rate = _rate(args)
    return {
        "op": "cost",
        "hours": args.hours,
        "rate_per_node_hour": rate,
        "dollars": None if rate is None else totals["node_hours"] * rate,
        # The series itself, so the page can draw it without a second endpoint.
        "series": _series_with_gaps(rows),
        **totals,
    }


def render(payload: dict[str, Any]) -> None:
    if not payload["samples"]:
        print("No pool samples recorded. The console's server writes them while it runs;")
        print("Batch keeps no node history, so this cannot be backfilled.")
        return
    observed = payload["observed_seconds"] / 3600.0
    window = payload["hours"]
    print("Pool allocation — NOT billed cost")
    print(f"  node-hours:  {payload['node_hours']:.2f}", end="")
    if payload["dollars"] is not None:
        print(f"   (${payload['dollars']:.2f} at ${payload['rate_per_node_hour']:.2f}/node-hr)")
    else:
        print("   (rate unknown)")
    # Coverage before anything else, because it decides how to read the total.
    # $2.15 across 47 observed minutes of a 24h window is not "$2.15 today".
    if window:
        share = observed / window if window else 0.0
        print(f"  observed:    {observed:.2f}h of the last {window:g}h ({share:.0%})")
    else:
        print(f"  observed:    {observed:.2f}h")
    print(f"  window:      {payload['first_at']} → {payload['last_at']}")
    if payload["unobserved_seconds"]:
        gap = payload["unobserved_seconds"] / 3600.0
        print(f"  NOT counted: {gap:.1f}h in which nothing was recording")


COMMAND = Command(
    name="cost",
    help="Node-hours from the recorded pool series (allocation, not billed cost).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
