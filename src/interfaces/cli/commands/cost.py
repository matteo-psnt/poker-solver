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
        instant = pool_samples._instant(row)  # noqa: SLF001
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
        "series": [{"at": row["at"], "nodes": row.get("nodes")} for row in rows],
        **totals,
    }


def render(payload: dict[str, Any]) -> None:
    if not payload["samples"]:
        print("No pool samples recorded. The console's server writes them while it runs;")
        print("Batch keeps no node history, so this cannot be backfilled.")
        return
    print(f"Pool allocation over the last {payload['hours']:g}h — NOT billed cost")
    print(f"  samples:     {payload['samples']:,} ({payload['first_at']} → {payload['last_at']})")
    print(f"  node-hours:  {payload['node_hours']:.2f}")
    if payload["dollars"] is not None:
        print(f"  at {payload['rate_per_node_hour']:.2f}/node-hr: ${payload['dollars']:.2f}")
    if payload["unobserved_seconds"]:
        gap = payload["unobserved_seconds"] / 3600.0
        print(f"  {gap:.1f}h unobserved (nothing was recording) — excluded from the total")


COMMAND = Command(
    name="cost",
    help="Node-hours from the recorded pool series (allocation, not billed cost).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
