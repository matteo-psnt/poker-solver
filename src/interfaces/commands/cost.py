"""The `cost` subcommand: node time, derived from the leg log.

Derived, not recorded. An earlier version sampled the pool's node count every
15s from inside the console's server -- which meant it only recorded while that
server happened to be running, so a 24h window was typically 3% observed and
the totals were worthless. The leg log has no such hole: every attempt is
written to the share by the node wrapper with its own start and end, whether or
not anything is watching, so the history is complete back to the first leg.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from typing import Any

from src.interfaces.cloud import node_time
from src.interfaces.commands import legs as legs_command
from src.interfaces.commands._base import Command


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
        help="Dollars per node-hour. Read from Terraform when omitted; an "
        "unreadable rate yields node-hours alone, because a wrong currency "
        "figure is worse than none.",
    )


def _rate(explicit: str) -> float | None:
    """The explicit rate, else Terraform's, else nothing.

    Terraform is asked lazily and its failure swallowed: node time is a property
    of the leg log, so it should still be reportable on a machine with no cloud
    credentials configured.
    """
    import re

    def parse(raw: str | None) -> float | None:
        if not raw:
            return None
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
        return float(match.group(1)) if match else None

    if explicit:
        return parse(explicit)
    try:
        from src.interfaces.cloud.config import CloudConfig

        return parse(CloudConfig.load().hourly_cost)
    except Exception:  # noqa: BLE001 -- cost is a display value; unavailable reads as unknown
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Summarise node time over the window."""
    now = datetime.now(UTC)
    since = now - timedelta(hours=args.hours) if args.hours > 0 else None
    rows = legs_command.COMMAND.invoke(limit=0, skip_reconcile=False, legs_dir=None)["rows"]

    totals = node_time.summarise(rows, now=now, since=since)
    rate = _rate(args.rate)
    return {
        "op": "cost",
        "hours": args.hours,
        "rate_per_node_hour": rate,
        "dollars": None if rate is None else totals["task_hours"] * rate,
        **totals,
    }


def render(payload: dict[str, Any]) -> None:
    if not payload["legs"]:
        window = f" in the last {payload['hours']:g}h" if payload["hours"] else ""
        print(f"No legs have run{window}.")
        return
    scope = f"last {payload['hours']:g}h" if payload["hours"] else "all recorded history"
    print(f"Node time over {scope} — a LOWER BOUND, not billed cost")
    print(f"  node-hours:  {payload['task_hours']:.2f}", end="")
    if payload["dollars"] is not None:
        print(f"   (${payload['dollars']:.2f} at ${payload['rate_per_node_hour']:.2f}/node-hr)")
    else:
        print("   (rate unknown)")
    print(f"  legs:        {payload['legs']:,}, peak {payload['peak_concurrency']} at once")
    print(f"  spanning:    {payload['first_at']} → {payload['last_at']}")
    print("  Counts time legs were EXECUTING; nodes are allocated a little before")
    print("  and released after, so the real allocation is somewhat higher.")


COMMAND = Command(
    name="cost",
    help="Node time derived from the leg log (a lower bound, not billed cost).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
