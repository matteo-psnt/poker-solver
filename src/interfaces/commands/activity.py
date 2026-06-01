"""The `activity` subcommand: what the commands themselves have been costing.

Reads the local telemetry log :mod:`src.interfaces.telemetry` writes — one row
per command that ran, from either surface. This is the only reader of it, and
it exists because a JSONL file nobody can query is a file nobody looks at.

**Percentiles, not a mean.** The failure mode this is for is a command that is
usually fine and occasionally terrible: `tasks` reconciling against Batch, a
share read that hit a cold cache. A mean folds that into a number that is wrong
about both halves, and the tail is the half someone is complaining about. p50
says what it normally costs and p95 says what it costs when it does not.

Refusals are counted apart from errors, because they are not faults: a refusal
is the command answering "no" to a question it understood — an unpublished run,
two checkpoints that cannot be paired — and a page full of them means someone
is exploring, not that anything is broken.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.interfaces import telemetry
from src.interfaces.commands._base import Command
from src.shared import records

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver activity`."""
    parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help="How far back to look (0 = every row the log still holds).",
    )
    parser.add_argument(
        "--command", default="", help="Only this subcommand, e.g. `--command tasks`."
    )
    parser.add_argument(
        "--surface",
        default="",
        choices=("", "cli", "console", "unknown"),
        help="Only invocations from this surface.",
    )
    parser.add_argument(
        "--failures",
        action="store_true",
        help="List the individual failures instead of summarising. This is the view "
        "that answers 'what actually went wrong', which a percentile cannot.",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Rows to show (0 = all). Slowest first."
    )


def _since(days: float) -> datetime | None:
    """The floor, or None for everything the log still holds."""
    return None if days <= 0 else datetime.now(UTC) - timedelta(days=days)


def _at(row: dict[str, Any]) -> datetime | None:
    """A row's timestamp, or None if it has none this can compare.

    Rows are written by a best-effort writer and read back after an arbitrary
    rotation, so a malformed one is expected rather than exceptional; it is
    dropped from a time-filtered view instead of taking the command down.
    """
    try:
        return datetime.fromisoformat(str(row.get("at", "")))
    except ValueError:
        return None


def _surface_of(row: dict[str, Any]) -> str:
    """Which surface a row claims, with absent and null both reading as unknown.

    One expression, used for the keys AND the counts, because computing them
    two ways is what let them disagree.
    """
    return str(row.get("surface") or "unknown")


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank, on a list already sorted. Exact for the sizes involved.

    Interpolating would be more standard and would invent a duration that never
    happened; at these counts the nearest actual observation is both simpler and
    more honest about what was measured.
    """
    if not values:
        return 0.0
    index = min(len(values) - 1, round(fraction * (len(values) - 1)))
    return values[index]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Summarise the local activity log."""
    path = telemetry.log_path()
    # Both generations, oldest first. Reading only the live file would make a
    # rotation look like the history had been deleted.
    rows = [row for generation in telemetry.logs() for row in records.read_log(generation)]

    floor = _since(args.days)
    selected = [
        row
        for row in rows
        if (not args.command or row.get("command") == args.command)
        and (not args.surface or row.get("surface") == args.surface)
        and (floor is None or ((at := _at(row)) is not None and at >= floor))
    ]

    by_command: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_command[str(row.get("command", "?"))].append(row)

    summary = []
    for command, group in by_command.items():
        seconds = sorted(float(row.get("seconds", 0.0)) for row in group)
        outcomes = [row.get("outcome") for row in group]
        summary.append(
            {
                "command": command,
                "calls": len(group),
                "p50_seconds": round(_percentile(seconds, 0.50), 3),
                "p95_seconds": round(_percentile(seconds, 0.95), 3),
                "max_seconds": round(seconds[-1], 3) if seconds else 0.0,
                # Total time is what says whether a command is worth optimising:
                # 0.4s that runs 3,000 times outranks 9s that runs twice.
                "total_seconds": round(sum(seconds), 1),
                "refusals": outcomes.count("refusal"),
                "errors": outcomes.count("error"),
            }
        )
    summary.sort(key=lambda entry: entry["total_seconds"], reverse=True)

    failures = [
        {
            "at": row.get("at"),
            "command": row.get("command"),
            "surface": row.get("surface"),
            "outcome": row.get("outcome"),
            "error_type": row.get("error_type"),
            "error": row.get("error"),
            "asked": row.get("asked", {}),
        }
        for row in selected
        if row.get("outcome") in {"refusal", "error"}
    ]
    failures.reverse()
    # Counted BEFORE truncation. `--limit` is a display cap, and reporting the
    # capped length as the total said "20 failure(s)" when there were 100.
    total_failures = len(failures)

    if args.limit > 0:
        summary = summary[: args.limit]
        failures = failures[: args.limit]

    return {
        "op": "activity",
        "log": str(path),
        "exists": path.is_file(),
        "enabled": telemetry.enabled(),
        "days": args.days,
        # What the CALLER asked to see, carried so the renderer does not need
        # the args. The console ignores it and chooses for itself — it gets both
        # halves of the payload either way.
        "failures_only": bool(args.failures),
        "rows": len(selected),
        "total_rows": len(rows),
        "first_at": min((str(row.get("at", "")) for row in selected), default=None) or None,
        "commands": summary,
        "failures": failures,
        "total_failures": total_failures,
        # Counted off ONE expression, so the parts sum to `rows`. Deriving the
        # keys and the counts separately meant a row with a missing or null
        # `surface` contributed a key it then failed to match, and the header
        # rendered `unknown 0` beside a total that included it.
        "by_surface": dict(sorted(Counter(_surface_of(row) for row in selected).items())),
    }


def render(payload: dict[str, Any]) -> None:
    if not payload["exists"]:
        # Two different states, and the fix differs. Nothing has run yet is
        # ordinary on a fresh checkout; switched off is a decision someone made.
        reason = (
            "no commands have been recorded yet"
            if payload["enabled"]
            else f"recording is OFF ({telemetry.ENV_VAR} is set)"
        )
        print(f"No activity log at {payload['log']} — {reason}.")
        return
    if not payload["rows"]:
        window = "ever" if payload["days"] <= 0 else f"in the last {payload['days']:g} day(s)"
        print(f"No matching invocations {window} ({payload['total_rows']} row(s) in the log).")
        return

    window = "all time" if payload["days"] <= 0 else f"last {payload['days']:g} day(s)"
    surfaces = ", ".join(f"{name} {count}" for name, count in payload["by_surface"].items())
    print(f"{payload['rows']} invocation(s), {window} — {surfaces}")

    if payload["failures"] and _listing_failures(payload):
        _render_failures(payload)
        return

    header = (
        f"{'command':<20} {'calls':>7} {'p50':>8} {'p95':>8} {'max':>8} "
        f"{'total':>9} {'refused':>8} {'errors':>7}"
    )
    print(header)
    print("-" * len(header))
    for entry in payload["commands"]:
        print(
            f"{entry['command'][:20]:<20} {entry['calls']:>7} "
            f"{entry['p50_seconds']:>8.3f} {entry['p95_seconds']:>8.3f} "
            f"{entry['max_seconds']:>8.3f} {entry['total_seconds']:>9.1f} "
            f"{entry['refusals']:>8} {entry['errors']:>7}"
        )
    total = payload.get("total_failures", len(payload["failures"]))
    if total:
        shown = (
            "" if total == len(payload["failures"]) else f" (showing {len(payload['failures'])})"
        )
        print(f"\n{total} failure(s){shown} — `--failures` to list them.")


def _listing_failures(payload: dict[str, Any]) -> bool:
    """Whether the caller asked for the failure listing.

    Read off the payload rather than the args so both surfaces see the same
    thing: the console gets `failures` either way and chooses for itself.
    """
    return bool(payload.get("failures_only"))


def _render_failures(payload: dict[str, Any]) -> None:
    for failure in payload["failures"]:
        asked = " ".join(f"{key}={value}" for key, value in (failure["asked"] or {}).items())
        print(
            f"{str(failure['at'])[:19]}  {failure['command']!s:<18} "
            f"{failure['outcome']!s:<9} {failure['error_type'] or ''}"
        )
        if asked:
            print(f"    asked: {asked}")
        if failure["error"]:
            print(f"    {failure['error']}")


COMMAND = Command(
    name="activity",
    help="What the commands have been costing: calls, p50/p95, refusals, errors.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
