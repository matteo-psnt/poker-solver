"""How much node time the pool has spent, derived from the task log.

This replaces a sampler. Polling the pool's node count only records while
something is polling, and the only thing that polled was the console's own
server -- so a 24h window was routinely 3% observed and the totals meant
nothing. The task log has none of that problem: the node wrapper writes a record
per attempt with its own start and end, on the share, for every task ever run.
The history is complete by construction and needs nothing running to collect it.

**It is a LOWER BOUND on allocated node time, and deliberately so.** A node is
allocated before its task starts and released some time after it ends, and the
pool's spin-up is not free. What is measured here is the interval a task was
actually executing, which is the part the record can prove. Better to
under-report a number that is checkable than to model the rest and be confidently
wrong -- `just credit-check` remains the authority on spend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def instant(value: Any) -> datetime | None:
    """Parse a task-record timestamp, tolerating a missing or malformed one."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def intervals(tasks: list[dict[str, Any]], *, now: datetime) -> list[tuple[datetime, datetime]]:
    """One (start, end) per task that ran.

    A task with no ``ended_at`` is still going, so it is credited up to ``now``
    rather than dropped -- excluding in-flight work would make the number dip
    exactly when the pool is busiest.
    """
    spans: list[tuple[datetime, datetime]] = []
    for task in tasks:
        start = instant(task.get("started_at"))
        if start is None:
            continue
        end = instant(task.get("ended_at")) or now
        if end > start:
            spans.append((start, end))
    return spans


def timeline(spans: list[tuple[datetime, datetime]]) -> list[tuple[datetime, int]]:
    """How many tasks were running at each moment something changed.

    A sweep over start/end events rather than a sampled series: the result is
    exact between events instead of approximate everywhere, and it has no gaps
    to apologise for.
    """
    events: list[tuple[datetime, int]] = []
    for start, end in spans:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda event: event[0])

    running = 0
    out: list[tuple[datetime, int]] = []
    for when, delta in events:
        running += delta
        # Collapse simultaneous events into one point: several tasks starting in
        # the same second is one change in concurrency, not several.
        if out and out[-1][0] == when:
            out[-1] = (when, running)
        else:
            out.append((when, running))
    return out


def summarise(
    tasks: list[dict[str, Any]], *, now: datetime, since: datetime | None = None
) -> dict[str, Any]:
    """Node-time over the window, plus the concurrency series to draw.

    ``task_hours`` is the integral of concurrency, which equals node-hours while
    Batch places one task per node -- what this pool does, since a task uses most
    of a box. Where it does not, this over-counts nodes and under-counts nothing,
    so it stays an upper bound on *nodes* and a lower bound on *allocated time*.
    """
    spans = intervals(tasks, now=now)
    if since is not None:
        spans = [(max(start, since), end) for start, end in spans if end > since]

    task_seconds = sum((end - start).total_seconds() for start, end in spans)
    series = timeline(spans)
    peak = max((count for _, count in series), default=0)

    return {
        "task_hours": task_seconds / 3600.0,
        "tasks": len(spans),
        "peak_concurrency": peak,
        "first_at": series[0][0].isoformat() if series else None,
        "last_at": series[-1][0].isoformat() if series else None,
        "series": [{"at": when.isoformat(), "running": count} for when, count in series],
    }
