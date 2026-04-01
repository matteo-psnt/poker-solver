"""The pool's node count over time, and what it integrates to.

Batch retains no node history, so this series exists only if something records
it: it starts when the sampler starts and can never be backfilled. That is the
whole reason it is written to disk rather than held in memory -- a console
restarted twice a day would otherwise always show an empty chart.

**This measures pool ALLOCATION, not billed cost.** It cannot see anything Azure
bills outside the pool, and the rate comes from Terraform rather than an
invoice. It is a trend line; `just credit-check` remains the authority.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/pool_samples.jsonl")

RETENTION = timedelta(days=30)

# A sample says "there were N nodes at time T", and the interval AFTER it is
# credited at N. That is only honest while sampling was actually happening: if
# the server was off for six hours, the last sample before the gap must not bill
# six hours at whatever was running then. Anything longer than this is treated as
# unobserved and reported separately rather than silently integrated.
MAX_GAP_SECONDS = 90.0

_RATE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def append(
    nodes: int, vm_size: str | None, *, path: Path = DEFAULT_PATH, at: str | None = None
) -> None:
    """Record one observation. Append-only, single writer, one line per sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "at": at or datetime.now(UTC).isoformat(timespec="seconds"),
        "nodes": nodes,
        "vm_size": vm_size,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read(path: Path = DEFAULT_PATH) -> list[dict[str, Any]]:
    """Every sample, oldest first, skipping anything unparseable.

    A half-written final line is the expected residue of a process killed
    mid-append, and must not take down the page that would explain why.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "at" in row and "nodes" in row:
            rows.append(row)
    return rows


def prune(path: Path = DEFAULT_PATH, *, retention: timedelta = RETENTION) -> int:
    """Drop samples older than ``retention``. Returns how many were removed.

    Rewrite-and-replace rather than truncate-in-place: the reader is not
    coordinated with the writer, so a half-rewritten file must never be
    observable. Called at startup, not per sample -- at ~5,760 lines a day this
    is a small file and rewriting it every 15s would be pure waste.
    """
    rows = read(path)
    if not rows:
        return 0
    cutoff = datetime.now(UTC) - retention
    kept = [row for row in rows if _instant(row) is not None and _instant(row) >= cutoff]  # type: ignore[operator]
    removed = len(rows) - len(kept)
    if removed:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
        tmp.replace(path)
    return removed


def _instant(row: dict[str, Any]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(row["at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def hourly_rate(hourly_cost: str | None) -> float | None:
    """Dollars per node-hour, out of Terraform's human string (`$0.80/hr/node`).

    ``None`` when it cannot be read, so the page shows node-hours without
    inventing a currency figure -- a wrong dollar number is worse than none.
    """
    if not hourly_cost:
        return None
    match = _RATE.search(hourly_cost)
    return float(match.group(1)) if match else None


def integrate(rows: list[dict[str, Any]], *, max_gap: float = MAX_GAP_SECONDS) -> dict[str, Any]:
    """Node-hours under the step function, plus the time it could not account for.

    Each sample's node count is credited for the interval until the NEXT sample,
    which is what "the pool held N nodes" means for a series of observations.
    Intervals longer than ``max_gap`` are excluded and their duration returned as
    ``unobserved_seconds``: the console shows it, because a total that silently
    omits a downtime gap reads as a complete accounting and is not one.
    """
    instants = [(_instant(row), row) for row in rows]
    usable = [(when, row) for when, row in instants if when is not None]
    usable.sort(key=lambda pair: pair[0])

    node_hours = 0.0
    unobserved = 0.0
    for (start, row), (end, _) in pairwise(usable):
        seconds = (end - start).total_seconds()
        if seconds <= 0:
            continue
        if seconds > max_gap:
            unobserved += seconds
            continue
        node_hours += float(row.get("nodes") or 0) * seconds / 3600.0

    return {
        "node_hours": node_hours,
        "unobserved_seconds": unobserved,
        "samples": len(usable),
        "first_at": usable[0][0].isoformat() if usable else None,
        "last_at": usable[-1][0].isoformat() if usable else None,
    }
