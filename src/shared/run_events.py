"""A run is an append-only log of what happened to it.

Every fact about a run is an event appended to ``<run>/run.jsonl`` and the
current state is what you get by folding them. Nothing is ever rewritten, so
there is no torn-write window -- a kill during a metadata write cannot strand a
run whose checkpoints are fine -- and extending the record means adding an event
type rather than a file, a writer and a reader.

What is NOT in here, and why:

* Bulk numeric arrays. Checkpoints are zarr sidecars and evaluations are
  documents under ``evals/``; the log names them and holds no arrays itself.
  Folding a log to list runs must stay cheap, and 180K of samples per evaluation
  would end that.
* ``STATIC_CHECKPOINT.json``. The loader resolves which arrays to mmap through
  it, atomically, on every load. Folding a log to answer that would be slower
  and less safe.

Listing stays cheap despite the fold: everything a run listing needs is either
in the FIRST event (identity, config, provenance) or the last event that carries
it (iterations, status), so :func:`head` and :func:`tail_value` answer without
reading the middle.
"""

from __future__ import annotations

import itertools
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared import records

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping

RUN_LOG_FILENAME = "run.jsonl"

# The vocabulary. Adding to a run's record means adding to this list -- not a
# file, a writer and a reader.
CREATED = "created"
ATTEMPT_STARTED = "attempt_started"
ATTEMPT_ENDED = "attempt_ended"
PROGRESS = "progress"
CHECKPOINT = "checkpoint"
STATUS = "status"

EVENT_KEY = "event"


# A fixed namespace, so the id an event gets does not depend on when or where it
# was computed.
_EVENT_NAMESPACE = uuid.UUID("6f9b1c2a-4a2e-4c1b-9a7f-2c1d3e4f5a6b")


def event_identity(run_id: str, event: str, body: Mapping[str, Any]) -> uuid.UUID:
    """The id one event has WHEREVER it is written.

    Both writers must agree or the record double-counts. They did not: the sink
    assigned `uuid4()` and the importer a positional `uuid5`, so every event
    written live and then re-imported landed TWICE -- same run, same timestamp,
    two rows. Measured on a task that ran and was then backfilled: 6 events on
    the share, 11 in the database.

    CONTENT-ADDRESSED, and the position the importer used to include is gone.
    Position is unknowable to a live writer, which is the whole problem. The
    trade is that two events identical in run, kind, timestamp AND body collapse
    to one -- and events that agree to the microsecond on every field they carry
    are the same event by any definition a reader can act on.

    `schema_version` is excluded: the file carries it and the sink does not, and
    an id that depended on it would disagree across the two for that reason
    alone.
    """
    material = json.dumps(
        {
            "run": run_id,
            "event": event,
            **{k: v for k, v in body.items() if k not in ("schema_version", "event")},
        },
        sort_keys=True,
        default=str,
    )
    return uuid.uuid5(_EVENT_NAMESPACE, material)


def rung_uri(run_id: str, iteration: int) -> str:
    """Where a rung lives, as the record names it.

    LOGICAL today. Snapshots are zarr directories on the share, so this is the
    name a rung is claimed under rather than an object that has been committed
    -- which is what it becomes when they move to blob storage, and why the
    claim is worth recording under the eventual name now.

    Here because two writers produce it -- the trainer as it checkpoints and
    the importer as it read -- and a name spelled two ways is two rungs
    where the record should hold one.
    """
    return f"rungs/{run_id}/{iteration}"


def log_path(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / RUN_LOG_FILENAME


def read(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every intact event, oldest first. A torn final line is skipped."""
    return records.read_log(log_path(run_dir))


def events_of(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get(EVENT_KEY) == kind]


def head(events: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``created`` event -- identity, config and provenance.

    First by construction, so a listing reads one line rather than folding.
    """
    for event in events:
        if event.get(EVENT_KEY) == CREATED:
            return event
    return {}


def tail_value(
    events: list[dict[str, Any]],
    field: str,
    default: Any = None,
    *,
    kind: str | None = None,
) -> Any:
    """The most recent value of a field that changes over a run's life.

    Scanned from the end: ``iterations`` and ``status`` are answered by the last
    event carrying them, without reading everything before it.

    ``kind`` restricts the scan to one event type, and is not optional in
    spirit: a bare field name is only safe when nothing else uses it. A run's
    ``status`` and an ATTEMPT's status are different facts, and reading the
    former without this returned the latter -- two runs still training were
    folded back as ``died`` because their last closed attempt had.
    """
    for event in reversed(events):
        if kind is not None and event.get(EVENT_KEY) != kind:
            continue
        if field in event:
            return event[field]
    return default


def checkpoints(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The per-checkpoint series: coverage, visits and throughput over the run."""
    return events_of(events, CHECKPOINT)


def plateau_iteration(rows: list[dict[str, Any]], *, rel_gain: float = 0.01) -> int | None:
    """First iteration after which coverage stopped climbing.

    Requires TWO consecutive intervals below the threshold, and reports the
    first of them. One flat interval is noise -- a chunk that happened to
    revisit -- and this number is quotable enough that a false positive would
    end a run early. Returns None while coverage is still climbing.
    """
    usable = [r for r in rows if isinstance(r.get("coverage"), int | float)]
    flat_since: int | None = None
    for previous, current in itertools.pairwise(usable):
        if previous["coverage"] <= 0:
            continue
        gain = (current["coverage"] - previous["coverage"]) / previous["coverage"]
        if gain >= rel_gain:
            flat_since = None
        elif flat_since is None:
            flat_since = int(current["iteration"])
        else:
            return flat_since
    return None
