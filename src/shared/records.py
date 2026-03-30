"""How this project writes and reads everything it records.

Six artifacts grew six hand-rolled conventions. Each writer decided
independently whether to stamp a schema version, whether to survive a torn
file, and whether to write atomically, and no two agreed:

    .run.json               versioned, NOT tolerant, NOT atomic
    progress.jsonl          versioned, tolerant,     NOT atomic
    STATIC_CHECKPOINT.json  NOT versioned, NOT tolerant, atomic
    eval_ledger.jsonl       versioned, tolerant,     atomic
    legs/*.json             versioned, tolerant,     NOT atomic
    baseline.json           NOT versioned, tolerant, atomic

That is not a style problem. ``.run.json`` -- the most-read artifact here --
was written with ``open("w")`` then ``json.dump``, so a kill inside that window
left it truncated, and a truncated ``.run.json`` made a resume die with
JSONDecodeError while intact checkpoints sat beside it.

There are exactly two storage shapes, and this module owns both.

SNAPSHOT is current state: the whole document is rewritten and only the latest
version matters (``.run.json``, the checkpoint manifest, the baseline).

LOG is history: rows are appended and never rewritten, so one file legitimately
spans code versions as a run is resumed across legs (``progress.jsonl``, the
eval ledger).

Atomicity is a property of the destination, not a preference. On local disk a
snapshot is written to a temporary file and renamed, which is atomic. The Azure
share is SMB and has **no atomic rename**, so a share-scoped snapshot is written
directly and safety comes from the layout instead: one writer per file, and
separate files per event so a torn write cannot destroy an earlier record. The
registry records which kind of destination an artifact lives on, so a caller
cannot get this wrong by accident. That is the only axis: ``local`` or
``share``.

Every artifact is declared in :data:`REGISTRY`. A test asserts nothing writes
outside it, so "what does this project store" is answerable by reading one list.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION_KEY = "schema_version"
UNVERSIONED = 0

Kind = Literal["snapshot", "log"]
Scope = Literal["local", "share"]


@dataclass(frozen=True)
class Artifact:
    """One thing this project records, and how.

    ``scope`` is the destination's capability, not a filing category: local disk
    can rename atomically and SMB cannot. Anything finer -- per-run vs global,
    which directory it sits in -- would describe where a file lives without
    changing how it is written, and nothing here would read it.
    """

    name: str
    kind: Kind
    scope: Scope
    version: int
    what: str

    @property
    def atomic(self) -> bool:
        return self.scope != "share"


REGISTRY: dict[str, Artifact] = {
    "run.jsonl": Artifact(
        name="run.jsonl",
        kind="log",
        scope="local",
        version=1,
        what="everything that happened to a run: creation, attempts, checkpoints, status",
    ),
    "STATIC_CHECKPOINT.json": Artifact(
        name="STATIC_CHECKPOINT.json",
        kind="snapshot",
        scope="local",
        version=1,
        what="current snapshot plus the retained ladder",
    ),
    "eval_ledger.jsonl": Artifact(
        name="eval_ledger.jsonl",
        kind="log",
        scope="local",
        version=1,
        what="every recorded evaluation; a rebuildable cache of the per-run records",
    ),
    "evals/*.json": Artifact(
        name="evals/*.json",
        kind="snapshot",
        scope="local",
        version=2,
        what="one evaluation entire: provenance, knobs, and full results with samples",
    ),
    "metadata.json": Artifact(
        name="metadata.json",
        kind="snapshot",
        scope="local",
        version=1,
        what="a precomputed card abstraction's config, hash and per-street shape",
    ),
    "baseline.json": Artifact(
        name="baseline.json",
        kind="snapshot",
        scope="local",
        version=1,
        what="which run is the current baseline, and why it was promoted",
    ),
    "legs/*.start.json": Artifact(
        name="legs/*.start.json",
        kind="snapshot",
        scope="share",
        version=1,
        what="a cloud leg's own account of starting, per task and attempt",
    ),
    "legs/*.exit.json": Artifact(
        name="legs/*.exit.json",
        kind="snapshot",
        scope="share",
        version=1,
        what="a cloud leg's own account of how it ended",
    ),
    "legs/*.observed.json": Artifact(
        name="legs/*.observed.json",
        kind="snapshot",
        scope="share",
        version=1,
        what="what Batch says became of a leg whose own record never arrived",
    ),
}


def stamp(payload: dict[str, Any], artifact: Artifact) -> dict[str, Any]:
    """Return ``payload`` carrying its schema version, first, for ``head -1``."""
    return {SCHEMA_VERSION_KEY: artifact.version, **payload}


def version_of(payload: dict[str, Any]) -> int:
    """Schema version of one record; 0 for anything written before versioning."""
    value = payload.get(SCHEMA_VERSION_KEY, UNVERSIONED)
    return value if isinstance(value, int) else UNVERSIONED


def version_span(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Lowest and highest version present, for reporting a mixed-version file."""
    if not rows:
        return (UNVERSIONED, UNVERSIONED)
    versions = [version_of(r) for r in rows]
    return (min(versions), max(versions))


def write_snapshot(
    path: str | os.PathLike[str], payload: dict[str, Any], artifact: Artifact
) -> None:
    """Replace a snapshot with a new complete version.

    Atomically where the destination allows it: written to a sibling temporary
    file and renamed, so a reader never observes a partial document and a kill
    leaves the previous version intact. On the SMB share there is no atomic
    rename, so the write is direct and the layout carries the safety instead.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(stamp(payload, artifact), indent=2, default=str)
    if not artifact.atomic:
        destination.write_text(body, encoding="utf-8")
        return
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(destination)


def read_snapshot(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read a snapshot, or None if it is absent or unreadable.

    None rather than an exception: a torn file is the expected residue of a kill
    under the pre-atomic writers, and a caller that can carry on without the
    document should not be stopped by it. Callers that genuinely require it say
    so at their own layer, where they can give an actionable message.
    """
    source = Path(path)
    if not source.is_file():
        return None
    try:
        loaded = json.loads(source.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def append_log(path: str | os.PathLike[str], row: dict[str, Any], artifact: Artifact) -> None:
    """Append one stamped row. Raises on IO failure.

    Deliberately propagating: the callers have different and individually
    correct failure policies -- a per-checkpoint writer swallows, the eval
    ledger does not -- and choosing one here would break the other.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stamp(row, artifact), default=str) + "\n")


def read_log(
    path: str | os.PathLike[str],
    *,
    on_bad_line: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Every intact row, oldest first; unparseable lines skipped.

    A process killed mid-append leaves a torn final line. Losing that one row is
    correct; losing the history above it would not be.

    ``on_bad_line`` receives the 1-based line number of anything skipped, so a
    caller with a repair path of its own can name it rather than staying silent.
    """
    source = Path(path)
    if not source.is_file():
        return []
    rows = []
    for number, line in enumerate(
        source.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if on_bad_line is not None:
                on_bad_line(number)
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
