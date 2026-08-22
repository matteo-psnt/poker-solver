"""How this project writes and reads everything it records.

Six artifacts had grown six hand-rolled conventions, disagreeing on all three
axes -- schema version, tolerance of a torn file, atomicity. There are exactly
two storage shapes, and this module owns both.

SNAPSHOT is current state: the whole document is rewritten and only the latest
version matters (``.run.json``, the checkpoint manifest).

LOG is history: rows are appended and never rewritten, so one file legitimately
spans code versions as a run is resumed across tasks (``progress.jsonl``, the
eval ledger).

Atomicity is a property of the DESTINATION, not a preference. On local disk a
snapshot is written to a temporary file and renamed. The Azure share is SMB and
has **no atomic rename**, so a share-scoped snapshot is written directly and
safety comes from the layout instead: one writer per file, separate files per
event. The registry records which kind of destination an artifact lives on, so
a caller cannot get this wrong by accident.

Every artifact is declared in :data:`REGISTRY`, with where it lives and how it
grows; a test asserts nothing writes outside it. ``growth`` is prose because it
is read by a person deciding what to compact -- two artifacts accumulate
forever and nothing prunes either: ``legs/*`` gains four files per task-attempt,
and code snapshots gain a tarball per submit.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.shared.jsonio import json_default

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

SCHEMA_VERSION_KEY = "schema_version"
UNVERSIONED = 0

# Named here rather than beside the parser because four unrelated places need the
# string and only one can import the parser: the node wrapper is stdlib-only, and
# `interfaces.cloud` reaching `engine` would put Azure code one import from the
# solver. Spelled once, so a rename cannot leave a copy behind.
STATIC_CHECKPOINT = "STATIC_CHECKPOINT.json"

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
    where: str
    growth: str

    @property
    def atomic(self) -> bool:
        return self.scope != "share"


REGISTRY: dict[str, Artifact] = {
    "run.jsonl": Artifact(
        name="run.jsonl",
        kind="log",
        scope="local",
        version=2,
        what="everything that happened to a run: creation, attempts, checkpoints, status",
        where="<run_dir>/run.jsonl",
        growth="one row per event; one file per run, read whole",
    ),
    STATIC_CHECKPOINT: Artifact(
        name=STATIC_CHECKPOINT,
        kind="snapshot",
        scope="local",
        version=1,
        what="current snapshot plus the retained ladder",
        where=f"<run_dir>/{STATIC_CHECKPOINT}",
        growth="rewritten in place; the ladder it names is what actually grows",
    ),
    "eval_ledger.jsonl": Artifact(
        name="eval_ledger.jsonl",
        kind="log",
        scope="local",
        version=1,
        what="every recorded evaluation; a rebuildable cache of the per-run records",
        where="<runs_dir>/eval_ledger.jsonl, in the throwaway tree a reader materialises",
        growth="DERIVED on every read and discarded; never accumulates",
    ),
    "evals/*.json": Artifact(
        name="evals/*.json",
        kind="snapshot",
        scope="local",
        version=3,
        what="one evaluation entire: provenance, knobs, and full results with samples",
        where="<run_dir>/evals/<timestamp>-<config>-<slug>.json",
        growth="one file per evaluation, all of them read to rebuild the ledger",
    ),
    "metadata.json": Artifact(
        name="metadata.json",
        kind="snapshot",
        scope="local",
        version=1,
        what="a precomputed card abstraction's config, hash and per-street shape",
        where="<abstraction_dir>/metadata.json",
        growth="one per abstraction; rewritten in place",
    ),
    "telemetry/invocations.jsonl": Artifact(
        name="telemetry/invocations.jsonl",
        kind="log",
        scope="local",
        version=1,
        what="one row per command that ran: how long, from which surface, how it ended",
        where="$POKER_SOLVER_CACHE/telemetry/invocations.jsonl (never the share)",
        growth="one row per invocation, so a console tab adds ~1,200/hour -- rotated at "
        "8 MB keeping one generation. The ONLY entry here that is disposable by "
        "design: it is a cache in the `shared.cache` sense, not a record. On the "
        "share it would be an append with no atomic rename, a per-document scheme "
        "outgrowing legs/ in hours, and a round trip added to every command",
    ),
    "legs/*.start.json": Artifact(
        name="legs/*.start.json",
        kind="snapshot",
        scope="share",
        version=3,
        what="a cloud task's own account of starting, per task and attempt",
        where="<share>/legs/<task_id>.<attempt>.start.json",
        growth="UNBOUNDED -- one per task attempt, forever, and `tasks` joins ALL of them",
    ),
    "legs/*.exit.json": Artifact(
        name="legs/*.exit.json",
        kind="snapshot",
        scope="share",
        version=3,
        what="a cloud task's own account of how it ended",
        where="<share>/legs/<task_id>.<attempt>.exit.json",
        growth="UNBOUNDED, as above. Its PRESENCE is what makes an attempt sealed, "
        "so it is also the only safe test for what could be compacted away",
    ),
    "evaluate-progress.json": Artifact(
        name="evaluate-progress.json",
        kind="snapshot",
        scope="local",
        version=1,
        what="flop branches walked so far, while a checkpoint is being scored",
        where="<run_dir>/evaluate-progress.json",
        growth="live-only; overwritten while a task runs and meaningless after it",
    ),
    "precompute-progress.json": Artifact(
        name="precompute-progress.json",
        kind="snapshot",
        scope="local",
        version=1,
        what="runouts enumerated so far, while a card abstraction is being built",
        where="<abstraction_dir>/precompute-progress.json",
        growth="live-only, as above. Separate from evaluate's only because the two "
        "are written by different services into different trees; the SHAPE is identical",
    ),
    "train-progress.json": Artifact(
        name="train-progress.json",
        kind="snapshot",
        scope="local",
        version=1,
        what="iterations trained so far, between checkpoints",
        where="<node scratch>/train-progress.json",
        growth="live-only, as above. The CHECKPOINT is the durable answer; this exists "
        "because a checkpoint lands every million iterations and a bar cannot wait",
    ),
    "legs/*.bundle.json": Artifact(
        name="legs/*.bundle.json",
        kind="snapshot",
        scope="share",
        version=1,
        what="many SEALED leg documents in one file, so reading them is one round trip, "
        "plus a `compactions` list recording each act that produced it -- the only "
        "record anywhere that a compaction happened",
        where="<share>/legs/<label>.bundle.json",
        growth="one per LABEL, not one per compaction: a second round at the same label "
        "ABSORBS the bundle already there rather than replacing it, because the "
        "documents it holds are no longer loose and exist nowhere else. It shrinks "
        "the directory -- the only entry here that does. Bundling an unsealed attempt "
        "would strand its `.observed.json` reconciliation, so only attempts with a "
        "terminal `.exit.json` go in",
    ),
    "legs/*.progress.json": Artifact(
        name="legs/*.progress.json",
        kind="snapshot",
        scope="share",
        version=1,
        what="how far along a running task is, refreshed while it runs",
        where="<share>/legs/<task_id>.progress.json",
        growth="one per task, overwritten as it runs; NOT per attempt, so it is "
        "stale rather than absent once the task ends",
    ),
    "legs/*.observed.json": Artifact(
        name="legs/*.observed.json",
        kind="snapshot",
        scope="share",
        version=1,
        what="what Batch says became of a task whose own record never arrived",
        where="<share>/legs/<task_id>.observed.json",
        growth="one per unreconciled task; written by the READER, not the node",
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
    body = json.dumps(stamp(payload, artifact), indent=2, default=json_default)
    if not artifact.atomic:
        destination.write_text(body, encoding="utf-8")
        return
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(destination)


def progress_writer(
    path: str | os.PathLike[str] | None, artifact: Artifact
) -> Callable[[int, int], None] | None:
    """A ``(done, total)`` callback that publishes completion, or None.

    Long unobservable jobs are the case: a precompute writes nothing until
    ``save()``, and scoring one checkpoint is ~10 minutes of silence -- so
    without this a multi-hour build and a hung one look identical from outside.
    ``None`` in, ``None`` out, so a caller with nowhere to publish passes the
    result straight through to whatever takes an optional callback.

    NEVER FATAL. The work must not die because the thing describing it could not
    be written, which is why every failure here is suppressed rather than
    reported: this is the observer, not the job.
    """
    if path is None:
        return None

    def write(done: int, total: int) -> None:
        with contextlib.suppress(OSError):
            write_snapshot(path, {"done": done, "total": total}, artifact)

    return write


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
        handle.write(json.dumps(stamp(row, artifact), default=json_default) + "\n")


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
