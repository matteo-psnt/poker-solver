"""Copying a run between the node's data disk and the SMB share.

Four rules, each a production failure rather than a preference, each argued at
the code that honours it:

* publish mid-run -- the node's disk dies with the task
* manifest LAST -- an interrupted copy must not leave the share naming a rung
  that is only half there
* a completion marker per snapshot -- manifest-last cannot protect a single
  directory's copy
* no timestamps, no modes -- the SMB mount refuses ``utime``, and reports it
  as failure only AFTER copying the data

The last one is the trap for an editor: every copy here must stay
:func:`shutil.copyfile`, NOT :func:`shutil.copytree`, whose default
``copy_function`` is ``copy2`` and would reintroduce it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from src.shared import records

if TYPE_CHECKING:
    from pathlib import Path

# The deleted dynamic backend's manifest, recognised only so a run predating the
# static tree is REFUSED rather than fetched and failed several minutes deeper.
LEGACY_MANIFEST = "CHECKPOINT.json"

MANIFESTS = (records.STATIC_CHECKPOINT, LEGACY_MANIFEST)

# WRITE-ONCE only: the completion marker doubles as "already published, skip it",
# which is wrong for a directory the trainer revisits. ``evals/`` grows, so
# marking it would freeze it at its first published state.
SNAPSHOT_PREFIXES = ("static-", "checkpoint-", "keys-")

MARKER_PREFIX = ".complete-"

# A zarr rung is thousands of small files and a single-stream SMB copy is
# latency-bound, not bandwidth-bound: ~750 MB took ~9 minutes (1.4 MB/s) with
# one thread. Override with `POKER_SOLVER_PUBLISH_WORKERS` -- 1 is the serial
# arm a throughput measurement compares against.
COPY_WORKERS_ENV = "POKER_SOLVER_PUBLISH_WORKERS"
DEFAULT_COPY_WORKERS = 16

Log = Callable[[str], None]


def _quiet(message: str) -> None:
    """Default sink, so every function here is callable from a test."""


def is_snapshot(name: str) -> bool:
    return name.startswith(SNAPSHOT_PREFIXES)


def marker_for(snapshot: str) -> str:
    return MARKER_PREFIX + snapshot


def needs_copy(source: Path, destination: Path) -> bool:
    """``cp -u``: copy when the destination is missing or older.

    Correct precisely BECAUSE timestamps are not preserved. The destination
    takes the copy time, which is newer than the source it came from, so an
    already-published file compares as up to date while a genuinely newer one
    does not.
    """
    if not destination.exists():
        return True
    return source.stat().st_mtime > destination.stat().st_mtime


def copy_workers() -> int:
    """How many files are copied at once. Clamped to 1..64, never zero."""
    try:
        wanted = int(os.environ.get(COPY_WORKERS_ENV) or DEFAULT_COPY_WORKERS)
    except ValueError:
        wanted = DEFAULT_COPY_WORKERS
    return max(1, min(64, wanted))


def copy_file(source: Path, destination: Path) -> None:
    """Content only -- no mode, no timestamps. See the module docstring."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_tree(source: Path, destination: Path, *, update: bool = True, atomic: bool = False) -> int:
    """Merge ``source`` into ``destination`` file by file; returns bytes copied.

    Merging rather than replacing is what makes an interrupted publish
    resumable. ``update=False`` copies unconditionally, for the fetch
    direction -- where a file already on the node is not evidence of a complete
    copy but of a cancelled task. ``atomic`` gives every file the ``.partial``
    dance :func:`_copy_one` does, for trees a reader fetches back (``evals/``).
    """
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    # Directories FIRST and serially, so the parallel pass below never races two
    # threads creating one parent -- and pays no per-file `mkdir` round trip.
    for item in sorted(source.rglob("*")):
        target = destination / item.relative_to(source)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            files.append((item, target))

    def transfer(pair: tuple[Path, Path]) -> int:
        item, target = pair
        if update and not needs_copy(item, target):
            return 0
        size = item.stat().st_size
        if atomic:
            partial = target.with_name(target.name + ".partial")
            shutil.copyfile(item, partial)
            partial.replace(target)
        else:
            shutil.copyfile(item, target)
        return size

    if not files:
        return 0
    with ThreadPoolExecutor(max_workers=min(copy_workers(), len(files))) as pool:
        return sum(pool.map(transfer, files))


def publish_run(run_dir: Path, destination: Path, log: Log = _quiet) -> bool:
    """Copy one run directory to the share. Returns False if anything failed.

    Idempotent and safe to call while training continues, which is what lets
    the mid-run watcher use it. Never raises: a failed publish must not kill a
    task that is still making progress on local disk.
    """
    destination.mkdir(parents=True, exist_ok=True)
    # READ BEFORE THE LADDER, published after it. A rung takes minutes over SMB
    # and the trainer commits new ones meanwhile, so a manifest read at the END
    # names rungs this pass never saw and the share advertises a rung it does
    # not hold. The snapshot always predates the manifest that names it, so a
    # manifest read first can only name rungs `children` below already has.
    manifests = {name: _read_bytes(run_dir / name) for name in MANIFESTS}
    children = sorted(run_dir.iterdir())
    failed = False

    for child in children:
        if not child.is_dir():
            continue
        if not is_snapshot(child.name):
            failed |= not _copy_dir(child, destination / child.name, log, atomic=True)
            continue

        marker = destination / marker_for(child.name)
        # ALREADY COMPLETE => NOTHING TO DO. Not just an optimisation: the
        # republish below drops the marker first, so re-copying a known-good
        # rung leaves it briefly unmarked and a task dying in that window makes
        # the manifest name a rung the next fetch refuses. Nor is it rare --
        # measured at 6.6 minutes re-uploading 809 MB already on the share.
        if marker.exists():
            continue
        _unlink(marker)
        if _publish_snapshot(child, destination / child.name, log):
            _touch(marker)
        else:
            failed = True

    # Loose files -- .run.json, metrics.jsonl, result json -- manifests excluded.
    for child in children:
        if child.is_file() and child.name not in MANIFESTS:
            failed |= not _copy_one(child, destination / child.name, log)

    if failed:
        # Reported, never swallowed: a publish that silently fails every time
        # turns "a killed task loses one rung" into "a killed task loses
        # everything".
        log(f"WARN publish incomplete for {run_dir.name} -- manifest NOT updated, so the")
        log("     share still describes the last fully-copied checkpoint.")
        return False

    # BOTH manifest names, and only now. The static backend's was once copied
    # by the unguarded loose-file pass above, so it was published even when a
    # snapshot copy had failed -- a manifest naming a half-copied rung, exactly
    # what publishing the manifest last exists to prevent.
    for name, body in manifests.items():
        if body is None:
            continue
        if not _rungs_landed(body, destination, log) or not _write_one(
            body, destination / name, log
        ):
            return False
    log(f"published {run_dir.name}")
    return True


def _rungs_landed(manifest: bytes, destination: Path, log: Log) -> bool:
    """Is every rung this manifest names actually a directory on the share?

    ABSENCE only. A rung present but unmarked is refused at FETCH time by
    :func:`require_complete`, which is where that case belongs -- refusing it
    here would freeze manifest publishing for the run forever, since nothing in
    a later publish can add a marker to a rung this node no longer has.
    """
    named = _named_rungs(manifest)
    missing = sorted(name for name in named if not (destination / name).is_dir())
    if missing:
        log(f"WARN manifest names {', '.join(missing)}, not on the share -- NOT publishing it")
        return False
    unmarked = sorted(n for n in named if not (destination / marker_for(n)).exists())
    if unmarked:
        log(f"WARN manifest names unmarked rung(s) {', '.join(unmarked)}; a fetch will refuse them")
    return True


def _named_rungs(manifest: bytes) -> set[str]:
    """Snapshot directory names a manifest advertises: current plus retained."""
    try:
        parsed = json.loads(manifest)
    except ValueError:
        return set()
    if not isinstance(parsed, dict):
        return set()
    entries = [parsed, *_entries(parsed.get("retained"))]
    return {str(entry["zarr"]) for entry in entries if isinstance(entry.get("zarr"), str)}


def _publish_snapshot(source: Path, destination: Path, log: Log) -> bool:
    """One rung, announced before and measured after.

    UNCONDITIONAL, not the update rule. An unmarked destination is the residue
    of an interrupted publish, and the file that was mid-copy is TRUNCATED yet
    NEWER than the source -- so the update rule would skip exactly the wrong
    file and the marker would bless it. `cp -ru` had this hole too; a Batch
    retry is what triggers it.
    """
    log(f"publishing {source.name} ({_megabytes(source):.0f} MB)")
    started = time.monotonic()
    try:
        copied = copy_tree(source, destination, update=False)
    except OSError as error:
        log(f"WARN copying {source.name} failed: {error}")
        return False
    if copied == 0:
        # Never legitimate for a checkpoint, and the marker would bless it: the
        # trainer prunes rungs the current manifest drops, so a source can
        # vanish between the `is_dir` above and this copy.
        log(f"WARN {source.name} copied 0 bytes -- not marking it complete")
        return False
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = copied / 1e6 / elapsed
    log(f"published {source.name}: {copied / 1e6:.0f} MB in {elapsed:.0f}s ({rate:.1f} MB/s)")
    return True


def _megabytes(directory: Path) -> float:
    """Local disk, so the stats are free next to the copy they precede."""
    with contextlib.suppress(OSError):
        return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1e6
    return 0.0


def _copy_dir(source: Path, destination: Path, log: Log, *, atomic: bool = False) -> bool:
    try:
        copy_tree(source, destination, atomic=atomic)
    except OSError as error:
        log(f"WARN copying {source.name} failed: {error}")
        return False
    return True


def _copy_one(source: Path, destination: Path, log: Log) -> bool:
    """A loose file, published atomically: a task killed mid-copy must not
    leave a 0-byte ``run.jsonl`` on the share, which every later fetch of the
    run pulls and then refuses as "no run record". Snapshots have their
    completion markers for this; loose files had nothing (measured 08-23: two
    reference runs' records zeroed under retrying evaluate tasks)."""
    # An empty loose file is never content: it is the residue of a truncating
    # publish, fetched back by a later task. Publishing it would spread the
    # zeroing to every copy of the run (measured 08-23: a restored record was
    # re-zeroed within minutes by tasks holding poisoned fetches). Skipping is
    # success -- the share keeps what it has.
    try:
        if source.stat().st_size == 0:
            log(f"skip publishing empty {source.name} (a record is never 0 bytes)")
            return True
    except OSError:
        return True
    partial = destination.with_name(destination.name + ".partial")
    try:
        copy_file(source, partial)
        partial.replace(destination)
    except OSError as error:
        log(f"WARN copying {source.name} failed: {error}")
        _unlink(partial)
        return False
    return True


def _read_bytes(path: Path) -> bytes | None:
    """A file's bytes, or None when it is absent, unreadable or empty."""
    try:
        body = path.read_bytes()
    except OSError:
        return None
    return body or None


def _write_one(body: bytes, destination: Path, log: Log) -> bool:
    """:func:`_copy_one` for bytes already in hand -- the captured manifest."""
    partial = destination.with_name(destination.name + ".partial")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(body)
        partial.replace(destination)
    except OSError as error:
        log(f"WARN writing {destination.name} failed: {error}")
        _unlink(partial)
        return False
    return True


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _touch(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.write_text("")


class FetchRefusedError(Exception):
    """The share cannot supply what this task needs, and guessing would be worse.

    Raised rather than logged because every case is one where continuing means
    training or scoring against data that is absent, truncated, or written by a
    backend this tree cannot read -- each of which surfaces minutes later as a
    confusing error in a different subsystem.
    """


def read_manifest(manifest: Path) -> dict:
    """Parse a checkpoint manifest, or ``{}`` if it is absent or torn."""
    try:
        parsed = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fetch_metadata(source: Path, destination: Path) -> None:
    """Everything that is not a snapshot: .run.json, metrics, eval records."""
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.name.startswith(MARKER_PREFIX) or is_snapshot(child.name):
            continue
        if child.is_dir():
            copy_tree(child, destination / child.name, update=False)
        else:
            copy_file(child, destination / child.name)


def fetch_snapshot(source: Path, destination: Path, name: str) -> None:
    """Copy one snapshot down, replacing whatever is on the node.

    Remove first, and no update check. A cancelled task leaves partial rungs on
    the node, and treating those as already-present means the next task
    inherits a TRUNCATED checkpoint and dies inside zarr. That is what happened
    to rung 10000000: "fetched" in one second, then a read error. Node-local
    state is never evidence of a complete copy.
    """
    target = destination / name
    shutil.rmtree(target, ignore_errors=True)
    copy_tree(source / name, target, update=False)


def require_complete(source: Path, name: str) -> None:
    """A rung without its marker is either pre-marker or was interrupted.

    The two are indistinguishable from here, and loading a truncated one yields
    a corrupt-chunk error deep inside zarr several minutes later, so refuse.

    There is deliberately no repair path. One existed, was never once run, and
    could only ever have helped runs published before markers existed -- all of
    which are gone. A rung that lands unmarked now means a publish was cut off,
    and the answer to that is to publish it again from the node that has it,
    not to bless whatever reached the share.
    """
    if not (source / name).is_dir():
        raise FetchRefusedError(f"the manifest names {name} but it is not on the share")
    if not (source / marker_for(name)).exists():
        raise FetchRefusedError(
            f"{name} has no completion marker -- refusing a possibly-partial "
            f"snapshot. Re-publish it from the node that produced it."
        )


def fetch_current_rung(source: Path, destination: Path, log: Log = _quiet) -> str:
    """Fetch the one rung the manifest calls current. Returns its name, or "".

    What both continuing a run and scoring "the latest checkpoint" need, and in
    both cases ONE rung, not the ladder. Taking the whole retained ladder was 31
    rungs, ~25 GB over SMB and ~40 minutes, to load the 809 MB actually read.

    Leaving the older rungs on the share loses nothing: ``_extend_ladder``
    builds the next manifest from the PREVIOUS manifest rather than from what
    is on disk, ``_prune`` only deletes what the manifest does not name, and
    publish copies per directory -- so rungs this node never had are neither
    re-uploaded nor removed.
    """
    fetch_metadata(source, destination)
    if (source / LEGACY_MANIFEST).is_file() and not (source / records.STATIC_CHECKPOINT).is_file():
        raise FetchRefusedError(
            f"{source.name} was trained by the dynamic backend, which no longer "
            f"exists. Its checkpoints are unreadable at HEAD by design, so this "
            f"run cannot be continued."
        )
    manifest = read_manifest(source / records.STATIC_CHECKPOINT)
    if not manifest:
        # An absent manifest is not an error: a task that died before its first
        # checkpoint publishes .run.json and nothing else, and the right thing
        # is to start the ladder rather than refuse.
        log(f"no published checkpoint for {source.name}")
        return ""
    current = manifest.get("zarr") or ""
    if not current:
        raise FetchRefusedError(
            f"{records.STATIC_CHECKPOINT} on the share names no current snapshot"
        )
    require_complete(source, current)
    fetch_snapshot(source, destination, current)
    copy_file(source / records.STATIC_CHECKPOINT, destination / records.STATIC_CHECKPOINT)
    log(f"fetched current rung {current} (ladder left on the share)")
    return current


def fetch_for_evaluation(
    source: Path, destination: Path, rungs: Sequence[str], log: Log = _quiet
) -> list[str]:
    """Fetch only the rungs being scored. Returns the ones that arrived.

    Selective because the whole ladder is thirty ~540 MB rungs, ~16 GB, to
    score three of them. A rung that cannot be fetched is skipped and named
    rather than fatal: a partial curve beats none.
    """
    fetch_metadata(source, destination)
    fetched = []
    for rung in rungs:
        name = f"static-{rung}.zarr"
        try:
            require_complete(source, name)
        except FetchRefusedError as refusal:
            log(f"  WARN rung {rung}: {refusal}")
            continue
        try:
            fetch_snapshot(source, destination, name)
        except OSError as error:
            # Reported, not swallowed: a silent copy failure becomes a
            # confusing load error minutes later, in a different subsystem.
            log(f"  WARN rung {rung} copy FAILED: {error}")
            continue
        fetched.append(rung)
        log(f"  fetched rung {rung}")
    return fetched


def ladder_state(run_dir: Path) -> str:
    """A fingerprint of ONE run's publishable progress, for the watcher.

    One run, never the runs directory: a node is reused, and an evaluate task
    leaves every checkpoint it fetched under ``runs/``. A watcher that walked
    the directory pushed all of them back to the share on every tick -- ~30
    minutes per training task re-uploading other runs' ladders. The run's id is
    fixed before the trainer starts (``plan.train_run_id``), so an absent
    directory simply reads as empty until the first rung lands.

    ``iteration`` as well as the retained ladder: with ``checkpoint_every``
    below the retain interval the current snapshot advances while the ladder
    does not, and watching only the ladder would sit idle through exactly those
    chunks.
    """
    parts = []
    for name in MANIFESTS:
        manifest = read_manifest(run_dir / name)
        if not manifest:
            continue
        retained = ",".join(
            str(entry.get("iteration", "")) for entry in _entries(manifest.get("retained"))
        )
        parts.append(f"{run_dir.name}:{manifest.get('iteration', '')}:{retained}")
    return "|".join(parts)


def _entries(retained: object) -> Iterable[dict]:
    if not isinstance(retained, list):
        return []
    return [entry for entry in retained if isinstance(entry, dict)]
