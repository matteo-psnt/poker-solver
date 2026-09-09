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
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from src.shared import records
from src.shared.cloudtask.node import blobstore

if TYPE_CHECKING:
    from pathlib import Path
from uuid import uuid4

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
# latency-bound, not bandwidth-bound: MEASURED on a pool node at ~93 ms per
# file either way, so a 300M rung's 5,508 files are ~8.5 minutes serial and
# ~1.2 at 16 threads (7.0x over a 478 MB ladder, 2026-08-24). Override with
# `POKER_SOLVER_PUBLISH_WORKERS`; 1 is the serial arm that was measured.
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
    copy but of a cancelled task. ``atomic`` gives every file a ``.partial``
    staging name and one rename, for trees a reader fetches back (``evals/``).
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
            partial = _partial_path(target)
            shutil.copyfile(item, partial)
            partial.replace(target)
        else:
            shutil.copyfile(item, target)
        return size

    if not files:
        return 0
    with ThreadPoolExecutor(max_workers=min(copy_workers(), len(files))) as pool:
        return sum(pool.map(transfer, files))


def _partial_path(destination: Path) -> Path:
    """A staging name unique to THIS writer.

    `<name>.partial` is deterministic, so two sessions publishing the same run
    stage to the SAME path: one `replace()` moves it and the other raises
    ENOENT, aborting a copy whose file was already written correctly. Measured
    09-03 -- two scoring tasks scored fine, logged "1 scored, 0 failed", and
    never reached the share because a concurrent publisher won the rename.
    """
    return destination.with_name(f"{destination.name}.{os.getpid()}-{uuid4().hex[:8]}.partial")


def publish_run(run_dir: Path, run_id: str, sas: str, log: Log = _quiet) -> bool:
    """Put one run's snapshots, metadata and manifests in the container.

    Idempotent and safe to call while training continues, which is what lets
    the mid-run watcher use it. Never raises: a failed publish must not kill a
    task that is still making progress on local disk.

    THE ORDER IS THE CONTRACT. Rungs, then loose metadata, then the manifests
    that name them -- because a manifest which lands before its snapshot
    advertises a rung nothing can fetch. This used to be two passes, a share
    copy that published the manifest and a separate Blob pass that uploaded the
    rungs, and they ran in that order: every tick of every training task PUT a
    container manifest naming a rung the container did not yet hold, then
    logged `manifest names static-N.ckpt.zst, nowhere to be found` because it
    checked before the upload it was racing. One pass cannot invert its own
    ordering.

    Existence is a HEAD against the object itself rather than a marker beside
    it: one rung is one atomically-committed blob, so there is no half-written
    state to guard against and nothing to keep in step.
    """
    if not sas:
        return False
    # READ BEFORE THE LADDER, published after it. The trainer commits new rungs
    # while this runs, so a manifest read at the END names rungs this pass never
    # uploaded. A snapshot always predates the manifest naming it, so a manifest
    # read first can only name rungs the pass below has already considered.
    manifests = {name: _read_bytes(run_dir / name) for name in MANIFESTS}
    children = sorted(run_dir.iterdir())
    failed = False

    for child in children:
        # A snapshot is a FILE now -- the `.ckpt.zst` the trainer wrote. The
        # directory form is a rung published before the format changed, and the
        # node never produces one, so it is not uploaded: `migrate-checkpoints`
        # converted those, because converting needs zarr and this module is
        # imported before `uv sync`.
        if not child.is_file() or not is_snapshot(child.name):
            continue
        if not _put_object(f"{run_id}/{child.name}", child, sas, log):
            failed = True

    # Loose files -- .run.json, metrics.jsonl, progress.jsonl -- manifests and
    # snapshots excluded, both already placed above. Kilobytes, and rewritten
    # each tick, so they are PUT unconditionally rather than skipped on
    # existence the way an immutable rung is.
    for child in children:
        if (
            child.is_file()
            and child.name not in MANIFESTS
            and not is_snapshot(child.name)
            and not _put_object(f"{run_id}/{child.name}", child, sas, log, overwrite=True)
        ):
            failed = True

    if failed:
        # Reported, never swallowed, and the manifest is withheld: a publish
        # that silently fails every time turns "a killed task loses one rung"
        # into "a killed task loses everything".
        log(f"WARN publish incomplete for {run_id} -- manifest NOT updated, so the")
        log("     container still describes the last fully-published checkpoint.")
        return False

    for name, body in manifests.items():
        if body is None:
            continue
        if not _rungs_landed(body, run_id, sas, log):
            return False
        try:
            blobstore.put_bytes(sas, f"{run_id}/{name}", body)
        except Exception as error:  # noqa: BLE001 -- a publish must not kill a live task
            log(f"WARN could not publish {name}: {type(error).__name__}: {error}")
            return False
    log(f"published {run_id}")
    return True


def _put_object(name: str, source: Path, sas: str, log: Log, *, overwrite: bool = False) -> bool:
    """PUT one file unless the container already has it. False only on failure.

    A rung is immutable, so an existing object is the same bytes and the upload
    is skipped -- that skip is the whole reason a resumed task does not re-send
    a ladder. Metadata is rewritten as a run progresses, so it passes
    ``overwrite`` and always goes.

    AN EMPTY FILE IS NEVER PUBLISHED. It is the residue of a truncating write,
    and uploading it spreads the zeroing to every later fetch of the run:
    measured 08-23, two reference runs' records were zeroed under retrying
    evaluate tasks and a restored copy was re-zeroed within minutes by tasks
    holding poisoned fetches. Skipping is SUCCESS -- the store keeps what it
    has, and a rung that never lands still withholds the manifest naming it.
    """
    try:
        if source.stat().st_size == 0:
            log(f"skip publishing empty {source.name} (a record is never 0 bytes)")
            return True
        if not overwrite and blobstore.exists(sas, name):
            return True
        size = blobstore.put_object(sas, name, source)
    except Exception as error:  # noqa: BLE001 -- a publish must not kill a live task
        # LOUD, because the alternative is the failure shape this project keeps
        # paying for: a write that reports success and lands nowhere.
        log(f"WARN {name} NOT in the container: {type(error).__name__}: {error}")
        return False
    log(f"blob {name} <- {size:,} bytes")
    return True


def _rungs_landed(manifest: bytes, run_id: str, sas: str, log: Log) -> bool:
    """Is every rung this manifest names actually in the container?

    The manifest is the last thing published for exactly this check: it must
    never advertise a rung a fetch would then fail on. A rung the manifest names
    and the container lacks was PRUNED -- the ladder is deliberately not
    rewritten when a rung is dropped -- so this refuses to move the pointer
    rather than treating the gap as a publish failure.
    """
    missing = sorted(
        name
        for name in _named_rungs(manifest)
        if not blobstore.exists(sas, f"{run_id}/{records.object_name(name)}")
    )
    if missing:
        log(f"WARN manifest names {', '.join(missing)}, not in the container -- NOT publishing it")
        return False
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


def _megabytes(directory: Path) -> float:
    """Local disk, so the stats are free next to the copy they precede."""
    with contextlib.suppress(OSError):
        return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1e6
    return 0.0


def _read_bytes(path: Path) -> bytes | None:
    """A file's bytes, or None when it is absent, unreadable or empty."""
    try:
        body = path.read_bytes()
    except OSError:
        return None
    return body or None


def _parse_manifest(body: str) -> dict:
    """Parse a manifest's text, or ``{}`` if it is empty or torn."""
    try:
        parsed = json.loads(body) if body else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def fetch_snapshot(source: Path, destination: Path, name: str, sas: str = "") -> None:
    """Get one snapshot onto the node, replacing whatever is there.

    THE CONTAINER FIRST, the share as the fallback, and that order is the whole
    read flip: while both stores hold rungs this prefers the one that is a
    single request, and when the share stops holding them the fallback simply
    stops finding anything. `source.name` IS the run id -- the archive
    directory is named for it -- so nothing has to thread one.

    Remove first, and no update check. A cancelled task leaves partial rungs on
    the node, and treating those as already-present means the next task
    inherits a TRUNCATED checkpoint and dies inside zarr. That is what happened
    to rung 10000000: "fetched" in one second, then a read error. Node-local
    state is never evidence of a complete copy.
    """
    stored = records.object_name(name)
    destination.mkdir(parents=True, exist_ok=True)
    # BOTH SPELLINGS, not merely the one asked for. The container holds the
    # object and the share may still hold the directory it was converted from,
    # so a node that fetched under one name can be holding the other from a
    # cancelled task -- and a loader that finds the stale one loads a rung this
    # fetch did not fetch.
    for stale in {name, stored}:
        path = destination / stale
        shutil.rmtree(path, ignore_errors=True)
        path.unlink(missing_ok=True)
    if sas and blobstore.get_object(sas, f"{source.name}/{stored}", destination):
        return
    published = source / name
    if published.is_file():
        copy_file(published, destination / name)
        return
    copy_tree(published, destination / name, update=False)


class FetchRefusedError(Exception):
    """The store cannot supply what this task needs, and guessing would be worse.

    Raised rather than logged because every case is one where continuing means
    training or scoring against data that is absent, truncated, or written by a
    backend this tree cannot read -- each of which surfaces minutes later as a
    confusing error in a different subsystem.
    """


def require_complete(source: Path, name: str, sas: str = "") -> None:
    """A rung without its marker is either pre-marker or was interrupted.

    The two are indistinguishable from here, and loading a truncated one yields
    a corrupt-chunk error deep inside zarr several minutes later, so refuse.

    There is deliberately no repair path. One existed, was never once run, and
    could only ever have helped runs published before markers existed -- all of
    which are gone. A rung that lands unmarked now means a publish was cut off,
    and the answer to that is to publish it again from the node that has it,
    not to bless whatever reached the share.
    """
    # PRESENCE IS COMPLETENESS. One rung is one atomically-committed object:
    # it is either there whole or not there at all. The completion marker this
    # used to demand existed because a DIRECTORY on SMB could be half-copied
    # and look finished, and there are no directories left in any store.
    if sas and blobstore.exists(sas, f"{source.name}/{records.object_name(name)}"):
        return
    if not (source / name).exists():
        raise FetchRefusedError(f"the manifest names {name} but no store holds it")


def fetch_current_rung(source: Path, destination: Path, log: Log = _quiet, sas: str = "") -> str:
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
    body = published_manifest(source, sas)
    if (source / LEGACY_MANIFEST).is_file() and not body:
        raise FetchRefusedError(
            f"{source.name} was trained by the dynamic backend, which no longer "
            f"exists. Its checkpoints are unreadable at HEAD by design, so this "
            f"run cannot be continued."
        )
    manifest = _parse_manifest(body)
    if not manifest:
        # An absent manifest is not an error: a task that died before its first
        # checkpoint publishes .run.json and nothing else, and the right thing
        # is to start the ladder rather than refuse.
        log(f"no published checkpoint for {source.name}")
        return ""
    current = manifest.get("zarr") or ""
    if not current:
        raise FetchRefusedError(f"{records.STATIC_CHECKPOINT} names no current snapshot")
    require_complete(source, current, sas)
    fetch_snapshot(source, destination, current, sas)
    # The node's own copy comes from WHICHEVER STORE ANSWERED, not the mount.
    (destination / records.STATIC_CHECKPOINT).write_text(body, encoding="utf-8")
    log(f"fetched current rung {current}")
    return current


ABSTRACTIONS_CONTAINER = "abstractions"
DIAGNOSTICS_CONTAINER = "diagnostics"
ABSTRACTION_SUFFIX = ".tar.zst"


def abstraction_object(name: str) -> str:
    """The object one abstraction lives at. `name` is its directory name."""
    return f"{name}{ABSTRACTION_SUFFIX}"


def pack_abstraction(source: Path, destination: Path) -> int:
    """Tar+zstd one abstraction directory into a single object.

    ONE OBJECT, for the reason a rung is one object: an abstraction is ~10
    files and a half-uploaded tree is something a node might try to load, where
    a half-uploaded blob is simply absent.
    """
    import subprocess  # noqa: PLC0415 -- stdlib, and only when publishing

    # `tar --zstd` rather than a Python codec: the node closure is stdlib-only
    # and `zstandard` is not importable before `uv sync`.
    subprocess.run(
        ["tar", "--zstd", "-cf", str(destination), "-C", str(source.parent), source.name],
        check=True,
    )
    return destination.stat().st_size


def unpack_abstraction(archive_file: Path, destination: Path) -> None:
    """Extract one packed abstraction into `destination`."""
    import subprocess  # noqa: PLC0415 -- stdlib, and only when fetching

    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--zstd", "-xf", str(archive_file), "-C", str(destination)], check=True)


def fetch_abstractions(sas: str, destination: Path, log: Log = _quiet) -> int:
    """Bring every abstraction the container holds onto this node.

    MERGES, like the share copy it replaces: a node cannot know which
    abstraction a task will resolve against until the trainer reads its config,
    and there are ten of them. An abstraction already on the node is not
    re-fetched, so the steady-state cost is one HEAD each rather than 2.83 GiB.
    """
    fetched = 0
    for name in blobstore.list_container(sas):
        if not name.endswith(ABSTRACTION_SUFFIX):
            continue
        directory = destination / name.removesuffix(ABSTRACTION_SUFFIX)
        if directory.is_dir():
            continue
        packed = destination / name
        if not blobstore.get_object(sas, name, destination):
            log(f"  WARN {name} vanished from the container")
            continue
        try:
            unpack_abstraction(packed, destination)
            fetched += 1
            log(f"  fetched abstraction {directory.name}")
        finally:
            packed.unlink(missing_ok=True)
    return fetched


def published_manifest(source: Path, sas: str = "") -> str:
    """A run's manifest as TEXT, from the container first and the share second.

    The manifest lives beside the rungs it names -- `<run>/STATIC_CHECKPOINT
    .json` in the checkpoints container -- because it is the thing that says
    which of them is current, and a pointer stored apart from what it points at
    is the drift this migration spent a day undoing. The share answers only
    while it still holds one.

    `source.name` IS the run id, as everywhere else here: the archive directory
    is named for it, so nothing has to thread one.
    """
    if sas:
        body = blobstore.read_object(sas, f"{source.name}/{records.STATIC_CHECKPOINT}")
        if body is not None:
            return body.decode("utf-8")
    path = source / records.STATIC_CHECKPOINT
    return path.read_text() if path.is_file() else ""


def manifest_entries(source: Path, sas: str = "") -> list[tuple[int, str]]:
    """Every (iteration, snapshot name) the manifest CLAIMS, ascending.

    The claim is what a fetch resolves and what a migration has to reproduce.
    Listing the share's directories answers a different question and cannot see
    a rung whose bytes are gone: six runs hold 316 marked rungs with no
    directory at all, and a directory-driven check reported nothing to do.
    """
    manifest = _parse_manifest(published_manifest(source, sas))
    if not manifest:
        return []
    entries = [*manifest.get("retained", [])]
    if manifest.get("zarr"):
        entries.append({"iteration": manifest.get("iteration"), "zarr": manifest["zarr"]})
    claimed: dict[int, str] = {}
    for entry in entries:
        iteration, name = entry.get("iteration"), entry.get("zarr")
        if iteration is None or not name:
            continue
        claimed[int(iteration)] = str(name)
    return sorted(claimed.items())


def _ladder_names(source: Path, sas: str = "") -> dict[str, str]:
    """Iteration (as a string) -> the snapshot name the manifest gives it.

    Keyed on the string because that is what a `--at` flag carries; an int key
    would make every caller convert, and one of them would forget.
    """
    return {str(iteration): name for iteration, name in manifest_entries(source, sas)}


def fetch_for_evaluation(
    source: Path, destination: Path, rungs: Sequence[str], log: Log = _quiet, sas: str = ""
) -> list[str]:
    """Fetch only the rungs being scored. Returns the ones that arrived.

    Selective because the whole ladder is thirty ~540 MB rungs, ~16 GB, to
    score three of them. A rung that cannot be fetched is skipped and named
    rather than fatal: a partial curve beats none.

    THE MANIFEST NAMES THE RUNG, and this used to build `static-<rung>.zarr` by
    hand instead. That is a second opinion about a name the manifest already
    holds, and it survives only as long as every snapshot is a zarr directory:
    a run whose manifest was repointed to the new format would have every rung
    "missing" here while sitting on the share untouched.
    """
    fetch_metadata(source, destination)
    ladder = _ladder_names(source, sas)
    fetched = []
    for rung in rungs:
        name = ladder.get(str(rung), "")
        if not name:
            # Skipped rather than guessed. There is no name to try: the manifest
            # is what says a rung was published, so a rung it does not name has
            # no bytes to fetch under any spelling.
            log(f"  WARN rung {rung}: the manifest names no snapshot at that iteration")
            continue
        try:
            require_complete(source, name, sas)
        except FetchRefusedError as refusal:
            log(f"  WARN rung {rung}: {refusal}")
            continue
        try:
            fetch_snapshot(source, destination, name, sas)
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
