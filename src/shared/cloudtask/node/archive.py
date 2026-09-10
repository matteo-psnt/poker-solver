"""Moving a run between the node's data disk and the containers.

Three rules, each a production failure rather than a preference, each argued at
the code that honours it:

* publish mid-run -- the node's disk dies with the task
* manifest LAST -- a manifest that lands before the rung it names advertises a
  checkpoint nothing can fetch, which every training tick used to do
* presence is completeness -- one rung is one atomically-committed object, so
  there is no half-written state and no completion marker to keep in step

The share was the fourth: it could not preserve timestamps and reported that
only AFTER copying, so every copy had to stay :func:`shutil.copyfile` rather
than :func:`shutil.copytree`. Nothing is published there now, and what is left
of ``copy_tree`` serves the FETCH direction, onto the node's own disk.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from src.shared import records
from src.shared.cloudtask.node import blobstore

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

Log = Callable[[str], None]


def _quiet(message: str) -> None:
    """Default sink, so every function here is callable from a test."""


def is_snapshot(name: str) -> bool:
    return name.startswith(SNAPSHOT_PREFIXES)


def marker_for(snapshot: str) -> str:
    return MARKER_PREFIX + snapshot


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


def fetch_metadata(run_id: str, destination: Path, sas: str) -> None:
    """Everything that is not a snapshot: the manifest, `.run.json`, the curve.

    Symmetric with what `publish_run` writes, rather than the manifest alone --
    fetching only the manifest would strand the rest under a name the publish
    had already chosen.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if not sas:
        return
    for name in blobstore.list_container(sas, f"{run_id}/"):
        leaf = name.partition("/")[2]
        if not leaf or leaf.startswith(MARKER_PREFIX) or is_snapshot(leaf):
            continue
        body = blobstore.read_object(sas, name)
        if body is None:
            continue
        target = destination / leaf
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)


def fetch_snapshot(run_id: str, destination: Path, name: str, sas: str) -> None:
    """Get one snapshot onto the node, replacing whatever is there.

    Remove first, and no update check. A cancelled task leaves partial rungs on
    the node, and treating those as already-present means the next task
    inherits a TRUNCATED checkpoint and dies in the loader. That is what
    happened to rung 10000000: "fetched" in one second, then a read error.
    Node-local state is never evidence of a complete copy.
    """
    stored = records.object_name(name)
    destination.mkdir(parents=True, exist_ok=True)
    # BOTH SPELLINGS, not merely the one asked for. A node that fetched under
    # one name can be holding the other from a cancelled task, and a loader
    # that finds the stale one loads a rung this fetch did not fetch.
    for stale in {name, stored}:
        path = destination / stale
        shutil.rmtree(path, ignore_errors=True)
        path.unlink(missing_ok=True)
    if not blobstore.get_object(sas, f"{run_id}/{stored}", destination):
        raise FetchRefusedError(f"the container does not hold {run_id}/{stored}")


class FetchRefusedError(Exception):
    """The store cannot supply what this task needs, and guessing would be worse.

    Raised rather than logged because every case is one where continuing means
    training or scoring against data that is absent, truncated, or written by a
    backend this tree cannot read -- each of which surfaces minutes later as a
    confusing error in a different subsystem.
    """


def require_complete(run_id: str, name: str, sas: str) -> None:
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
    if not blobstore.exists(sas, f"{run_id}/{records.object_name(name)}"):
        raise FetchRefusedError(f"the manifest names {name} but the container does not hold it")


def fetch_current_rung(run_id: str, destination: Path, sas: str, log: Log = _quiet) -> str:
    """Fetch the one rung the manifest calls current. Returns its name, or "".

    What both continuing a run and scoring "the latest checkpoint" need, and in
    both cases ONE rung, not the ladder. Taking the whole retained ladder was 31
    rungs, ~25 GB over SMB and ~40 minutes, to load the 809 MB actually read.

    Leaving the older rungs in the container loses nothing: ``_extend_ladder``
    builds the next manifest from the PREVIOUS manifest rather than from what
    is on disk, and a rung is skipped on existence -- so rungs this node never
    had are neither re-uploaded nor removed.
    """
    fetch_metadata(run_id, destination, sas)
    body = published_manifest(run_id, sas)
    # Only when there is no static manifest, so the common path pays nothing:
    # a legacy manifest ALONE means the dynamic backend, whose checkpoints are
    # unreadable at HEAD by design.
    if not body and blobstore.read_object(sas, f"{run_id}/{LEGACY_MANIFEST}") is not None:
        raise FetchRefusedError(
            f"{run_id} was trained by the dynamic backend, which no longer "
            f"exists. Its checkpoints are unreadable at HEAD by design, so this "
            f"run cannot be continued."
        )
    manifest = _parse_manifest(body)
    if not manifest:
        # An absent manifest is not an error: a task that died before its first
        # checkpoint publishes .run.json and nothing else, and the right thing
        # is to start the ladder rather than refuse.
        log(f"no published checkpoint for {run_id}")
        return ""
    current = manifest.get("zarr") or ""
    if not current:
        raise FetchRefusedError(f"{records.STATIC_CHECKPOINT} names no current snapshot")
    require_complete(run_id, current, sas)
    fetch_snapshot(run_id, destination, current, sas)
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


def published_manifest(run_id: str, sas: str) -> str:
    """A run's manifest as TEXT, or "" when the container holds none.

    The manifest lives beside the rungs it names -- `<run>/STATIC_CHECKPOINT
    .json` -- because it is the thing that says which of them is current, and a
    pointer stored apart from what it points at is the drift this migration
    spent a day undoing.

    NO CREDENTIAL IS "NOTHING PUBLISHED", not an error: a task sealed without
    one cannot see the store at all, and every caller here already treats an
    absent manifest as a run with nothing to fetch. Without this the empty SAS
    builds `/<run>/STATIC_CHECKPOINT.json` and urllib raises `unknown url type`.
    """
    if not sas:
        return ""
    body = blobstore.read_object(sas, f"{run_id}/{records.STATIC_CHECKPOINT}")
    return body.decode("utf-8") if body is not None else ""


def is_published(run_id: str, sas: str) -> bool:
    """Does this run exist in a store a fetch can reach?

    ASK THE MANIFEST, NOT A STORE FOR A DIRECTORY. The directory check this
    replaced gated four node paths, and the quiet one was the worst -- a RESUME
    skipped its fetch instead of failing, so the trainer started from zero and
    republished a ladder whose pointer no longer described the run.
    """
    return bool(published_manifest(run_id, sas))


def manifest_entries(run_id: str, sas: str) -> list[tuple[int, str]]:
    """Every (iteration, snapshot name) the manifest CLAIMS, ascending.

    The claim is what a fetch resolves. LISTING a store answers a different
    question and cannot see a rung whose bytes are gone: six runs once held 316
    marked rungs with no directory at all, and a listing-driven check reported
    nothing to do.
    """
    return ladder_entries(_parse_manifest(published_manifest(run_id, sas)))


def local_entries(run_dir: Path) -> list[tuple[int, str]]:
    """The same claim, read from the copy a fetch already put on this node.

    Split from :func:`manifest_entries` when that one started taking a run id:
    one of its two callers was asking about a LOCAL directory, and a store
    reader and a disk reader that share a signature are one refactor away from
    silently asking the wrong one.
    """
    return ladder_entries(read_manifest(run_dir / records.STATIC_CHECKPOINT))


def ladder_entries(manifest: dict) -> list[tuple[int, str]]:
    """Current plus retained, deduplicated by iteration and sorted."""
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


def _ladder_names(run_id: str, sas: str) -> dict[str, str]:
    """Iteration (as a string) -> the snapshot name the manifest gives it.

    Keyed on the string because that is what a `--at` flag carries; an int key
    would make every caller convert, and one of them would forget.
    """
    return {str(iteration): name for iteration, name in manifest_entries(run_id, sas)}


def fetch_for_evaluation(
    run_id: str, destination: Path, rungs: Sequence[str], sas: str, log: Log = _quiet
) -> list[str]:
    """Fetch only the rungs being scored. Returns the ones that arrived.

    Selective because the whole ladder is thirty ~540 MB rungs, ~16 GB, to
    score three of them. A rung that cannot be fetched is skipped and named
    rather than fatal: a partial curve beats none.

    THE MANIFEST NAMES THE RUNG, and this used to build `static-<rung>.zarr` by
    hand instead. That is a second opinion about a name the manifest already
    holds, and it survives only as long as every snapshot is a zarr directory.
    """
    fetch_metadata(run_id, destination, sas)
    ladder = _ladder_names(run_id, sas)
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
            require_complete(run_id, name, sas)
        except FetchRefusedError as refusal:
            log(f"  WARN rung {rung}: {refusal}")
            continue
        try:
            fetch_snapshot(run_id, destination, name, sas)
        except (OSError, FetchRefusedError) as error:
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
