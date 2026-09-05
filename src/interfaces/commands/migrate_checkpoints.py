"""The `migrate-checkpoints` subcommand: move published rungs into the container.

RUNS ON A NODE, and that is the whole reason it is a task rather than something
you type. The rungs live on the SMB share, and a rung is ~4,200 zarr chunk
files -- roughly five million across the ~1,200 retained. Reading those from a
laptop in another country is not slow, it is impossible; a node has the same
share mounted inside the region.

IDEMPOTENT AND RESUMABLE, because it will not finish in one task. Every rung is
checked against the container before it is read, so a second run costs one HEAD
per rung and uploads only what is missing. That is also what makes it safe to
run while training continues: a rung republished underneath it is simply
already there.

`--drop-share` is the other end of the migration: once a rung is in the
container and its header opens, the zarr directory it was converted from is
duplicate weight. It deletes the DIRECTORY and keeps the MARKER -- the marker
is a zero-byte file and it is the share's index of what was ever published,
which `prune-checkpoints` and `verify_published_rungs` both read. A run left
with markers and no directories is exactly the shape the post-flip publish
already produces.

NOTHING IS DELETED WITHOUT `--apply`, and nothing is deleted that the container
cannot supply: every drop is gated on a HEAD plus a header read of the object
that replaces it.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared import records
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


# Enough for the JSON header of any rung this project writes: five arrays at
# ~120 bytes of spec each, plus the attrs. Measured largest is 812 bytes.
_HEADER_PROBE_BYTES = 16_384


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver migrate-checkpoints`."""
    parser.add_argument(
        "--share",
        default="",
        help="Share root holding archive/. Defaults to the node's mount.",
    )
    parser.add_argument(
        "--runs", nargs="*", default=None, help="Only these run ids (default: all published)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many rungs (0 = no limit). A task has a deadline.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Upload nothing; report which published rungs the container lacks.",
    )
    parser.add_argument(
        "--drop-share",
        action="store_true",
        help="Delete zarr directories the container already holds. Reports unless --apply.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --drop-share, actually delete. Without it nothing is removed.",
    )


class MigratedPayload(BaseModel):
    """What moved, and what is still on the share alone."""

    op: Literal["migrate-checkpoints"] = "migrate-checkpoints"
    runs_considered: int = 0
    rungs_uploaded: int = 0
    rungs_already_there: int = 0
    bytes_uploaded: int = 0
    failures: list[str] = Field(default_factory=list)
    stopped_early: bool = False
    verified: bool = False
    """Rungs the SHARE holds that the container does not.

    THE COMPLETENESS TEST, and it is deliberately not "every rung in
    `checkpoints`": the record claims a rung the moment a trainer commits it,
    and the retain policy drops some before any publish runs -- so the record
    names rungs no store has ever held. The share is what the container has to
    reproduce.
    """
    missing: list[str] = Field(default_factory=list)
    """Claimed rungs the container lacks and the share still holds, marked.
    These are migratable and a sweep will move them."""
    unmarked: list[str] = Field(default_factory=list)
    """Claimed rungs on the share WITHOUT a marker, which no sweep will move.

    Pre-marker or interrupted, and indistinguishable from here. They are not a
    migration failure -- they are snapshots nothing has ever been willing to
    load, since `require_complete` refuses them at fetch time too.
    """
    phantom: list[str] = Field(default_factory=list)
    """Claimed rungs with no bytes in either store.

    PRUNE MAKES THESE BY DESIGN and they are not data loss: dropping a rung
    deletes the snapshot without rewriting the manifest that advertises it, so
    a settled run's ladder names rungs it has not held for weeks -- measured
    1,030 of them against 3 that are genuinely gone. They block no deletion,
    because deleting the share cannot take away bytes that are not there. They
    are why this gate cannot treat the manifest as the authority on what
    EXISTS; it is only the authority on what a fetch will ask for.
    """
    unclaimed: list[str] = Field(default_factory=list)
    """Share directories NO manifest names. Nothing can resolve them, so they
    are not migrated -- and they are the one thing safe to delete outright."""
    dropped_mode: bool = False
    applied: bool = False
    rungs_dropped: int = 0
    """Zarr directories deleted from the share because the container holds them."""
    kept: list[str] = Field(default_factory=list)
    """Share directories NOT dropped, and why. Each is a rung the container
    cannot supply, so the share copy is still the only one."""
    unreadable: list[str] = Field(default_factory=list)
    """Objects that are present but whose header would not parse, or whose
    fingerprint is not the one the manifest claims."""
    runs_without_manifest: int = 0


def run(args: argparse.Namespace) -> MigratedPayload:
    """Walk the share's archive and upload every rung the container lacks."""
    import os  # noqa: PLC0415 -- node-only, and only when actually migrating
    import time  # noqa: PLC0415 -- see above
    from pathlib import Path  # noqa: PLC0415 -- see above

    sas = os.environ.get("POKER_SOLVER_CHECKPOINT_SAS", "")
    if not sas:
        raise CommandError(
            "No POKER_SOLVER_CHECKPOINT_SAS: there is nowhere to migrate to. "
            "This runs as a task, which is sealed with one at dispatch."
        )
    mounts = os.environ.get("AZ_BATCH_NODE_MOUNTS_DIR") or "/mnt/batch/tasks/fsmounts"
    root = Path(args.share or f"{mounts}/shared") / "archive"
    if not root.is_dir():
        raise CommandError(f"no archive directory at {root}")

    # Node-local scratch: staging is the point, so it must not be on the share.
    work = Path(os.environ.get("RUN_WORK_DIR") or "/mnt/work") / "migrate-staging"
    work.mkdir(parents=True, exist_ok=True)
    payload = MigratedPayload(
        verified=bool(args.verify),
        dropped_mode=bool(args.drop_share),
        applied=bool(args.drop_share and args.apply),
    )
    wanted = set(args.runs or [])
    # TIMED AND PRINTED AT EVERY STAGE, because the first version of this
    # printed only after a successful upload and told me nothing when it spent
    # thirty minutes reaching zero. Reading a share directory is latency-bound
    # and this project has lost hours to reasoning about that instead of
    # measuring it.
    started = time.monotonic()
    listed = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"listed {len(listed)} run(s) in {time.monotonic() - started:.1f}s", flush=True)
    for run_dir in listed:
        if wanted and run_dir.name not in wanted:
            continue
        payload.runs_considered += 1
        at = time.monotonic()
        # THE MANIFEST, not the directory listing. What has to reach the
        # container is what something can ask for, and only the manifest says
        # that -- a rung it does not name is unreachable however many bytes sit
        # beside it, and a rung it names with no bytes is invisible to a listing.
        claimed = [name for _iteration, name in archive.manifest_entries(run_dir)]
        on_share = set(_snapshots(run_dir))
        if not claimed:
            payload.runs_without_manifest += 1
        for orphan in sorted(on_share - set(claimed)):
            payload.unclaimed.append(f"{run_dir.name}/{orphan}")
        if claimed:
            print(
                f"{run_dir.name}: {len(claimed)} claimed rung(s), "
                f"{len(on_share - set(claimed))} unclaimed, "
                f"listed in {time.monotonic() - at:.1f}s",
                flush=True,
            )
        # DROPPING WALKS THE SHARE, not the manifest: an unclaimed directory is
        # duplicate weight too, and the manifest is exactly what cannot see it.
        targets = sorted(on_share) if args.drop_share else claimed
        for snapshot in targets:
            if args.limit and payload.rungs_uploaded + payload.rungs_dropped >= args.limit:
                payload.stopped_early = True
                return payload
            try:
                if args.drop_share:
                    _drop_one(args, payload, sas, run_dir, snapshot)
                else:
                    _one_rung(args, payload, sas, run_dir, snapshot, on_share, work)
            except Exception as error:  # noqa: BLE001 -- one bad rung must not end the sweep
                # THE MESSAGE, AND IMMEDIATELY. Recording only the exception
                # CLASS threw away the one thing that identifies the fault, and
                # holding it until `render` meant a task that hit its ceiling
                # first took every failure with it. An HTTP error also carries
                # the service's own explanation in its body, which is the
                # difference between "HTTPError" and knowing which header the
                # service objected to.
                detail = f"{type(error).__name__}: {error}"
                body = getattr(error, "read", None)
                if callable(body):
                    with contextlib.suppress(Exception):
                        detail += f" | {body().decode('utf-8', 'replace')[:400]}"
                print(f"  {snapshot}: FAILED {detail}", flush=True)
                payload.failures.append(f"{run_dir.name}/{snapshot}: {detail}")
    return payload


def _drop_one(
    args: argparse.Namespace,
    payload: MigratedPayload,
    sas: str,
    run_dir: Path,
    snapshot: str,
) -> None:
    """Delete one zarr directory, but only if the container can supply it.

    THE MARKER STAYS. It is zero bytes and it is the share's index of what was
    published: `prune-checkpoints` reads markers to know the ladder, and
    `verify_published_rungs` builds from them before it asks the container.
    Deleting it would make a rung the container holds unscoreable.
    """
    import shutil  # noqa: PLC0415 -- node-only

    from src.shared import records  # noqa: PLC0415 -- node-only
    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- node-only

    stored = records.object_name(snapshot)
    if not blobstore.exists(sas, run_dir.name, stored):
        print(f"  {snapshot}: KEPT, the container has no {stored}", flush=True)
        payload.kept.append(f"{run_dir.name}/{snapshot}: not in the container")
        return
    fault = _header_fault(sas, run_dir, stored)
    if fault:
        print(f"  {snapshot}: KEPT, {fault}", flush=True)
        payload.kept.append(f"{run_dir.name}/{snapshot}: {fault}")
        return
    if not args.apply:
        payload.rungs_dropped += 1
        return
    shutil.rmtree(run_dir / snapshot)
    payload.rungs_dropped += 1
    print(f"  {snapshot}: dropped", flush=True)


def _one_rung(
    args: argparse.Namespace,
    payload: MigratedPayload,
    sas: str,
    run_dir: Path,
    snapshot: str,
    on_share: set[str],
    work: Path,
) -> None:
    """Classify ONE claimed rung, and move it if that is what was asked."""
    import time  # noqa: PLC0415 -- node-only

    from src.shared import records  # noqa: PLC0415 -- see above
    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- node-only

    object_name = records.object_name(snapshot)
    at = time.monotonic()
    present = blobstore.exists(sas, run_dir.name, object_name)
    print(
        f"  {snapshot}: HEAD {time.monotonic() - at:.1f}s -> {'present' if present else 'absent'}",
        flush=True,
    )
    if present:
        if args.verify:
            fault = _header_fault(sas, run_dir, object_name)
            if fault:
                print(f"  {snapshot}: UNREADABLE {fault}", flush=True)
                payload.unreadable.append(f"{run_dir.name}/{snapshot}: {fault}")
                return
        payload.rungs_already_there += 1
        return

    if snapshot not in on_share:
        payload.phantom.append(f"{run_dir.name}/{snapshot}")
        return

    # THE SHARE'S MARKER GOVERNS what may be uploaded. A rung with no marker is
    # either pre-marker or was interrupted mid-publish, and the two are
    # indistinguishable from here -- so copying one in would launder a
    # possibly-partial snapshot into a store where existence MEANS complete.
    marker = run_dir / archive.marker_for(snapshot)
    if not marker.exists():
        # PRINTED, not merely counted. A skip used to produce no output at all,
        # so a sweep that skipped everything looked identical to one that was
        # working -- forty minutes of HEADs and no upload, with the summary
        # never reaching the log because the task hit its ceiling first.
        print(f"  {snapshot}: SKIP, no marker at {marker.name}", flush=True)
        payload.unmarked.append(f"{run_dir.name}/{snapshot}")
        return
    if args.verify:
        payload.missing.append(f"{run_dir.name}/{snapshot}")
        return
    print(f"  {snapshot}: converting...", flush=True)
    _name, uploaded = _upload(sas, run_dir, snapshot, work)
    payload.bytes_uploaded += uploaded
    payload.rungs_uploaded += 1


def _upload(sas: str, run_dir: Path, snapshot: str, work: Path) -> tuple[str, int]:
    """CONVERT one zarr rung to `.ckpt.zst` and upload it. Returns (name, bytes).

    Not a copy. The share holds zarr directories and the container holds the
    format that replaced them, so migrating means reading the arrays and
    re-encoding -- which is also why this runs here rather than in the node
    wrapper: `zarr` is a dependency the wrapper may not import.

    STAGED TO LOCAL DISK FIRST, and that is the whole cost. A legacy rung is
    ~5,500 chunk files and reading them one at a time over SMB took more than
    eight minutes; `archive.copy_tree` is the parallel copier the publish path
    already uses, and it turns that into ~8 seconds.
    """
    import shutil  # noqa: PLC0415 -- node-only
    import time  # noqa: PLC0415 -- node-only

    from src.engine.solver.storage import snapshot_format  # noqa: PLC0415 -- node-only
    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- node-only

    staged = work / snapshot
    shutil.rmtree(staged, ignore_errors=True)
    at = time.monotonic()
    copied = archive.copy_tree(run_dir / snapshot, staged, update=False)
    staged_in = time.monotonic() - at

    at = time.monotonic()
    arrays, attrs = _read_zarr(staged)
    converted = work / records.object_name(snapshot)
    size = snapshot_format.write_snapshot(converted, arrays, attrs)
    encoded_in = time.monotonic() - at
    del arrays

    at = time.monotonic()
    try:
        blobstore.put_rung(sas, run_dir.name, converted.name, converted)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
        converted.unlink(missing_ok=True)
    print(
        f"    staged {copied / 1024**2:.0f} MiB in {staged_in:.1f}s, "
        f"encoded to {size / 1024**2:.0f} MiB in {encoded_in:.1f}s, "
        f"uploaded in {time.monotonic() - at:.1f}s",
        flush=True,
    )
    return converted.name, size


def _header_fault(sas: str, run_dir: Path, object_name: str) -> str:
    """Why this object is not a loadable snapshot, or "" when it is.

    A HEAD says an object exists; it does not say the upload finished or that
    what landed is a snapshot. This is the check that gates deleting the other
    copy, so it opens the header -- a few hundred bytes -- and, where the
    manifest carries a fingerprint, insists the two agree.
    """
    from src.engine.solver.storage import snapshot_format  # noqa: PLC0415 -- node-only
    from src.shared import records  # noqa: PLC0415 -- see above
    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- node-only

    prefix = blobstore.read_head(sas, run_dir.name, object_name, _HEADER_PROBE_BYTES)
    if prefix is None:
        return "vanished between the HEAD and the read"
    try:
        length = int.from_bytes(prefix[: snapshot_format.HEADER_LENGTH_BYTES], "little")
        body = prefix[snapshot_format.HEADER_LENGTH_BYTES :][:length]
        if len(body) < length:
            return f"header claims {length} bytes, only {len(body)} arrived"
        header = json.loads(body)
    except (ValueError, IndexError) as error:
        return f"header would not parse: {error}"

    manifest = archive.read_manifest(run_dir / records.STATIC_CHECKPOINT)
    expected = manifest.get("fingerprint", "")
    current = records.object_name(str(manifest.get("zarr", "")))
    stored = (header.get("attrs") or {}).get("fingerprint", "")
    # Only the CURRENT rung: the manifest carries one fingerprint and it
    # describes that rung. A retained entry has none to compare against.
    if expected and object_name == current and stored and stored != expected:
        return f"fingerprint {stored} but the manifest claims {expected}"
    return ""


def _read_zarr(path: Path) -> tuple[dict, dict]:
    """A legacy rung's arrays and attrs. The only zarr read left in the sweep."""
    import zarr  # noqa: PLC0415 -- the format being migrated away from

    root = zarr.open(zarr.DirectoryStore(path), mode="r")
    return {name: root[name][:] for name in root.array_keys()}, dict(root.attrs)


def _snapshots(run_dir: Path) -> list[str]:
    return [c.name for c in run_dir.iterdir() if c.is_dir() and archive.is_snapshot(c.name)]


def render(payload: MigratedPayload) -> None:
    if payload.verified:
        _render_verification(payload)
        return
    if payload.dropped_mode:
        verb = "dropped" if payload.applied else "would drop"
        print(f"runs considered:  {payload.runs_considered:,}")
        print(f"zarr dirs {verb}: {payload.rungs_dropped:,}")
        print(f"KEPT (container cannot supply): {len(payload.kept):,}")
        for line in payload.kept[:20]:
            print(f"  kept  {line}")
        if len(payload.kept) > 20:
            print(f"  ... and {len(payload.kept) - 20:,} more kept")
        if payload.stopped_early:
            print("STOPPED EARLY at --limit; re-run to continue.")
        if not payload.applied:
            print("\nDRY RUN -- nothing was deleted. Re-run with --apply.")
        return
    print(f"runs considered:  {payload.runs_considered:,}")
    print(
        f"rungs uploaded:   {payload.rungs_uploaded:,}  ({payload.bytes_uploaded / 1024**3:.1f} GiB)"
    )
    print(f"already present:  {payload.rungs_already_there:,}")
    if payload.stopped_early:
        print("STOPPED EARLY at --limit; re-run to continue where this left off.")
    if payload.unmarked:
        print(f"skipped, unmarked: {len(payload.unmarked):,}  (a fetch refuses these too)")
    if payload.failures:
        print(f"\n{len(payload.failures)} rung(s) NOT migrated:")
        for line in payload.failures[:20]:
            print(f"  {line}")
    print("\nNothing was deleted. The share still holds every byte.")


def _render_verification(payload: MigratedPayload) -> None:
    """What the container is missing, split by what could be done about it.

    Every line here is a different ANSWER, not a different severity: a sweep
    fixes one and the others cannot be fixed at all. A verification that merged them into one count is what let a
    deletion gate report "nothing missing" while 316 claimed rungs had no share
    copy to migrate from.
    """
    print(f"runs considered:      {payload.runs_considered:,}")
    if payload.runs_without_manifest:
        print(f"runs with NO manifest: {payload.runs_without_manifest:,}  <- claim nothing")
    print(f"claimed and present:  {payload.rungs_already_there:,}")
    print(f"MISSING (marked):     {len(payload.missing):,}   <- a sweep will move these")
    print(f"unmarked:             {len(payload.unmarked):,}   <- no sweep will, ever")
    print(f"UNREADABLE object:    {len(payload.unreadable):,}   <- present but will not open")
    print(f"unclaimed on share:   {len(payload.unclaimed):,}   <- no manifest names them")
    print(f"phantom ladder rows:  {len(payload.phantom):,}   <- pruned; block nothing")
    for label, lines, cap in (
        ("missing", payload.missing, 20),
        ("UNREADABLE", payload.unreadable, 20),
        ("unmarked", payload.unmarked, 10),
        ("unclaimed", payload.unclaimed, 10),
    ):
        for line in lines[:cap]:
            print(f"  {label}  {line}")
        if len(lines) > cap:
            print(f"  ... and {len(lines) - cap:,} more {label}")

    print()
    blocks_share = payload.missing + payload.unreadable
    if blocks_share:
        print(f"SHARE: DO NOT DELETE -- {len(blocks_share):,} rung(s) it holds are not in the")
        print("       container, or are there and will not open.")
    else:
        print("SHARE: every rung it holds is in the container and opens.")


COMMAND = Command(
    name="migrate-checkpoints",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Upload published rungs from the share into the checkpoint container.",
)
