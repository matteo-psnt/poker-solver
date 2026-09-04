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

IT DELETES NOTHING. The share keeps every byte until someone looks at the
verification and decides otherwise.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


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
    """Rungs the share holds with a completion marker, that the container
    lacks. These are migratable and a sweep will move them."""
    unmarked: list[str] = Field(default_factory=list)
    """Rungs the share holds WITHOUT a marker, which no sweep will move.

    Pre-marker or interrupted, and indistinguishable from here. They are not a
    migration failure -- they are snapshots nothing has ever been willing to
    load, since `require_complete` refuses them at fetch time too.
    """


def run(args: argparse.Namespace) -> MigratedPayload:
    """Walk the share's archive and upload every rung the container lacks."""
    import os  # noqa: PLC0415 -- node-only, and only when actually migrating
    import time  # noqa: PLC0415 -- see above
    from pathlib import Path  # noqa: PLC0415 -- see above

    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- see above

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
    payload = MigratedPayload(verified=bool(args.verify))
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
        snapshots = sorted(_snapshots(run_dir))
        if snapshots:
            print(
                f"{run_dir.name}: {len(snapshots)} rung(s), listed in {time.monotonic() - at:.1f}s",
                flush=True,
            )
        for snapshot in snapshots:
            if args.limit and payload.rungs_uploaded >= args.limit:
                payload.stopped_early = True
                return payload
            try:
                at = time.monotonic()
                present = blobstore.exists(sas, run_dir.name, snapshot)
                print(
                    f"  {snapshot}: HEAD {time.monotonic() - at:.1f}s -> "
                    f"{'present' if present else 'absent'}",
                    flush=True,
                )
                if present:
                    payload.rungs_already_there += 1
                    continue
                # THE SHARE'S MARKER GOVERNS what may be uploaded. A rung with
                # no marker is either pre-marker or was interrupted mid-publish,
                # and the two are indistinguishable from here -- so copying one
                # in would launder a possibly-partial snapshot into a store
                # where existence MEANS complete.
                #
                # Classified BEFORE the verify branch, because "missing" and
                # "can never be migrated as it stands" are different answers and
                # a count that merges them cannot gate a deletion.
                marker = run_dir / archive.marker_for(snapshot)
                if not marker.exists():
                    # PRINTED, not merely counted. A skip used to produce no
                    # output at all, so a sweep that skipped everything looked
                    # identical to one that was working -- forty minutes of
                    # HEADs and no upload, with the summary never reaching the
                    # log because the task hit its ceiling first.
                    print(f"  {snapshot}: SKIP, no marker at {marker.name}", flush=True)
                    payload.unmarked.append(f"{run_dir.name}/{snapshot}")
                    continue
                if args.verify:
                    payload.missing.append(f"{run_dir.name}/{snapshot}")
                    continue
                print(f"  {snapshot}: uploading...", flush=True)
                payload.bytes_uploaded += _upload(sas, run_dir, snapshot, work)
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
                continue
            payload.rungs_uploaded += 1
    return payload


def _upload(sas: str, run_dir: Path, snapshot: str, work: Path) -> int:
    """Stage the rung to LOCAL DISK, then tar and upload it from there.

    THE WHOLE COST IS PER-FILE LATENCY, not bytes. A rung is ~4,200 zarr chunk
    files and `tarfile.add` walks them SERIALLY: over SMB that is minutes per
    rung, and the first sweep spent its entire 30-minute budget without
    finishing one. Across ~1,200 rungs it is five million round trips.

    `archive.copy_tree` already solves this -- it is the parallel copier the
    publish path uses, with up to 64 workers -- so the fix is to pay the SMB
    cost once, concurrently, and let tar read a local disk.
    """
    import shutil  # noqa: PLC0415 -- node-only
    import time  # noqa: PLC0415 -- node-only

    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- node-only

    staged = work / snapshot
    shutil.rmtree(staged, ignore_errors=True)
    at = time.monotonic()
    copied = archive.copy_tree(run_dir / snapshot, staged, update=False)
    fetched = time.monotonic() - at
    print(f"    staged {copied / 1024**2:.0f} MiB in {fetched:.1f}s", flush=True)
    at = time.monotonic()
    try:
        size = blobstore.put_rung(sas, run_dir.name, snapshot, staged)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    print(
        f"  {snapshot}: staged {copied / 1024**2:.0f} MiB in {fetched:.1f}s, "
        f"uploaded {size / 1024**2:.0f} MiB in {time.monotonic() - at:.1f}s",
        flush=True,
    )
    return size


def _snapshots(run_dir: Path) -> list[str]:
    return [c.name for c in run_dir.iterdir() if c.is_dir() and archive.is_snapshot(c.name)]


def render(payload: MigratedPayload) -> None:
    if payload.verified:
        print(f"runs considered:  {payload.runs_considered:,}")
        print(f"already in blob:  {payload.rungs_already_there:,}")
        print(f"MISSING (marked): {len(payload.missing):,}   <- a sweep will move these")
        print(f"unmarked:         {len(payload.unmarked):,}   <- no sweep will, ever")
        for line in payload.missing[:20]:
            print(f"  missing  {line}")
        if len(payload.missing) > 20:
            print(f"  ... and {len(payload.missing) - 20:,} more")
        for line in payload.unmarked[:10]:
            print(f"  unmarked {line}")
        if len(payload.unmarked) > 10:
            print(f"  ... and {len(payload.unmarked) - 10:,} more")
        if not payload.missing:
            print("\nEvery MARKED rung on the share is in the container.")
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


COMMAND = Command(
    name="migrate-checkpoints",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Upload published rungs from the share into the checkpoint container.",
)
