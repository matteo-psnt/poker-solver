"""The `migrate-checkpoints` subcommand: does the container hold every rung?

WHAT IS LEFT OF THE MIGRATION. Moving the rungs is done -- 1,269 objects in the
container, zero zarr directories on the share -- so the conversion and the
share-drop that performed it are gone with the bytes they moved. This is the
half worth keeping: the integrity check that gated the deletion, and the one
question worth asking again whenever a run's ladder matters.

RUNS ON A NODE. The manifests are on the SMB share and there are 343 of them;
the node has the same share mounted inside the region.

IT ANSWERS AGAINST THE STORES, not the record. The manifest is what a fetch
resolves, so it says which rungs must be reachable -- but it OVER-CLAIMS by
design: `prune-checkpoints` drops a snapshot without rewriting the ladder that
advertises it, and 1,030 claimed rungs have no bytes anywhere as a result.
Present is not readable either: an upload that died mid-stream still answers a
HEAD, so every present rung has its header read as well.
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


class MigratedPayload(BaseModel):
    """What the container holds, and what it is missing."""

    op: Literal["migrate-checkpoints"] = "migrate-checkpoints"
    runs_considered: int = 0
    runs_without_manifest: int = 0
    rungs_present: int = 0
    failures: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    """Claimed rungs the container lacks while the share still holds them,
    marked. Nothing produces these any more; one would mean a publish that
    reached the share and not the container."""
    unmarked: list[str] = Field(default_factory=list)
    """Claimed rungs on the share WITHOUT a marker. Pre-marker or interrupted,
    and indistinguishable from here -- `require_complete` refuses them too."""
    phantom: list[str] = Field(default_factory=list)
    """Claimed rungs with no bytes anywhere.

    PRUNE MAKES THESE BY DESIGN and they are not loss: dropping a rung deletes
    the snapshot without rewriting the manifest that advertises it, measured at
    1,030 against 3 genuinely gone. They are why this cannot treat the manifest
    as the authority on what EXISTS -- only on what a fetch will ask for.
    """
    unclaimed: list[str] = Field(default_factory=list)
    """Share directories no manifest names. Nothing can resolve them."""
    unreadable: list[str] = Field(default_factory=list)
    """Objects that are present but whose header will not parse, or whose
    fingerprint is not the one the manifest claims."""


def run(args: argparse.Namespace) -> MigratedPayload:
    """Walk every published manifest and check its rungs against the container."""
    import os  # noqa: PLC0415 -- node-only
    import time  # noqa: PLC0415 -- see above
    from pathlib import Path  # noqa: PLC0415 -- see above

    sas = os.environ.get("POKER_SOLVER_CHECKPOINT_SAS", "")
    if not sas:
        raise CommandError(
            "No POKER_SOLVER_CHECKPOINT_SAS: there is nothing to check against. "
            "This runs as a task, which is sealed with one at dispatch."
        )
    mounts = os.environ.get("AZ_BATCH_NODE_MOUNTS_DIR") or "/mnt/batch/tasks/fsmounts"
    root = Path(args.share or f"{mounts}/shared") / "archive"
    if not root.is_dir():
        raise CommandError(f"no archive directory at {root}")

    payload = MigratedPayload()
    # SPLIT ON WHITESPACE. A shell that fails to word-split `--runs $ids` hands
    # over ONE space-joined value; no run id contains a space, so the intent is
    # unambiguous. Taking it literally filtered on a run that cannot exist:
    # 29 tasks queued, considered 0 runs, and exited 0 having done nothing.
    wanted = {name for value in (args.runs or []) for name in value.split()}
    started = time.monotonic()
    listed = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"listed {len(listed)} run(s) in {time.monotonic() - started:.1f}s", flush=True)

    for run_dir in listed:
        if wanted and run_dir.name not in wanted:
            continue
        payload.runs_considered += 1
        # THE MANIFEST, not the directory listing. What must be reachable is
        # what something can ask for, and only the manifest says that.
        claimed = [name for _iteration, name in archive.manifest_entries(run_dir)]
        on_share = set(_snapshots(run_dir))
        if not claimed:
            payload.runs_without_manifest += 1
        for orphan in sorted(on_share - set(claimed)):
            payload.unclaimed.append(f"{run_dir.name}/{orphan}")
        for snapshot in claimed:
            try:
                _one_rung(payload, sas, run_dir, snapshot, on_share)
            except Exception as error:  # noqa: BLE001 -- one bad rung must not end the sweep
                # THE MESSAGE, AND IMMEDIATELY. Recording only the exception
                # CLASS threw away the one thing that identifies the fault, and
                # an HTTP error carries the service's own explanation in its
                # body -- the difference between "HTTPError" and knowing which
                # header the service objected to.
                detail = f"{type(error).__name__}: {error}"
                body = getattr(error, "read", None)
                if callable(body):
                    with contextlib.suppress(Exception):
                        detail += f" | {body().decode('utf-8', 'replace')[:400]}"
                print(f"  {snapshot}: FAILED {detail}", flush=True)
                payload.failures.append(f"{run_dir.name}/{snapshot}: {detail}")
    return payload


def _one_rung(
    payload: MigratedPayload,
    sas: str,
    run_dir: Path,
    snapshot: str,
    on_share: set[str],
) -> None:
    """Classify ONE claimed rung by what could be done about it."""
    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- node-only

    object_name = records.object_name(snapshot)
    if blobstore.exists(sas, run_dir.name, object_name):
        fault = _header_fault(sas, run_dir, object_name)
        if fault:
            print(f"  {snapshot}: UNREADABLE {fault}", flush=True)
            payload.unreadable.append(f"{run_dir.name}/{snapshot}: {fault}")
            return
        payload.rungs_present += 1
        return

    if snapshot not in on_share:
        payload.phantom.append(f"{run_dir.name}/{snapshot}")
        return
    if not (run_dir / archive.marker_for(snapshot)).exists():
        payload.unmarked.append(f"{run_dir.name}/{snapshot}")
        return
    payload.missing.append(f"{run_dir.name}/{snapshot}")


def _header_fault(sas: str, run_dir: Path, object_name: str) -> str:
    """Why this object is not a loadable snapshot, or "" when it is.

    A HEAD says an object exists; it does not say the upload finished or that
    what landed is a snapshot. So this opens the header -- a few hundred bytes
    -- and, where the manifest carries a fingerprint, insists the two agree.
    """
    from src.engine.solver.storage import snapshot_format  # noqa: PLC0415 -- node-only
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


def _snapshots(run_dir: Path) -> list[str]:
    return [c.name for c in run_dir.iterdir() if c.is_dir() and archive.is_snapshot(c.name)]


def render(payload: MigratedPayload) -> None:
    """What the container is missing, split by what could be done about it.

    Every line is a different ANSWER, not a different severity. A verification
    that merged them is what let a deletion gate report "nothing missing" while
    316 claimed rungs had no share copy to migrate from.
    """
    print(f"runs considered:      {payload.runs_considered:,}")
    if payload.runs_without_manifest:
        print(f"runs with NO manifest: {payload.runs_without_manifest:,}  <- claim nothing")
    print(f"claimed and present:  {payload.rungs_present:,}")
    print(f"MISSING (marked):     {len(payload.missing):,}   <- on the share, not in the container")
    print(f"unmarked:             {len(payload.unmarked):,}   <- nothing will ever move these")
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
    if payload.failures:
        print(f"\n{len(payload.failures)} rung(s) could not be checked:")
        for line in payload.failures[:20]:
            print(f"  {line}")

    print()
    broken = payload.missing + payload.unreadable
    if broken:
        print(f"UNSOUND: {len(broken):,} rung(s) a manifest claims are not in the container,")
        print("         or are there and will not open.")
        return
    print("Every rung any manifest claims is in the container and opens.")


COMMAND = Command(
    name="migrate-checkpoints",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Check that the container holds every rung the manifests claim.",
)
