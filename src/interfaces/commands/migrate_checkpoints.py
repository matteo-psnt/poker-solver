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


class MigratedPayload(BaseModel):
    """What moved, and what is still on the share alone."""

    op: Literal["migrate-checkpoints"] = "migrate-checkpoints"
    runs_considered: int = 0
    rungs_uploaded: int = 0
    rungs_already_there: int = 0
    bytes_uploaded: int = 0
    failures: list[str] = Field(default_factory=list)
    stopped_early: bool = False


def run(args: argparse.Namespace) -> MigratedPayload:
    """Walk the share's archive and upload every rung the container lacks."""
    import os  # noqa: PLC0415 -- node-only, and only when actually migrating
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

    payload = MigratedPayload()
    wanted = set(args.runs or [])
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted and run_dir.name not in wanted:
            continue
        payload.runs_considered += 1
        for snapshot in sorted(_snapshots(run_dir)):
            if args.limit and payload.rungs_uploaded >= args.limit:
                payload.stopped_early = True
                return payload
            try:
                if blobstore.exists(sas, run_dir.name, snapshot):
                    payload.rungs_already_there += 1
                    continue
                # THE SHARE'S MARKER STILL GOVERNS what may be uploaded. A rung
                # with no marker was interrupted mid-publish, and copying it
                # into the container would launder a partial snapshot into a
                # store where existence MEANS complete.
                if not (run_dir / archive.marker_for(snapshot)).exists():
                    payload.failures.append(f"{run_dir.name}/{snapshot}: no completion marker")
                    continue
                payload.bytes_uploaded += blobstore.put_rung(
                    sas, run_dir.name, snapshot, run_dir / snapshot
                )
            except Exception as error:  # noqa: BLE001 -- one bad rung must not end the sweep
                payload.failures.append(f"{run_dir.name}/{snapshot}: {type(error).__name__}")
                continue
            payload.rungs_uploaded += 1
            print(f"  {run_dir.name}/{snapshot}", flush=True)
    return payload


def _snapshots(run_dir: Path) -> list[str]:
    return [c.name for c in run_dir.iterdir() if c.is_dir() and archive.is_snapshot(c.name)]


def render(payload: MigratedPayload) -> None:
    print(f"runs considered:  {payload.runs_considered:,}")
    print(
        f"rungs uploaded:   {payload.rungs_uploaded:,}  ({payload.bytes_uploaded / 1024**3:.1f} GiB)"
    )
    print(f"already present:  {payload.rungs_already_there:,}")
    if payload.stopped_early:
        print("STOPPED EARLY at --limit; re-run to continue where this left off.")
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
