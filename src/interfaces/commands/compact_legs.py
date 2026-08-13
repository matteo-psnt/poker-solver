"""The `compact-legs` subcommand: many sealed task records into one file.

``legs/`` is the durable per-task account, and `tasks` answers by joining ALL of
it. Measured on 2026-08-10: 375 files totalling 0.18 MB. The bytes are nothing;
the round trips are the whole cost, and they are paid by `tasks`, by `cost`
(which invokes it), and so by the console's landing page.

What makes this safe rather than clever
---------------------------------------
The bundle stores documents VERBATIM, keyed by the filename they had. Reading is
already indifferent to which container a document is in
(:func:`task_log.read_documents`), so bundling is a change of container and
provably nothing else -- which is what lets this command verify itself by
comparing the joined rows before and after, rather than by trusting an argument.

Only SEALED attempts move. An attempt with no terminal record is one that
reconciliation still writes a ``<task>.observed.json`` for, and that is a
filename rather than a bundle entry.

A bundle is per LABEL, not per compaction
-----------------------------------------
The second round at a given label ABSORBS the bundle already there. It has to:
the documents that bundle holds are no longer loose, so ``compactable`` does not
offer them, and a payload built from its answer alone would replace them. The
default label is ``sealed``, so this is the ordinary path -- the verification
below caught the resulting loss, but only after the share had been overwritten.

And the bundle records the act. Nothing else does: this command can remove
hundreds of files from the share, which is the only copy of the task record, and
the sole trace used to be a backup directory on whichever laptop ran it.

Deleting is a separate decision from bundling
---------------------------------------------
The default is a dry run: it says what WOULD move. ``--apply`` writes the bundle
and verifies it. Only then, and only with ``--delete``, are the loose files
removed -- and the local backup written first is the thing that makes that
reversible, because nothing else on this machine holds a copy of the record.
"""

from __future__ import annotations

import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import share
from src.interfaces.commands._base import Command
from src.interfaces.commands.tasks import download_tasks
from src.interfaces.errors import CommandError
from src.shared import gitinfo, records, task_history
from src.shared.cloudtask import task_log

if TYPE_CHECKING:
    import argparse

_PARALLEL_SHARE_IO = 64


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver compact-legs`."""
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the bundle. Without this, only report what would move.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="After the bundle verifies, remove the loose files it replaced. "
        "This is the irreversible half and the only one that buys the speedup.",
    )
    parser.add_argument(
        "--backup",
        default="",
        metavar="DIR",
        help="Where to copy every leg file before anything is deleted. "
        "Required with --delete; nothing else holds a copy of the record.",
    )
    parser.add_argument(
        "--label",
        default="sealed",
        help="Names the bundle file: <label>.bundle.json.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Bundle the sealed leg records, verifying the join is unchanged."""
    if args.delete and not args.apply:
        raise CommandError("--delete needs --apply: there is nothing to delete until it verifies.")
    if args.delete and not args.backup:
        raise CommandError(
            "--delete needs --backup DIR. The share is the only copy of the task "
            "record, so a local one has to exist before any of it is removed."
        )

    config = CloudConfig.load()
    service = share.share_client(config)

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp)
        download_tasks(service, config.share_name, local)
        directory = task_log.tasks_dir(local)

        before = task_history.read_tasks(local)
        movable, names = task_history.compactable(directory)
        result: dict[str, Any] = {
            "op": "compact-legs",
            "bundle": f"{args.label}{task_log.BUNDLE_SUFFIX}",
            "files_before": len(list(directory.glob("*.json"))),
            "movable": len(names),
            "attempts": len(before),
            "applied": False,
            "verified": False,
            "deleted": 0,
            "backup": "",
            # How many records an existing bundle at this name already held, and
            # this run therefore carries forward rather than replaces.
            "carried": 0,
        }
        # The bundle already at this name, if any. Passing it is what stops a
        # second compaction from overwriting the first one's records -- the
        # documents it absorbed are no longer loose, so `compactable` does not
        # offer them and they exist only here.
        #
        # Read before the early returns so the DRY RUN can say so too. The
        # preview is where someone decides whether to apply, and a second round
        # that reported only its own 54 documents against a bundle holding 321
        # would look exactly like the bug this replaced.
        bundle_path = directory / result["bundle"]
        previous = records.read_snapshot(bundle_path) if bundle_path.exists() else None
        result["carried"] = len((previous or {}).get("records", {}))

        if not names:
            return result
        if not args.apply:
            return result

        if args.backup:
            result["backup"] = _back_up(directory, Path(args.backup))

        remote = f"{task_log.RECORDS_DIRNAME}/{result['bundle']}"
        records.write_snapshot(
            bundle_path,
            task_history.bundle_document(
                movable, previous=previous, compaction=_provenance(args, result)
            ),
            records.REGISTRY[f"legs/*{task_log.BUNDLE_SUFFIX}"],
        )
        share.write_text(service, config.share_name, remote, bundle_path.read_text())
        result["applied"] = True

        # Verified against a FRESH download, not the tree just written: the
        # question is whether the share now answers the same, and only the share
        # can be asked that.
        with tempfile.TemporaryDirectory() as check_tmp:
            check = Path(check_tmp)
            download_tasks(service, config.share_name, check)
            after = task_history.read_tasks(check)
        if after != before:
            raise CommandError(
                "The bundle landed but the joined task log CHANGED, so nothing was "
                f"deleted ({len(before)} rows before, {len(after)} after). The loose "
                "files are all still there; delete the bundle to undo."
            )
        result["verified"] = True

        if args.delete:
            result["deleted"] = _delete(service, config.share_name, names)
            result["files_after"] = result["files_before"] - result["deleted"] + 1
        return result


def _provenance(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    """What this compaction was, recorded into the bundle it writes.

    Written BEFORE the delete, which is the only order available: the bundle has
    to land and then be verified against a fresh download before anything may be
    removed, and rewriting it afterwards would invalidate the verification it
    just passed. So ``deleting`` is an intent rather than an outcome -- it says
    what was about to happen and, crucially, WHERE THE COPY WENT, which is the
    fact someone reconstructing the record actually needs. How many files
    survived is a property of the directory, and `tasks` can already be asked.

    The branch is here for the reason worktree provenance is on every other
    record: several checkouts run in parallel and a commit does not identify
    one.
    """
    return {
        "at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "label": args.label,
        "bundled": result["movable"],
        "carried": result["carried"],
        "backup": result["backup"],
        "deleting": bool(args.delete),
        "git_branch": gitinfo.get_git_branch() or "",
    }


def _back_up(directory: Path, destination: Path) -> str:
    """Copy every leg file somewhere durable. Returns where."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(directory.glob("*.json")):
        (destination / path.name).write_bytes(path.read_bytes())
    return str(destination.resolve())


def _delete(service: Any, share_name: str, names: list[str]) -> int:
    """Remove the loose files the bundle now holds. Returns how many existed."""

    def remove(name: str) -> bool:
        return share.delete_file(service, share_name, f"{task_log.RECORDS_DIRNAME}/{name}")

    with ThreadPoolExecutor(max_workers=min(_PARALLEL_SHARE_IO, len(names))) as pool:
        return sum(pool.map(remove, names))


def render(payload: dict[str, Any]) -> None:
    if not payload["movable"]:
        print(f"Nothing to compact: {payload['attempts']} attempt(s), none sealed and loose.")
        return
    if not payload["applied"]:
        carried = payload.get("carried", 0)
        print(
            f"Would bundle {payload['movable']} of {payload['files_before']} file(s) "
            f"into {payload['bundle']}, covering {payload['attempts']} attempt(s)."
        )
        if carried:
            print(f"  {payload['bundle']} already holds {carried}; they are carried forward.")
        print("  Nothing has changed. Re-run with --apply to write the bundle.")
        return

    # What the bundle HOLDS, not what this round moved. A second compaction at
    # the same label absorbs the first, and reporting only the new documents
    # would read as though the earlier ones had been dropped -- which is exactly
    # what used to happen.
    carried = payload.get("carried", 0)
    held = payload["movable"] + carried
    print(
        f"Wrote {payload['bundle']} holding {held} document(s)"
        + (f" ({payload['movable']} new, {carried} carried forward)." if carried else ".")
    )
    print(
        "  verified: the joined task log is row-identical"
        if payload["verified"]
        else "  NOT verified"
    )
    if payload["backup"]:
        print(f"  backup:   {payload['backup']}")
    if payload["deleted"]:
        print(
            f"  deleted:  {payload['deleted']} loose file(s) — legs/ is now ~{payload['files_after']}"
        )
    else:
        print("  kept:     every loose file. Add --delete to remove them.")


COMMAND = Command(
    name="compact-legs",
    help="Bundle sealed task records into one file, so reading legs/ is one round trip.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
