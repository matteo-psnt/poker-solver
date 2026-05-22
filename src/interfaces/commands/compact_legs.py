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

Deleting is a separate decision from bundling
---------------------------------------------
The default is a dry run: it says what WOULD move. ``--apply`` writes the bundle
and verifies it. Only then, and only with ``--delete``, are the loose files
removed -- and the local backup written first is the thing that makes that
reversible, because nothing else on this machine holds a copy of the record.
"""

from __future__ import annotations

import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import share
from src.interfaces.commands._base import Command
from src.interfaces.commands.tasks import download_tasks
from src.interfaces.errors import CommandError
from src.shared import records
from src.shared.cloudtask import task_log

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

        before = task_log.read_tasks(local)
        movable, names = task_log.compactable(directory)
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
        }
        if not names:
            return result
        if not args.apply:
            return result

        if args.backup:
            result["backup"] = _back_up(directory, Path(args.backup))

        remote = f"{task_log.RECORDS_DIRNAME}/{result['bundle']}"
        bundle_path = directory / result["bundle"]
        records.write_snapshot(
            bundle_path,
            task_log.bundle_document(movable),
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
            after = task_log.read_tasks(check)
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
        print(
            f"Would bundle {payload['movable']} of {payload['files_before']} file(s) "
            f"into {payload['bundle']}, covering {payload['attempts']} attempt(s)."
        )
        print("  Nothing has changed. Re-run with --apply to write the bundle.")
        return

    print(f"Wrote {payload['bundle']} holding {payload['movable']} document(s).")
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
