"""The `fetch` subcommand: bring the published record back, and rebuild the index.

Defaults to METADATA ONLY, which is a deliberate reversal of what the shell
recipe did. Every local consumer -- ``ledger``, ``report``, ``compare``,
``curve``, ``promote`` -- reads nothing but small JSON, and the local runs
directory is already a few megabytes of ``evals/``. The old blanket download
pulled whole zarr checkpoints (~540 MB each) to feed them, which is why
fetching was something you scheduled rather than something you did.

``--full`` and ``--run`` exist for the rare case that genuinely needs a
checkpoint, and both obey the manifest: only what ``CHECKPOINT.json`` names is
pulled. That rule previously lived only on the node, so the local fetch could
pull the partially-copied snapshot directories a killed task leaves behind.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from azure.storage.fileshare import ShareServiceClient

from src.interfaces.cli.commands._base import Command
from src.interfaces.cloud import share
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.errors import CommandError
from src.pipeline.evaluation import ledger as eval_ledger
from src.shared.config import DEFAULT_RUNS_DIR


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run fetch`."""
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also pull checkpoint data, not just the JSON record. Large.",
    )
    parser.add_argument("--run", default="", help="Fetch only this run id.")
    parser.add_argument(
        "--runs-dir", default=DEFAULT_RUNS_DIR, help="Local destination for published runs."
    )
    parser.add_argument(
        "--ledger", default=str(eval_ledger.DEFAULT_LEDGER_PATH), help="Eval ledger path."
    )


def _fetch_run(
    service: ShareServiceClient,
    share_name: str,
    run_name: str,
    destination: Path,
    *,
    full: bool,
) -> tuple[int, int]:
    """Pull one published run. Returns (files, skipped-because-unnamed)."""
    run_path = f"{share.ARCHIVE_DIR}/{run_name}"
    members = share.manifest_members(service, share_name, run_path) if full else None

    fetched = 0
    skipped = 0
    for remote in share.walk_files(service, share_name, run_path):
        relative = remote[len(f"{share.ARCHIVE_DIR}/") :]
        parts = Path(relative).parts

        if share.is_snapshot_path(relative):
            if not full:
                continue
            # A snapshot directory the manifest does not name is unfinished by
            # construction, whatever the listing shows.
            snapshot_dir = next(part for part in parts if part.endswith(".zarr"))
            if members is not None and snapshot_dir not in members:
                skipped += 1
                continue
        elif not full and not share.is_metadata(parts[-1]):
            continue

        share.download_file(service, share_name, remote, destination / relative)
        fetched += 1
    return fetched, skipped


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Download published runs, then rebuild the ledger from what landed."""
    config = CloudConfig.load()
    service = share.share_client(config)
    destination = Path(args.runs_dir)

    published = [
        entry.name
        for entry in share.list_entries(service, config.share_name, share.ARCHIVE_DIR)
        if entry.is_directory
    ]
    wanted = [args.run] if args.run else published
    if args.run and args.run not in published:
        raise CommandError(f"'{args.run}' is not published. Published runs: {', '.join(published)}")

    total = 0
    skipped = 0
    for run_name in wanted:
        fetched, unnamed = _fetch_run(
            service, config.share_name, run_name, destination, full=args.full
        )
        total += fetched
        skipped += unnamed

    rebuilt, preserved = eval_ledger.rebuild_ledger(destination, Path(args.ledger))
    return {
        "op": "fetch",
        "runs": wanted,
        "files": total,
        "skipped_unnamed": skipped,
        "mode": "full" if args.full else "metadata",
        "ledger_rows": rebuilt,
        "ledger_preserved": preserved,
    }


def render(payload: dict[str, Any]) -> None:
    print(
        f"Fetched {payload['files']} file(s) from {len(payload['runs'])} run(s) [{payload['mode']}]"
    )
    if payload["skipped_unnamed"]:
        print(
            f"  {payload['skipped_unnamed']} snapshot file(s) skipped — not named by "
            "CHECKPOINT.json, so unfinished by construction."
        )
    if payload["mode"] == "metadata":
        print("  JSON only. Use --full (or --run <id> --full) to pull checkpoint data.")
    print(f"  Ledger: {payload['ledger_rows']} row(s), {payload['ledger_preserved']} preserved.")


COMMAND = Command(
    name="fetch",
    help="Bring published runs back (JSON by default, --full for checkpoints).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
