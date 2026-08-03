"""The `push-data` subcommand: publish card abstractions to the share.

**Copied, never recomputed on a node.** A recompute can change bucket
assignments without changing ``card_abstraction_hash``, so the provenance check
that guards every evaluation would pass over silently different buckets. The
abstraction is computed once, here, and this uploads the single authoritative
copy -- which is also why ``precompute`` remains a local operation while
everything else that costs CPU moved to the pool.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands._base import Command
from src.interfaces.cloud import share
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.errors import CommandError


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run push-data`."""
    parser.add_argument(
        "--source",
        default="data/combo_abstraction",
        help="Local abstractions directory to publish.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Publish only this abstraction directory (default: all of them).",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Upload each abstraction directory to the share."""
    config = CloudConfig.load()
    service = share.share_client(config)
    source = Path(args.source)
    if not source.is_dir():
        raise CommandError(f"No abstractions at {source}. Run `precompute` first.")

    directories = sorted(entry for entry in source.iterdir() if entry.is_dir())
    if args.name:
        directories = [entry for entry in directories if entry.name == args.name]
        if not directories:
            raise CommandError(f"No abstraction named '{args.name}' under {source}.")

    uploaded: dict[str, int] = {}
    for directory in directories:
        remote_root = f"{share.ABSTRACTION_DIR}/{directory.name}"
        share.ensure_directory(service, config.share_name, remote_root)
        count = 0
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                relative = path.relative_to(directory).as_posix()
                share.upload_file(service, config.share_name, f"{remote_root}/{relative}", path)
                count += 1
        uploaded[directory.name] = count

    return {"op": "push-data", "uploaded": uploaded}


def render(payload: dict[str, Any]) -> None:
    if not payload["uploaded"]:
        print("Nothing to upload.")
        return
    for name, count in payload["uploaded"].items():
        print(f"  {name}: {count} file(s)")
    print("  abstractions published — this copy is now the authoritative one")


COMMAND = Command(
    name="push-data",
    help="Publish card abstractions to the share (copied, never recomputed on a node).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
