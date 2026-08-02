"""The `submit-precompute` subcommand: build a card abstraction on a node.

The cloud-first counterpart to the local ``precompute``, and the reason the
local one is no longer the only door. The invariant that made precompute look
local-only is *computed once, never recomputed* -- not *computed on a laptop*.
A node that builds an abstraction and publishes it once satisfies it exactly as
well, and does it on 16 cores instead of a laptop's.

``push-data`` is what remains of the old workflow: a seeding path for getting an
existing local copy onto a fresh share, not something to run routinely.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands._base import Command
from src.interfaces.cloud import dispatch, share, spec
from src.interfaces.cloud.config import CloudConfig
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.paths import abstraction_output_path

PRECOMPUTE_TIMEOUT = "12h"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run submit-precompute`."""
    parser.add_argument(
        "--config", required=True, help="Abstraction config stem (e.g. ochs_gate_ochs)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count on the node. Default is all CPUs.",
    )
    parser.add_argument(
        "--timeout",
        default=PRECOMPUTE_TIMEOUT,
        help="Wall-clock ceiling on the precompute process. Longer than a training "
        "leg's default: an exact-runout abstraction enumerates every canonical "
        "board, which is the dominant cost and is independent of bucket count.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Publish even if an abstraction of this name already exists. Almost "
        "never right: bucket assignment is not pinned by the abstraction hash, so "
        "replacing a published copy silently invalidates the provenance of every "
        "run trained against it.",
    )


def _published(config: CloudConfig) -> list[str]:
    """Abstraction directories already on the share."""
    service = share.share_client(config)
    return [
        entry.name
        for entry in share.list_entries(service, config.share_name, share.ABSTRACTION_DIR)
        if entry.is_directory
    ]


def _target_name(abstraction_config: str) -> str:
    """The directory this config will produce, derived without computing it.

    The name is a pure function of the config -- bucket counts, runout mode and
    a hash of the rest -- so the collision that the node would refuse can be
    caught here instead, before a snapshot is uploaded and a node allocated.
    """
    return abstraction_output_path(Path(), PrecomputeConfig.from_yaml(abstraction_config)).name


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Stage the tree and queue one precompute leg."""
    config = CloudConfig.load()
    existing = _published(config)
    target = _target_name(args.config)

    if target in existing and not args.force:
        raise SystemExit(
            f"'{target}' is already published, and republishing it would silently "
            "change bucket assignment under an unchanged abstraction hash -- "
            "invalidating the provenance of every run trained against it.\n"
            "Nothing was uploaded and no node was allocated. Pass --force only if "
            "no such run matters."
        )

    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.LegSpec(
                code_snapshot=snapshot,
                op=spec.PRECOMPUTE,
                config=args.config,
                workers=args.workers,
                timeout=args.timeout,
                force_publish=args.force,
            )
        ]
    )
    return {
        "op": "submit-precompute",
        "abstraction_config": args.config,
        "target_name": target,
        "already_published": existing,
        "force": args.force,
        **payload,
    }


def render(payload: dict[str, Any]) -> None:
    print(f"Precomputing abstraction '{payload['abstraction_config']}' on the pool.")
    print(f"  will publish as: {payload['target_name']}")
    if payload["force"]:
        print("  --force: an existing copy of this abstraction WILL be replaced.")
    else:
        print("  Not yet on the share; the node also refuses to overwrite.")
    dispatch.render_queued(payload)


COMMAND = Command(
    name="submit-precompute",
    help="Build a card abstraction on a node and publish it to the share.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
