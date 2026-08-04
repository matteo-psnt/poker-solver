"""The `repair-ladder` subcommand: prove a published ladder, rung by rung.

Rungs published before completion markers covered ``static-*`` are refused by
the fetch, because an unmarked rung and an interrupted one are
indistinguishable. Blanket-marking them would reinstate exactly the bug the
markers prevent, and deleting a multi-hour run whose data is mostly fine is
absurd. So each rung is PROVEN instead: copied, loaded, and marked only if zarr
can decompress every chunk. A rung that fails is left unmarked and named --
corrupt data is then discovered once, here, rather than minutes deep into a
scoring leg.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.cli.commands._base import Command
from src.interfaces.cloud import dispatch, spec


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver repair-ladder`."""
    parser.add_argument("--run", required=True, help="Published run id whose ladder to verify.")
    parser.add_argument(
        "--config", required=True, help="Training config stem the run was trained with."
    )
    parser.add_argument(
        "--timeout", default=spec.DEFAULT_TIMEOUT, help="Wall-clock ceiling on the verifier."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Stage the tree and queue one ladder-verification task."""
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.LegSpec(
                code_snapshot=snapshot,
                op=spec.REPAIR_LADDER,
                run_id=args.run,
                config=args.config,
                timeout=args.timeout,
            )
        ]
    )
    return {"op": "repair-ladder", "run_id": args.run, **payload}


def render(payload: dict[str, Any]) -> None:
    print(f"Verifying the published ladder for {payload['run_id']}.")
    dispatch.render_queued(payload)


COMMAND = Command(
    name="repair-ladder",
    help="Verify a published static ladder on a node, marking the rungs that load.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
