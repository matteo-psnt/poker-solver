"""The `promote` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store.share import share_client
from src.interfaces.cloud.store.workspace import write_baseline
from src.interfaces.commands._base import (
    Command,
    records_root,
    resolve_run_dir,
)
from src.interfaces.errors import CommandError
from src.pipeline import services

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver promote`."""
    parser.add_argument("--run", required=True, help="Run id to promote. Must be published.")
    parser.add_argument(
        "--rationale",
        required=True,
        help="Why this run becomes the baseline. Required — a lineage that moved for "
        "an unrecorded reason cannot be audited later.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Point the baseline at a run, closing one turn of the base-fork loop.

    Publishing is no longer optional. A `--local-only` baseline was a pointer
    that existed on one laptop, which is precisely the failure the published
    baseline was introduced to fix -- and with no local runs left there is
    nothing for it to point AT that another machine could resolve.

    The pointer is still written to a temporary file first, because
    :func:`services.promote_baseline` composes the document there; it is the
    published copy that is the record.
    """
    with records_root(args) as root:
        run_dir = resolve_run_dir(args.run, str(root))
        local = root / "baseline.json"
        baseline = services.promote_baseline(
            run_dir.name,
            args.rationale,
            path=local,
            checkpoint_iteration=services.checkpoint_iteration_of(run_dir),
        )
        refused = _publish(local)
    if refused is not None:
        raise CommandError(
            "The baseline was composed but NOT published, so nothing has moved. "
            "The share is the only copy now that local runs are gone — check "
            f"`az login` and Terraform state, then run this again.\n  Cause: {refused}"
        )
    return {"op": "promote", **dataclasses.asdict(baseline)}


def _publish(local: Path) -> str | None:
    """Copy the pointer to the share. Returns None on success, else why it failed.

    The baseline is the conclusion of every experiment -- which run the next one
    forks from -- and it was the only artifact that never left the machine that
    wrote it.

    Why the reason is returned rather than discarded:
        This used to answer a bare ``False``, so the caller's advice ("check
        `az login` and Terraform state") was a GUESS at a cause it had thrown
        away -- and unreachable-share, bad-credential and no-such-container all
        arrived looking identical. The handler still refuses rather than raising
        the SDK's own error, because a promotion is not the place to meet a
        forty-line Azure traceback.
    """
    try:
        config = CloudConfig.load()
        write_baseline(share_client(config), config.share_name, local.read_text())
    except Exception as error:  # noqa: BLE001 -- the reason is returned to the caller, not discarded
        return f"{type(error).__name__}: {error}"
    return None


def render(payload: dict[str, Any]) -> None:
    print(f"Baseline is now {payload['run_id']}")
    if payload["checkpoint_iteration"] is not None:
        print(f"  Checkpoint:  {payload['checkpoint_iteration']:,}")
    print(f"  Rationale:   {payload['rationale']}")
    print(f"  Recorded in: {payload['baseline']}")
    print("  Published:   yes — the share carries it, so a fresh checkout finds it")


COMMAND = Command(
    name="promote",
    help="Make a run the new baseline (closes one turn of the base-fork loop).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
