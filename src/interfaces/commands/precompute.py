"""The `precompute` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import (
    Command,
)
from src.pipeline import services

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver precompute`."""
    parser.add_argument(
        "--config", required=True, help="Abstraction config stem (e.g. production)."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Parallel workers (default: config value)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if a complete abstraction exists."
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Write street completion here as it goes. Nothing reaches the output "
        "directory until the build succeeds, so this is the only way a caller can "
        "see the work advance.",
    )


class PrecomputePayload(BaseModel):
    """Where a built abstraction landed. NODE-ONLY: no endpoint serves this."""

    op: Literal["precompute"] = "precompute"
    abstraction_config: str
    output_dir: str


def run(args: argparse.Namespace) -> PrecomputePayload:
    """Precompute a combo abstraction into ``data/combo_abstraction/<name>``."""
    out = services.precompute_abstraction(
        args.config,
        num_workers=args.workers,
        overwrite=args.overwrite,
        progress_file=Path(args.progress_file) if args.progress_file else None,
    )
    return PrecomputePayload(abstraction_config=args.config, output_dir=str(out))


def render(payload: PrecomputePayload) -> None:
    print("Precompute complete.")
    print(f"  Abstraction: {payload.abstraction_config}")
    print(f"  Output:      {payload.output_dir}")


COMMAND = Command(
    name="precompute",
    help="Precompute a combo abstraction into data/combo_abstraction/.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
