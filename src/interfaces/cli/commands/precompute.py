"""The `precompute` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.cli.commands._base import (
    Command,
)
from src.pipeline import services


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run precompute`."""
    parser.add_argument(
        "--config", required=True, help="Abstraction config stem (e.g. production)."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Parallel workers (default: config value)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if a complete abstraction exists."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Precompute a combo abstraction into ``data/combo_abstraction/<name>``."""
    out = services.precompute_abstraction(
        args.config,
        num_workers=args.workers,
        overwrite=args.overwrite,
    )
    return {
        "op": "precompute",
        "abstraction_config": args.config,
        "output_dir": str(out),
    }


def render(payload: dict[str, Any]) -> None:
    print("Precompute complete.")
    print(f"  Abstraction: {payload['abstraction_config']}")
    print(f"  Output:      {payload['output_dir']}")


COMMAND = Command(
    name="precompute",
    help="Precompute a combo abstraction into data/combo_abstraction/.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
