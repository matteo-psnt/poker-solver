"""The `autoscale-check` subcommand: evaluate the live formula, errors included.

The error half is the entire point. An invalid or throwing formula still
returns partial ``results``, so reporting the results alone makes a broken
formula look healthy -- which is how a ``#`` comment (Batch wants ``//``) and a
one-argument ``GetSample`` both went unnoticed while the pool quietly stopped
scaling up. Run this after every formula change; server-side evaluation is
free, safe and instant.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver autoscale-check`. It takes none."""


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: ARG001
    """Evaluate the deployed formula against the live pool."""
    config = CloudConfig.load()
    result = batch.evaluate_autoscale(
        batch.client(config), config.pool_id, config.autoscale_formula
    )
    return {"op": "autoscale-check", "pool_id": config.pool_id, **result}


def render(payload: dict[str, Any]) -> None:
    error = payload["error"]
    if error is None:
        print("  no error")
    else:
        print(f"  ERROR: {error['code']}")
        if error["message"]:
            print(f"    {error['message']}")
        for name, value in error["values"].items():
            print(f"    {name}: {value}")
        print("  (variables below are PARTIAL — the formula did not fully evaluate)")
    for variable in payload["variables"]:
        print(f"    {variable}")


COMMAND = Command(
    name="autoscale-check",
    help="Evaluate the deployed autoscale formula on the live pool, errors included.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
