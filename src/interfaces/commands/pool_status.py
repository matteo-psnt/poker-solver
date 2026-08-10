"""The `pool-status` subcommand: node counts, and why a resize failed."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver pool-status`. It takes none."""


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: ARG001
    """Read the pool's allocation state and any resize errors."""
    config = CloudConfig.load()
    status = batch.pool_status(batch.client(config), config.pool_id)
    return {"op": "pool-status", "hourly_cost": config.hourly_cost, **status}


def _print_values(values: dict[str, str | None]) -> None:
    """Print a resize error's values, pretty-printing the escaped JSON ones.

    Batch nests the real cause as a JSON *string* inside a value, so printing
    the value raw gives a single unreadable line. Anything that parses as JSON
    is re-indented; anything that does not is printed as-is, which is what
    keeps an unfamiliar value visible rather than swallowed.
    """
    for name, value in values.items():
        if value is None:
            continue
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            print(f"      {name}: {value}")
            continue
        rendered = json.dumps(parsed, indent=2).replace("\n", "\n      ")
        print(f"      {name}:\n      {rendered}")


def render(payload: dict[str, Any]) -> None:
    print(f"Pool {payload['pool_id']} ({payload['vm_size']})")
    print(f"  state:   {payload['allocation_state']}")
    print(f"  nodes:   {payload['current_dedicated_nodes']} / {payload['target_dedicated_nodes']}")
    print(f"  cost:    {payload['hourly_cost']} (pool is 0 nodes at rest)")
    if not payload["resize_errors"]:
        return
    print("\n  RESIZE ERRORS — Batch reports every allocation problem as a generic")
    print("  AllocationFailed; the real cause is in the values below.")
    for error in payload["resize_errors"]:
        print(f"    code: {error['code']}")
        if error["message"]:
            print(f"    {error['message']}")
        _print_values(error["values"])


COMMAND = Command(
    name="pool-status",
    help="Pool node counts, and the real cause behind any allocation failure.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
