"""The `pool-status` subcommand: every node, the last autoscale decision, and
why a resize failed."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command
from src.shared.task_states import NodePhase

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver pool-status`. It takes none."""


class PoolPayload(batch.PoolStatus):
    """What `pool-status` answers: the pool, plus what it costs while up.

    Subclasses the shape `batch.pool_status` already produces rather than
    restating its six fields -- the command adds the op tag and the rate, which
    come from config and not from Batch.
    """

    op: Literal["pool-status"] = "pool-status"
    """Dollars per node-hour while the pool is up. It is 0 nodes at rest."""
    hourly_cost: str | None = None
    """`hourly_cost` times the nodes allocated now. Null when the rate is unreadable."""
    burn_per_hour: float | None = None


def run(args: argparse.Namespace) -> PoolPayload:  # noqa: ARG001
    """Read the pool, its nodes, its last autoscale run and any resize errors."""
    config = CloudConfig.load()
    status = batch.pool_status(batch.client(config), config.pool_id)
    return PoolPayload(
        hourly_cost=config.hourly_cost,
        burn_per_hour=burn_per_hour(config.hourly_cost, status.current_dedicated_nodes),
        **status.model_dump(),
    )


def burn_per_hour(hourly_cost: str | None, nodes: int | None) -> float | None:
    """The number out of `$0.688/hr/node`, times the nodes -- or nothing."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", hourly_cost or "")
    if match is None or nodes is None:
        return None
    return round(float(match.group(1)) * nodes, 3)


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


def render(payload: PoolPayload) -> None:
    print(f"Pool {payload.pool_id} ({payload.vm_size})")
    print(f"  state:   {payload.allocation_state}")
    print(f"  nodes:   {payload.current_dedicated_nodes} / {payload.target_dedicated_nodes}")
    burn = f" -> ${payload.burn_per_hour:.2f}/hr now" if payload.burn_per_hour is not None else ""
    print(f"  cost:    {payload.hourly_cost} (pool is 0 nodes at rest){burn}")
    if payload.autoscale is not None:
        wants = payload.autoscale.variables.get("$TargetDedicatedNodes", "?")
        print(f"  autoscale: wants {wants} nodes, evaluated {payload.autoscale.evaluated_at}")
        if payload.autoscale.error is not None:
            print(f"    ERROR: {payload.autoscale.error.code} {payload.autoscale.error.message}")
    for node in payload.nodes:
        running = ", ".join(node.tasks) if node.tasks else ""
        detail = running or ("" if node.phase is NodePhase.IDLE else f"since {node.since}")
        flag = "  !! " + "; ".join(node.errors) if node.errors else ""
        print(f"    {node.phase:<8} {node.id[-12:]}  {detail}{flag}")
    if not payload.resize_errors:
        return
    print("\n  RESIZE ERRORS — Batch reports every allocation problem as a generic")
    print("  AllocationFailed; the real cause is in the values below.")
    for error in payload.resize_errors:
        print(f"    code: {error.code}")
        if error.message:
            print(f"    {error.message}")
        _print_values(error.values)


COMMAND = Command(
    name="pool-status",
    help="Pool node counts, and the real cause behind any allocation failure.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
