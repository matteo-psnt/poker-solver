"""The `pool-status` subcommand: every node, the last autoscale decision, and
why a resize failed."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command
from src.shared.task_states import NodePhase

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver pool-status`. It takes none."""


class PoolView(batch.PoolStatus):
    """One pool, plus what it costs while up.

    Subclasses the shape `batch.pool_status` already produces rather than
    restating its six fields -- what is added is the rate, which comes from
    config and not from Batch.
    """

    hourly_cost: str | None = None
    # Dollars per node-hour while up; null when the SKU has no measured rate.
    burn_per_hour: float | None = None
    # `hourly_cost` times the nodes allocated now.


class PoolPayload(BaseModel):
    """EVERY configured pool, not the one whose id happens to be `pool_id`.

    `train-big` ran live work that this command could not see at all, so a
    reader asking "what is the pool doing" was answered about half the account.
    """

    op: Literal["pool-status"] = "pool-status"
    pools: list[PoolView]
    total_nodes: int
    # Both summed across pools -- the console header counts one of each.
    burn_per_hour: float | None = None


def run(args: argparse.Namespace) -> PoolPayload:  # noqa: ARG001
    """Read every pool, its nodes, its last autoscale run and any resize errors.

    Concurrently: the reads are independent and the cost is the round trip.
    """
    config = CloudConfig.load()
    client = batch.client(config)
    rates = {config.pool_id: config.hourly_cost, config.pool_big_id: config.pool_big_hourly_cost}
    wanted = [pool_id for pool_id in (config.pool_id, config.pool_big_id) if pool_id]
    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        statuses = list(pool.map(lambda pool_id: batch.pool_status(client, pool_id), wanted))
    views = [
        PoolView(
            hourly_cost=rates.get(status.pool_id) or None,
            burn_per_hour=burn_per_hour(rates.get(status.pool_id), status.current_dedicated_nodes),
            **status.model_dump(),
        )
        for status in statuses
    ]
    burns = [view.burn_per_hour for view in views if view.burn_per_hour is not None]
    return PoolPayload(
        pools=views,
        total_nodes=sum(view.current_dedicated_nodes or 0 for view in views),
        burn_per_hour=round(sum(burns), 3) if burns else None,
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
    if not payload.pools:
        print("No pools. Check `terraform output` in infra/ -- the account has none deployed.")
        return
    for view in payload.pools:
        _render_pool(view)
    if len(payload.pools) <= 1:
        return
    burn = f" -> ${payload.burn_per_hour:.2f}/hr" if payload.burn_per_hour is not None else ""
    # Named, never silently summed over: a total that quietly drops a pool's
    # burn reads as the whole bill and is not one.
    unpriced = [view.pool_id for view in payload.pools if view.burn_per_hour is None]
    missing = f" (EXCLUDES {', '.join(unpriced)} — no rate for that SKU)" if unpriced else ""
    print(f"\nAll pools: {payload.total_nodes} node(s){burn}{missing}")


def _render_pool(payload: PoolView) -> None:
    print(f"Pool {payload.pool_id} ({payload.vm_size})")
    print(f"  state:   {payload.allocation_state}")
    print(f"  nodes:   {payload.current_dedicated_nodes} / {payload.target_dedicated_nodes}")
    burn = f" -> ${payload.burn_per_hour:.2f}/hr now" if payload.burn_per_hour is not None else ""
    rate = payload.hourly_cost or "rate unknown for this SKU"
    print(f"  cost:    {rate} (pool is 0 nodes at rest){burn}")
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
