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
    vcpus_per_node: int | None = None
    # Parsed from the SKU name. Pool sizes diverged (D16/D32/D64), so a node
    # count alone no longer says how much compute is up.
    vcpus: int | None = None
    # `vcpus_per_node` times the nodes allocated now.
    max_nodes: int | None = None
    # The deployed formula's own `maxNodes` -- the pool's real ceiling, which
    # variables.tf can drift from (it has, 30 vs a live 60).
    max_vcpus: int | None = None
    # `vcpus_per_node` times `max_nodes`: the pool's capacity in vCPUs.


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
    total_vcpus: int = 0
    # Live vCPUs across pools, the cross-pool number that stayed comparable.
    max_vcpus: int | None = None
    # Sum of the pools' ceilings; null unless EVERY pool reported one, so a
    # partial sum can never understate capacity.


def run(args: argparse.Namespace) -> PoolPayload:  # noqa: ARG001
    """Read every pool, its nodes, its last autoscale run and any resize errors.

    Concurrently: the reads are independent and the cost is the round trip.
    """
    config = CloudConfig.load()
    client = batch.client(config)
    rates = {
        config.pool_id: config.hourly_cost,
        config.pool_big_id: config.pool_big_hourly_cost,
        config.pool_huge_id: config.pool_huge_hourly_cost,
    }
    wanted = [
        pool_id for pool_id in (config.pool_id, config.pool_big_id, config.pool_huge_id) if pool_id
    ]
    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        statuses = list(pool.map(lambda pool_id: batch.pool_status(client, pool_id), wanted))
    views = [_view(status, rates.get(status.pool_id)) for status in statuses]
    burns = [view.burn_per_hour for view in views if view.burn_per_hour is not None]
    ceilings = [view.max_vcpus for view in views]
    return PoolPayload(
        pools=views,
        total_nodes=sum(view.current_dedicated_nodes or 0 for view in views),
        total_vcpus=sum(view.vcpus or 0 for view in views),
        max_vcpus=sum(ceilings) if views and None not in ceilings else None,  # type: ignore[arg-type]
        burn_per_hour=round(sum(burns), 3) if burns else None,
    )


def _view(status: batch.PoolStatus, rate: str | None) -> PoolView:
    """One pool's view: the Batch status plus rate and vCPU arithmetic."""
    per_node = vcpus_per_node(status.vm_size)
    ceiling = _max_nodes(status)
    return PoolView(
        hourly_cost=rate or None,
        burn_per_hour=burn_per_hour(rate, status.current_dedicated_nodes),
        vcpus_per_node=per_node,
        vcpus=(
            per_node * status.current_dedicated_nodes
            if per_node is not None and status.current_dedicated_nodes is not None
            else None
        ),
        max_nodes=ceiling,
        max_vcpus=per_node * ceiling if per_node is not None and ceiling is not None else None,
        **status.model_dump(),
    )


def vcpus_per_node(vm_size: str | None) -> int | None:
    """`standard_d32als_v6` -> 32; None for a SKU whose D-number is not its cores."""
    match = re.search(r"_d(\d+)", (vm_size or "").lower())
    return int(match.group(1)) if match else None


def _max_nodes(status: batch.PoolStatus) -> int | None:
    """The formula's own `maxNodes` assignment -- Batch echoes every variable."""
    if status.autoscale is None:
        return None
    value = status.autoscale.variables.get("maxNodes", "")
    return int(value) if value.isdigit() else None


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
    capacity = f" of {payload.max_vcpus}" if payload.max_vcpus is not None else ""
    print(
        f"\nAll pools: {payload.total_nodes} node(s), "
        f"{payload.total_vcpus}{capacity} vCPU{burn}{missing}"
    )


def _render_pool(payload: PoolView) -> None:
    size = payload.vm_size or "?"
    if payload.vcpus_per_node is not None:
        size = f"{size}, {payload.vcpus_per_node} vCPU/node"
    print(f"Pool {payload.pool_id} ({size})")
    print(f"  state:   {payload.allocation_state}")
    cap = f" (cap {payload.max_nodes})" if payload.max_nodes is not None else ""
    vcpus = f" = {payload.vcpus} vCPU" if payload.vcpus is not None else ""
    print(
        f"  nodes:   {payload.current_dedicated_nodes} / "
        f"{payload.target_dedicated_nodes}{cap}{vcpus}"
    )
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
