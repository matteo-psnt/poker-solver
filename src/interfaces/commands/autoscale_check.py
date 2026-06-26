"""The `autoscale-check` subcommand: evaluate the live formula, errors included.

The error half is the entire point. An invalid or throwing formula still
returns partial ``results``, so reporting the results alone makes a broken
formula look healthy -- which is how a ``#`` comment (Batch wants ``//``) and a
one-argument ``GetSample`` both went unnoticed while the pool quietly stopped
scaling up. Run this after every formula change; server-side evaluation is
free, safe and instant.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver autoscale-check`. It takes none."""


class AutoscaleView(batch.AutoscaleResult):
    """One pool's formula, evaluated against that pool.

    `error` is a FIELD, not a failed request: Batch evaluates the formula and
    reports that it did not compute, which is the answer to "why is the pool not
    growing" and must reach the screen rather than blanking the panel.
    """

    pool_id: str


class AutoscalePayload(BaseModel):
    """EVERY pool's formula. They are separate formulas on separate pools, so
    checking one and reporting "no error" answers about half the account."""

    op: Literal["autoscale-check"] = "autoscale-check"
    results: list[AutoscaleView]


def run(args: argparse.Namespace) -> AutoscalePayload:  # noqa: ARG001
    """Evaluate each deployed formula against its own pool, concurrently."""
    config = CloudConfig.load()
    client = batch.client(config)
    # ALL pools, same set pool-status walks — checking a subset is the exact
    # half-blind failure the payload docstring warns about (train-big was once
    # missing here; train-huge repeated it).
    wanted = [
        pool_id for pool_id in (config.pool_id, config.pool_big_id, config.pool_huge_id) if pool_id
    ]
    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        evaluated = list(
            pool.map(lambda pool_id: batch.evaluate_autoscale(client, pool_id), wanted)
        )
    return AutoscalePayload(
        results=[
            AutoscaleView(pool_id=pool_id, **result.model_dump())
            for pool_id, result in zip(wanted, evaluated, strict=True)
        ]
    )


def render(payload: AutoscalePayload) -> None:
    for view in payload.results:
        print(f"Pool {view.pool_id}")
        if view.error is None:
            print("  no error")
        else:
            print(f"  ERROR: {view.error.code}")
            if view.error.message:
                print(f"    {view.error.message}")
            for name, value in view.error.values.items():
                print(f"    {name}: {value}")
            print("  (variables below are PARTIAL — the formula did not fully evaluate)")
        for name, value in view.variables.items():
            print(f"    {name} = {value}")


COMMAND = Command(
    name="autoscale-check",
    help="Evaluate the deployed autoscale formula on the live pool, errors included.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
