"""The `autoscale-check` subcommand: evaluate the live formula, errors included.

The error half is the entire point. An invalid or throwing formula still
returns partial ``results``, so reporting the results alone makes a broken
formula look healthy -- which is how a ``#`` comment (Batch wants ``//``) and a
one-argument ``GetSample`` both went unnoticed while the pool quietly stopped
scaling up. Run this after every formula change; server-side evaluation is
free, safe and instant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver autoscale-check`. It takes none."""


class AutoscalePayload(batch.AutoscaleResult):
    """The deployed formula, evaluated against the live pool.

    `error` is a FIELD, not a failed request: Batch evaluates the formula and
    reports that it did not compute, which is the answer to "why is the pool not
    growing" and must reach the screen rather than blanking the panel.
    """

    op: Literal["autoscale-check"] = "autoscale-check"
    pool_id: str


def run(args: argparse.Namespace) -> AutoscalePayload:  # noqa: ARG001
    """Evaluate the deployed formula against the live pool."""
    config = CloudConfig.load()
    result = batch.evaluate_autoscale(batch.client(config), config.pool_id)
    return AutoscalePayload(pool_id=config.pool_id, **result.model_dump())


def render(payload: AutoscalePayload) -> None:
    if payload.error is None:
        print("  no error")
    else:
        print(f"  ERROR: {payload.error.code}")
        if payload.error.message:
            print(f"    {payload.error.message}")
        for name, value in payload.error.values.items():
            print(f"    {name}: {value}")
        print("  (variables below are PARTIAL — the formula did not fully evaluate)")
    for name, value in payload.variables.items():
        print(f"    {name} = {value}")


COMMAND = Command(
    name="autoscale-check",
    help="Evaluate the deployed autoscale formula on the live pool, errors included.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
