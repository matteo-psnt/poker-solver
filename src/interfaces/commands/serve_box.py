"""The `serve-box` subcommand: wake or stop the blueprint host.

A reader command in the sense that matters here -- it answers against Azure, not
against a local tree -- but one that can also act. Grouped with **see and
dispatch** for that reason: it is the same shape as `cancel`, which also reads
the cloud and then changes it.

The console drives this through :meth:`Command.invoke`, so the button and the
command line are the same code and cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.cloud import serve_box
from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse

DEFAULT_RESOURCE_GROUP = "poker-solver-serve-rg"
DEFAULT_VM = "blueprint-server"
DEFAULT_SUBSCRIPTION = "f9c31345-15ac-413f-8841-5d0151baca66"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver serve-box`."""
    parser.add_argument(
        "--action",
        choices=("status", "start", "stop"),
        default="status",
        help="What to do. Default reports without changing anything.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until the box reaches its new state (~2 minutes to start).",
    )
    parser.add_argument("--resource-group", default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--vm", default=DEFAULT_VM)
    parser.add_argument("--subscription", default=DEFAULT_SUBSCRIPTION)


class BoxPayload(BaseModel):
    """The blueprint host's power state.

    `power` carries transitional values ("starting", "stopping") as well as the
    two stable ones, because a UI that only knew "running" and "deallocated"
    would show "stopped" for the whole two minutes a box takes to wake -- which
    reads as the button having done nothing.
    """

    op: Literal["serve-box"] = "serve-box"
    action: str
    vm: str
    resource_group: str
    power: str
    usable: bool
    location: str


def run(args: argparse.Namespace) -> BoxPayload:
    """Act if asked, then report the state either way.

    Always reporting -- rather than returning "started" -- keeps one payload
    shape for all three actions, so a caller renders the result of a click and
    the result of a poll with the same code. `start` on a running box and `stop`
    on a stopped one are both accepted for the same reason: they are idempotent
    requests for a state, and refusing them would make the console guard every
    button against a race it cannot win.
    """
    where = (args.subscription, args.resource_group, args.vm)

    if args.action == "start":
        serve_box.start(*where, wait=args.wait)
    elif args.action == "stop":
        serve_box.deallocate(*where, wait=args.wait)

    state = serve_box.status(*where)
    return BoxPayload(
        action=args.action,
        vm=state.name,
        resource_group=args.resource_group,
        power=state.power,
        usable=state.usable,
        location=state.location,
    )


def render(payload: BoxPayload) -> None:
    usable = "ready" if payload.usable else "not serving"
    print(f"{payload.vm}  {payload.power}  ({usable})")
    if payload.power == "deallocated":
        print("  `poker-solver serve-box --action start` to wake it (~2 min).")


COMMAND = Command(
    name="serve-box",
    help="Report, wake, or stop the blueprint host.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
