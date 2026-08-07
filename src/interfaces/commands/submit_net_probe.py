"""The `submit-net-probe` subcommand: run `net-probe` on a pool node.

The question it settles is whether a node may talk to anything that is not an
Azure SDK endpoint -- outbound 5432, and an AAD token for a Postgres audience.
It exists as a dispatch rather than a one-off script because the answer is a
property of the POOL, changes when the pool's networking does, and is worth
being able to re-ask on the day something stops working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from src.interfaces.cloud.tasks import dispatch, spec
from src.interfaces.commands._base import Command
from src.shared.cloudtask.kinds import TaskName

if TYPE_CHECKING:
    import argparse

# Seconds of connects, not minutes. The ceiling is here so a probe that hits a
# black hole still returns a task record instead of occupying a node.
PROBE_TIMEOUT = "15m"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver submit-net-probe`."""
    parser.add_argument(
        "--port",
        type=int,
        action="append",
        dest="ports",
        default=None,
        help="TCP port to test outbound; repeatable (default 443 and 5432).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="A real endpoint as host:port, to re-run this probe end to end once a server exists.",
    )
    parser.add_argument("--tls", action="store_true", help="Complete a TLS handshake to --host.")
    parser.add_argument("--pool", default=None, help="Pool to run on (default: the training pool).")
    parser.add_argument("--timeout", default=PROBE_TIMEOUT, help="Wall-clock ceiling.")


def _flags(args: argparse.Namespace) -> tuple[str, ...]:
    """The probe's own command line, carried verbatim on ``eval_flags``."""
    flags: list[str] = []
    for port in args.ports or ():
        flags += ["--port", str(port)]
    if args.host:
        flags += ["--host", args.host]
    if args.tls:
        flags.append("--tls")
    return tuple(flags)


class SubmitNetProbePayload(dispatch.Dispatched):
    """A dispatch, plus the command line the node will run."""

    op: Literal["submit-net-probe"] = "submit-net-probe"
    flags: list[str] = Field(default_factory=list)


def run(args: argparse.Namespace) -> SubmitNetProbePayload:
    """Queue exactly one probe task."""
    flags = _flags(args)
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=TaskName.NET_PROBE,
                eval_flags=flags,
                timeout=args.timeout,
            )
        ],
        pool=args.pool or "train",
    )
    return payload.extend(SubmitNetProbePayload, flags=list(flags))


def render(payload: SubmitNetProbePayload) -> None:
    print(f"Queued a network probe: net-probe {' '.join(payload.flags)}".rstrip())
    dispatch.render_queued(payload)
    print("\nRead the answer with:  poker-solver logs --task <id>")


COMMAND = Command(
    name="submit-net-probe",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Run a reachability probe on a pool node: outbound ports, IMDS, and an AAD token.",
)
