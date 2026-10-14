"""The `serve` subcommand: run the console.

Bound to the loopback interface and NOT configurable. There is no
authentication here, so binding it anywhere reachable would publish an
unauthenticated interface to the experiment record; making that a flag would
make it a mistake someone can make. The serving box runs a ``--read-only`` copy
on loopback too, and `tailscale serve` is its only door: the tailnet is the
authentication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse

HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver serve`."""
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes. For working on the server itself; the "
        "console's own hot reload comes from `just console-dev`.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Leave out every endpoint that dispatches, cancels, pushes code or "
        "profiles: the copy served off the laptop, to the tailnet.",
    )


class ServePayload(BaseModel):
    """WHERE the console will listen. `render` is what actually serves.

    A server never returns, which does not fit `run() -> payload`, and teaching
    the dispatcher about one command's behaviour would put a command-specific
    concern in the one place that has none. So the payload says where it would
    listen and the renderer -- already the only terminal-specific part -- is what
    blocks. Under `--json` nothing is served, which is right: a machine consumer
    wants the address, not a process.
    """

    op: Literal["serve"] = "serve"
    url: str
    host: str
    port: int
    reload: bool = False
    read_only: bool = False


def run(args: argparse.Namespace) -> ServePayload:
    """Describe where the server WILL listen; :func:`render` starts it."""
    return ServePayload(
        url=f"http://{HOST}:{args.port}",
        host=HOST,
        port=args.port,
        reload=args.reload,
        read_only=args.read_only,
    )


def render(payload: ServePayload) -> None:
    # Imported here, not at module scope: `--help` and every other subcommand
    # would otherwise pay to import uvicorn and the whole FastAPI stack.
    import uvicorn  # noqa: PLC0415 -- see above

    factory = "create_read_only_app" if payload.read_only else "create_app"
    print(
        f"Console on {payload.url}{'  read-only' if payload.read_only else ''}   (Ctrl-C to stop)"
    )
    try:
        uvicorn.run(
            f"src.interfaces.web.app:{factory}",
            factory=True,
            host=payload.host,
            port=payload.port,
            reload=payload.reload,
            log_level="warning",
        )
    except KeyboardInterrupt:
        # The documented way to stop it, so it exits like one rather than
        # unwinding a traceback through `headless.main`.
        print()


COMMAND = Command(
    name="serve",
    help="Serve the console on localhost.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
