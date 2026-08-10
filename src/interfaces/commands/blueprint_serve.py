"""The `blueprint-serve` subcommand: hold one run, and answer questions about it.

This runs on the long-lived reader box (`infra/serve/`), NOT on the training
pool. That distinction is the whole reason the box exists: the pool is for work
that *finishes* -- its autoscale formula sizes itself on running tasks,
`taskcompletion` deallocation assumes tasks end, and the task wall-clock guard
exists to kill anything that does not. A server is the shape all three are aimed
at, which is why there is deliberately no `TaskName` for it.

It keeps `--runs-dir` because it reads a checkpoint and the card abstraction from
LOCAL disk. Not a preference: a checkpoint is ~5,500 small files that the read
path mmaps, and over SMB every page fault becomes a network round trip.

Loopback only, and not configurable -- same rule as `serve`. There is no
authentication here, so binding anywhere reachable would publish a run to
whoever finds the port. Reaching it from elsewhere is a tunnel's job, which
keeps the authentication question with the thing that already answers it.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.commands._base import Command, resolve_run_dir
from src.interfaces.errors import CommandError

HOST = "127.0.0.1"
DEFAULT_PORT = 8790


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver blueprint-serve`."""
    parser.add_argument("--run", required=True, help="Run id, fragment, or path to a run dir.")
    parser.add_argument(
        "--runs-dir",
        default="/mnt/work/runs",
        help="Where runs live on this box. Local disk, never the share.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})."
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=0,
        help="Exit after this many seconds with no request. 0 stays up. On the "
        "hosted box the systemd unit turns this exit into a deallocate.",
    )
    parser.add_argument(
        "--at",
        type=int,
        default=None,
        help="Serve the checkpoint at this iteration rather than the newest.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the run and describe what would be served; :func:`render` serves it.

    Same shape as `serve`: a server never returns, which does not fit
    ``run() -> payload``. Resolving the run HERE rather than in the renderer is
    deliberate -- a bad `--run` is the most likely mistake, and it should be a
    refusal before a minute of loading rather than after it.
    """
    run_dir = resolve_run_dir(args.run, args.runs_dir)
    if not run_dir.is_dir():
        raise CommandError(f"No run directory at {run_dir}.")
    return {
        "op": "blueprint-serve",
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "at_iteration": args.at,
        "idle_timeout": args.idle_timeout,
        "url": f"http://{HOST}:{args.port}",
        "host": HOST,
        "port": args.port,
    }


def render(payload: dict[str, Any]) -> None:
    # Imported here, not at module scope, so `--help` and every other subcommand
    # is spared uvicorn, FastAPI and the whole pipeline import chain.
    from pathlib import Path

    import uvicorn

    from src.interfaces.blueprint.app import create_app
    from src.interfaces.blueprint.idle import IDLE_EXIT_CODE
    from src.pipeline.services.evaluation._shared import build_blueprint_for
    from src.pipeline.training.run_tracker import RunTracker

    run_dir = Path(payload["run_dir"])

    def _load():
        """Load once, at app construction. Takes ~1 min on a production run."""
        metadata = RunTracker.load(run_dir).metadata
        solver, _storage = build_blueprint_for(
            run_dir,
            metadata,
            abstraction_hash=metadata.card_abstraction_hash,
            at_iteration=payload["at_iteration"],
        )
        return solver

    print(f"Loading {payload['run']} …")
    app = create_app(_load, run_id=payload["run"], idle_timeout_seconds=payload["idle_timeout"])
    print(f"Blueprint server on {payload['url']}   (Ctrl-C to stop)")

    # An explicit Server rather than `uvicorn.run(...)`, so idle expiry can ask
    # it to stop instead of signalling the process. MEASURED: uvicorn re-raises
    # the captured signal after restoring the default handler, so on the SIGTERM
    # path `run()` never returns and the process is 143 no matter what this
    # function would rather exit with. `should_exit` returns control here.
    server = uvicorn.Server(
        uvicorn.Config(app, host=payload["host"], port=payload["port"], log_level="warning")
    )
    app.state.idle.expire_with(lambda: setattr(server, "should_exit", True))

    try:
        server.run()
    except KeyboardInterrupt:
        print()
        return

    # WHY the server stopped, as an exit code, because that is all the systemd
    # unit can read. Idle expiry means "switch the box off"; every other way of
    # stopping -- Ctrl-C, `systemctl stop`, the `restart` in deploy.sh -- must
    # NOT, and they all look identical by this point. See `idle.IDLE_EXIT_CODE`
    # for the 62 hours this cost.
    if app.state.idle.fired:
        raise SystemExit(IDLE_EXIT_CODE)


COMMAND = Command(
    name="blueprint-serve",
    help="Serve one trained run for reading, on localhost.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
