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

It serves ONE run until it is stopped. The box also holds a Chipzen ladder slot,
so the run to read is whichever the deploy seated -- the same deploy, staging it
once for both.

Loopback only, and not configurable -- same rule as `serve`. There is no
authentication here, so binding anywhere reachable would publish a run to
whoever finds the port. Reaching it from elsewhere is a tunnel's job, which
keeps the authentication question with the thing that already answers it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command, resolve_run_dir
from src.interfaces.errors import CommandError
from src.shared.cloudtask.node.paths import NodePaths

if TYPE_CHECKING:
    import argparse

    from src.engine.solver.policy.source import ScorableBlueprint

HOST = "127.0.0.1"
DEFAULT_PORT = 8790


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver blueprint-serve`."""
    parser.add_argument("--run", required=True, help="Run id, fragment, or path to a run dir.")
    parser.add_argument(
        "--runs-dir",
        default=str(NodePaths.from_environment().runs),
        help="Where runs live on this box; the node's own runs directory by default.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})."
    )
    parser.add_argument(
        "--at",
        type=int,
        default=None,
        help="Serve the checkpoint at this iteration rather than the newest.",
    )
    parser.add_argument(
        "--rung",
        action="append",
        default=None,
        metavar="RUN[:AT]",
        help="A SHALLOWER blueprint to serve beside `--run`, repeatable. Same "
        "spelling as `chipzen-seat --rung`, so one deploy can hand both the same "
        "ladder. `/api/run` lists the stacks held and `/api/node?stack=N` picks "
        "one; omitting it reads the deepest. Serving the seat's whole ladder costs "
        "the load, not another copy of the deepest run.",
    )


class BlueprintServePayload(BaseModel):
    """WHERE the blueprint server will listen, and which run it will hold."""

    op: Literal["blueprint-serve"] = "blueprint-serve"
    run: str
    run_dir: str
    at_iteration: int | None = None
    url: str
    host: str
    port: int
    rungs: list[str] = []
    """Shallower blueprints beside `run`, as `run[:at]`. Empty serves one depth."""


def run(args: argparse.Namespace) -> BlueprintServePayload:
    """Resolve the run and describe what would be served; :func:`render` serves it.

    Same shape as `serve`: a server never returns, which does not fit
    ``run() -> payload``. Resolving the run HERE rather than in the renderer is
    deliberate -- a bad `--run` is the most likely mistake, and it should be a
    refusal before a minute of loading rather than after it.
    """
    run_dir = resolve_run_dir(args.run, args.runs_dir)
    if not run_dir.is_dir():
        raise CommandError(f"No run directory at {run_dir}.")
    # Resolved HERE too: a bad `--rung` should be refused beside a bad `--run`,
    # not a minute into the load of the rungs that were fine.
    rungs = []
    for spec in args.rung or []:
        # `run[:at[:threshold]]`, the SEAT's spelling, so one deploy can hand both
        # units the same list. The threshold is the seat's and is ignored here -- a
        # reader shows the table as trained. Splitting on the FIRST colon only would
        # read "12000:0.20" as the iteration and refuse every fielded rung.
        name, _, rest = spec.partition(":")
        at, _, _threshold = rest.partition(":")
        directory = resolve_run_dir(name, args.runs_dir)
        if not directory.is_dir():
            raise CommandError(f"No run directory at {directory} (from --rung {spec!r}).")
        if at and not at.isdigit():
            raise CommandError(f"--rung {spec!r}: {at!r} is not an iteration.")
        rungs.append(f"{directory}:{at}" if at else str(directory))
    return BlueprintServePayload(
        run=run_dir.name,
        run_dir=str(run_dir),
        at_iteration=args.at,
        url=f"http://{HOST}:{args.port}",
        host=HOST,
        port=args.port,
        rungs=rungs,
    )


def render(payload: BlueprintServePayload) -> None:
    # Imported here, not at module scope, so `--help` and every other subcommand
    # is spared uvicorn, FastAPI and the whole pipeline import chain. This is a
    # `render()`, so it runs only when this command is the one being run --
    # which is what makes the whole block deferrable rather than just uvicorn.
    from pathlib import Path  # noqa: PLC0415 -- see above

    import uvicorn  # noqa: PLC0415 -- see above

    from src.adapters.postgres import connect  # noqa: PLC0415 -- see above
    from src.interfaces.blueprint.app import create_app  # noqa: PLC0415 -- see above
    from src.pipeline.services.scoring._shared import (  # noqa: PLC0415 -- see above
        build_blueprint_for,
    )
    from src.pipeline.training.run_tracker import RunTracker  # noqa: PLC0415 -- see above

    run_dir = Path(payload.run_dir)
    # The serving box needs the DSN in its environment once the log stops
    # being published; without one this falls back to the file, as before.
    source = connect.record_source_from_environment()

    def _build(directory: Path, at_iteration: int | None):
        """A run directory on local disk -> a blueprint. ~1 min in production."""
        metadata = RunTracker.load(directory, source).metadata
        solver, _storage, _policy = build_blueprint_for(
            directory,
            metadata,
            abstraction_hash=metadata.card_abstraction_hash,
            at_iteration=at_iteration,
            # Same reason the SEAT does it: a reader answers `/api/node`,
            # `/api/combos` and `/api/play` from `strategy_sum` and `visited`, and
            # never resumes. The three resume-only arrays are 1.23 GB of a 2.7 GB
            # blueprint, and this box holds the seat's whole ladder beside us --
            # measured 09-29 at 1.84 GB private here against the seat's 2.82 GB
            # for EIGHT rungs, entirely because this call loaded all five arrays.
            play_only=True,
        )
        return solver

    def _load_ladder() -> tuple[dict[int, ScorableBlueprint], dict[int, str]]:
        """Every rung, keyed by starting stack, plus which run each came from.

        `--run` is applied LAST so it wins a tie: two rungs at one stack is a
        deploy mistake, and the deepest is the one the operator named explicitly.
        """
        ladder: dict[int, ScorableBlueprint] = {}
        names: dict[int, str] = {}
        specs = [*payload.rungs, f"{run_dir}:{payload.at_iteration or ''}"]
        for spec in specs:
            directory, _, at = spec.rpartition(":")
            path = Path(directory or spec)
            solver = _build(path, int(at) if at else None)
            stack = solver.config.game.starting_stack
            ladder[stack] = solver
            names[stack] = path.name
        return ladder, names

    print(f"Loading {payload.run} and {len(payload.rungs)} rung(s) …")
    ladder, names = _load_ladder()
    app = create_app(lambda: ladder, run_id=payload.run, run_ids=names)
    print(f"Blueprint server on {payload.url}   (Ctrl-C to stop)")

    try:
        uvicorn.run(app, host=payload.host, port=payload.port, log_level="warning")
    except KeyboardInterrupt:
        # The documented way to stop it, so it exits like one rather than
        # unwinding a traceback through `headless.main`.
        print()


COMMAND = Command(
    name="blueprint-serve",
    help="Serve one trained run for reading, on localhost.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
