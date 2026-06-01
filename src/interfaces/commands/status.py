"""The `status` subcommand: one screen for "what is the pool doing right now".

Three commands already answer a third of the question each, and answering it
meant running all three and holding the join in your head: ``pool-status`` says
how many nodes exist, ``jobs`` says what Batch is running, and ``tasks`` is the
only one that can say why something DIED -- the run log cannot record a death
because the container is gone first.

Two properties are load-bearing, and both come from measuring rather than
guessing.

**Panels fail independently.** A panel is a :class:`CommandError` or an Azure
exception away from being unavailable at any moment -- an expired ``az login``
takes out both Batch panels at once -- and the whole point of a status screen is
that it still tells you the other two things. Nothing here lets one failure
blank the board.

**Panels are fetched concurrently**, because they are not equally cheap.
Measured against the live pool: ``pool-status`` 0.9s warm, ``jobs`` ~11s (it
issues one ``list_tasks`` call per job), ``tasks`` 23s when it has unresolved
tasks to reconcile against Batch and 9.3s when it does not. Serially that is a
~35s screen; concurrently it is however long the slowest panel takes. This is
also why ``--watch`` has a floor: a tick that cannot finish before the next one
starts is not a refresh interval, it is a queue.

This module composes; it does not read. Every panel is
:meth:`Command.invoke`, so there is exactly one implementation of each question
and this cannot drift from the command that owns it -- and a future non-terminal
surface calls :func:`gather` and gets the same payload rather than growing a
second read path.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.interfaces.commands import jobs, pool_status, tasks
from src.interfaces.commands._base import Command
from src.interfaces.commands._compose import Part, compose

if TYPE_CHECKING:
    import argparse

PANELS: tuple[tuple[str, Command], ...] = (
    ("pool", pool_status.COMMAND),
    ("jobs", jobs.COMMAND),
    ("tasks", tasks.COMMAND),
)

# Below the measured cost of a full cycle, a tick cannot finish before the next
# one is due. The screen would not refresh faster; it would only spend the whole
# interval mid-fetch, and every panel would be showing a different instant.
MIN_INTERVAL = 30


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver status`."""
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help=f"Refresh every N seconds until interrupted (minimum {MIN_INTERVAL}; "
        "0 = print once and exit).",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Show only the last N jobs (0 = all)."
    )
    parser.add_argument(
        "--no-tasks",
        action="store_true",
        help="Skip the task history. It is the slowest panel by a wide margin -- it "
        "downloads the share's account and reconciles unresolved tasks against "
        "Batch -- and the only one that can explain a death.",
    )


def gather(*, limit: int = 10, with_tasks: bool = True) -> dict[str, Any]:
    """Fetch every panel concurrently and return them keyed by name.

    The entry point for any surface that is not this command. The fan-out lives
    in :mod:`_compose` rather than here: this screen had the only copy of it
    until the console needed the same three properties -- independent failure,
    concurrency, and a context copy per submit -- and a second copy is how the
    telemetry-surface bug fixed in `d67411f` would have come back.

    ``panels`` rather than `compose`'s ``parts``, because that is the word this
    command's renderer and its `--json` consumers already use, and renaming a
    published key to match an internal one is a break with nothing behind it.
    """
    # `tasks` defaults to the whole history on purpose (a death is the row worth
    # finding), but a screen meant to be glanced at cannot carry 200 rows.
    arguments: dict[str, dict[str, Any]] = {"jobs": {"limit": limit}, "tasks": {"limit": limit}}
    parts = [
        Part(key=name, command=command, arguments=arguments.get(name, {}))
        for name, command in PANELS
        if with_tasks or name != "tasks"
    ]
    composed = compose("status", parts)
    composed["panels"] = composed.pop("parts")
    return composed


def run(args: argparse.Namespace) -> dict[str, Any]:
    """One snapshot, plus what a follower would need to take the next one.

    ``--watch`` rides on the payload rather than looping here, and rather than
    being special-cased in ``headless``. ``run`` returning a value that is not
    a snapshot would break ``--json`` and every programmatic caller; teaching
    the dispatcher about one command's flag would put a command-specific
    concern in the one place that is supposed to have none. So the snapshot
    carries the interval, and the renderer -- which is the only part that is
    terminal-specific anyway -- is what repeats. Under ``--json`` the loop
    never runs, which is right: a machine consumer polls on its own schedule.
    """
    interval = max(args.watch, MIN_INTERVAL) if args.watch else 0
    snapshot = gather(limit=args.limit, with_tasks=not args.no_tasks)
    return {
        **snapshot,
        "watch": interval,
        "requested_watch": args.watch,
        "limit": args.limit,
        "with_tasks": not args.no_tasks,
    }


PANEL_RENDERERS: dict[str, Any] = {
    "pool": pool_status.render,
    "jobs": jobs.render,
    "tasks": tasks.render,
}

PANEL_TITLES: dict[str, str] = {
    "pool": "POOL",
    "jobs": "BATCH",
    "tasks": "TASKS",
}


def _render_once(payload: dict[str, Any]) -> None:
    """Print each panel under a heading, delegating to the command that owns it.

    Deliberately no formatting of its own. A status screen that re-rendered the
    job table would be a second renderer for the same payload, free to disagree
    with ``jobs`` about what a task looks like -- the ``checkpoint-profile``
    failure with the arrow reversed.
    """
    print(f"{payload['at']}   ({payload['elapsed_seconds']}s)")
    for name, panel in payload["panels"].items():
        print(f"\n── {PANEL_TITLES.get(name, name.upper())} " + "─" * 40)
        if panel["error"]:
            print(f"  unavailable: {panel['error']}")
            continue
        PANEL_RENDERERS[name](panel["payload"])


def render(payload: dict[str, Any]) -> None:
    """Print the snapshot, then keep printing if it asked to be followed.

    Ctrl-C is the DOCUMENTED way out of the loop, so it has to be an ordinary
    exit rather than an escaping ``KeyboardInterrupt``: uncaught it unwinds
    through ``headless.main``, which only translates ``CommandError``, and
    prints a traceback every single time the user stops watching.
    """
    _render_once(payload)
    interval = payload.get("watch") or 0
    if not interval:
        return
    if interval != payload.get("requested_watch", interval):
        print(f"\nnote: interval raised to {interval}s — a full cycle takes longer than that.")
    try:
        while True:
            print(f"\nrefreshing every {interval}s — Ctrl-C to stop")
            time.sleep(interval)
            snapshot = gather(limit=payload["limit"], with_tasks=payload["with_tasks"])
            # Home the cursor and clear, rather than scrolling: this is meant to
            # be watched, and a scrolling log of identical tables is not.
            print("\033[H\033[J", end="")
            _render_once(snapshot)
    except KeyboardInterrupt:
        print()


COMMAND = Command(
    name="status",
    help="Pool, Batch and task history on one screen (--watch to follow).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
