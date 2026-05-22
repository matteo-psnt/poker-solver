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

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from src.interfaces.commands import jobs, pool_status, tasks
from src.interfaces.commands._base import Command
from src.interfaces.errors import attempt

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


def _panel(command: Command, **kwargs: Any) -> dict[str, Any]:
    """Answer one panel, or record why it could not be answered.

    Which failures are survivable is :func:`attempt`'s decision, not this
    module's -- an expired ``az login`` and an unreachable endpoint are the two
    a status screen exists to outlive, and the console needs the same list.
    Anything else propagates: a bug in a panel is still a bug, and swallowing it
    would turn this screen into the place exceptions go to be silently rendered
    as "unavailable".

    The screen shows a reason, not a kind -- a greyed-out panel says why in
    words either way -- so the classification is dropped here and used by the
    console, which has to pick a status code from it.
    """
    payload, failure = attempt(lambda: command.invoke(**kwargs))
    return {"payload": payload, "error": failure.message if failure else None}


def gather(*, limit: int = 10, with_tasks: bool = True) -> dict[str, Any]:
    """Fetch every panel concurrently and return them keyed by name.

    The entry point for any surface that is not this command. Each panel builds
    its own Batch client, so there is no shared mutable state between the
    threads; the concurrency is here rather than inside the panels because it
    is a property of showing them together, not of any one of them.
    """
    wanted = [(name, command) for name, command in PANELS if with_tasks or name != "tasks"]
    # `tasks` defaults to the whole history on purpose (a death is the row worth
    # finding), but a screen meant to be glanced at cannot carry 200 rows.
    arguments: dict[str, dict[str, Any]] = {"jobs": {"limit": limit}, "tasks": {"limit": limit}}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        futures = {
            name: pool.submit(_panel, command, **arguments.get(name, {}))
            for name, command in wanted
        }
        panels = {name: future.result() for name, future in futures.items()}
    return {
        "op": "status",
        "at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "panels": panels,
    }


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
