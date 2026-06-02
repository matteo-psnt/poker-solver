"""The `runs` subcommand: every published run, newest first.

Listing runs used to need no command: a caller globbed the local runs directory.
With the record living only on the share there is no directory to glob, and two
surfaces immediately needed one -- the interactive picker, which had started
reporting "no trained runs found" against a path that no longer exists, and the
console's `/runs` page.

Both go through here rather than reaching for the share themselves. That is the
same rule the rest of the command layer follows: one implementation per question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command, records_root
from src.pipeline import services
from src.pipeline.services.runs import RunSummary

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver runs`."""
    parser.add_argument(
        "--limit", type=int, default=0, help="Show only the newest N runs (0 = all)."
    )
    parser.add_argument(
        "--loadable-only",
        action="store_true",
        help="Hide runs that never checkpointed. They are still worth listing by "
        "default: a run that died before its first checkpoint is exactly the one "
        "someone is looking for when they ask what happened.",
    )


class RunsPayload(BaseModel):
    """Every published run, newest first."""

    op: Literal["runs"] = "runs"
    runs: list[services.RunSummary] = []


def run(args: argparse.Namespace) -> RunsPayload:
    """Summarise every published run, newest first."""
    with records_root(args) as root:
        summaries = services.describe_runs(root)
    if args.loadable_only:
        summaries = [summary for summary in summaries if summary.loadable]
    if args.limit > 0:
        summaries = summaries[: args.limit]
    return RunsPayload(runs=summaries)


def render(payload: RunsPayload) -> None:
    rows = payload.runs
    if not rows:
        print("No published runs.")
        return
    print(f"{len(rows)} published run(s), newest first")
    header = f"{'run':<34} {'config':<14} {'iterations':>13} {'status':<11} {'age':<16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        commits = row.commits_ago
        age = "commit unknown" if commits is None else f"{commits} commit(s) ago"
        iterations = row.iterations
        print(
            f"{row.name[:34]:<34} "
            f"{(row.config_name or '—')[:14]:<14} "
            f"{(f'{iterations:,}' if iterations is not None else '—'):>13} "
            f"{(row.status or '—')[:11]:<11} "
            f"{age:<16}" + ("" if row.loadable else f"  ({row.blocker})")
        )


__all__ = ["COMMAND", "RunSummary", "add_arguments", "render", "run"]

COMMAND = Command(
    name="runs",
    help="Every published run, newest first.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
