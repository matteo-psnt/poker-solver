"""The `jobs` subcommand: what is queued, running or finished on the pool."""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.cli.commands._base import Command
from src.interfaces.cloud import batch
from src.interfaces.cloud.config import CloudConfig

LIVE_TASK_STATES = {"active", "preparing", "running"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver jobs`."""
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include jobs whose tasks have all finished. The default hides them: a "
        "day of scoring leaves dozens of completed single-task jobs, and burying "
        "the one running leg in that list is how a stalled run goes unnoticed.",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Show only the last N jobs (0 = all)."
    )


def short_state(state: str | None) -> str:
    """``BatchTaskState.RUNNING`` -> ``running``."""
    return (state or "").rsplit(".", 1)[-1].lower()


def is_active(job: dict[str, Any]) -> bool:
    """The half of ``is_live`` that can be decided WITHOUT listing tasks.

    Load-bearing on its own, and not only as an optimisation. A terminated job
    leaves its tasks frozen in whatever state they held when the node went
    away, so the account is full of tasks reading ``running`` or ``active``
    that have not existed for days -- 13 of them against a pool sitting at zero
    nodes, at the time this was written. Trusting task state alone reports an
    idle pool as busy, which is precisely backwards for the one question this
    command exists to answer.
    """
    return short_state(job["state"]) == "active"


def is_live(job: dict[str, Any]) -> bool:
    """A job worth showing by default: real work, still in flight."""
    if not is_active(job):
        return False
    return any(short_state(task["state"]) in LIVE_TASK_STATES for task in job["tasks"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    """List jobs and their tasks, newest last.

    Tasks are fetched only for the jobs that can survive the filter. Listing
    them costs ~0.39s per job against ~1.1s for the whole job list, so fetching
    all 44 to render the 2 that were active was the entire cost of this command
    -- measured at ~11s, and paid again by every caller, including `legs` and
    the status screen.
    """
    client = batch.client(CloudConfig.load())
    jobs = batch.attach_tasks(client, batch.list_jobs(client), want=None if args.all else is_active)
    shown = jobs if args.all else [job for job in jobs if is_live(job)]
    if args.limit > 0:
        shown = shown[-args.limit :]
    return {
        "op": "jobs",
        "jobs": shown,
        "total_jobs": len(jobs),
        "hidden_jobs": len(jobs) - len(shown),
    }


def render(payload: dict[str, Any]) -> None:
    if not payload["jobs"]:
        print("Nothing running.")
    for job in payload["jobs"]:
        print(f"== {job['job']}  [{short_state(job['state'])}]")
        for task in job["tasks"]:
            state = short_state(task["state"]) or "?"
            exit_code = "" if task["exit_code"] is None else f"  exit={task['exit_code']}"
            print(f"   {state:<10} {task['task']}{exit_code}")
    if payload["hidden_jobs"]:
        print(f"\n  {payload['hidden_jobs']} finished job(s) hidden — show with --all")


COMMAND = Command(
    name="jobs",
    help="Every queued/running task on the pool (--all includes finished jobs).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
