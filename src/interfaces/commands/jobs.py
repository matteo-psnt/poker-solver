"""The `jobs` subcommand: what is queued, running or finished on the pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.cloud.tasks.batch import Job
from src.interfaces.commands._base import Command
from src.shared import task_states

if TYPE_CHECKING:
    import argparse


class JobsPayload(BaseModel):
    """What `jobs` answers. The console reads this through `/api/jobs`."""

    op: Literal["jobs"] = "jobs"
    jobs: list[Job] = []
    total_jobs: int
    """Two different absences, counted apart. One number for both read "20 of
    72 jobs" as "total 0" once, because nothing said which flag would show the
    rest: `filtered_jobs` are finished ones the default view hides (`--all`),
    `hidden_jobs` are the ones `--limit` cut (`--limit 0`)."""
    filtered_jobs: int = 0
    hidden_jobs: int = 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver jobs`."""
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include finished jobs and finished tasks. The default shows what is in "
        "flight: a day of scoring leaves hundreds of completed tasks, and burying "
        "the one running task in that list is how a stalled run goes unnoticed.",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Show only the last N jobs (0 = all)."
    )


def is_active(job: Job) -> bool:
    """The half of ``is_live`` that can be decided WITHOUT listing tasks.

    Load-bearing on its own, and not only as an optimisation. A terminated job
    leaves its tasks frozen in whatever state they held when the node went
    away, so the account is full of tasks reading ``running`` or ``active``
    that have not existed for days -- 13 of them against a pool sitting at zero
    nodes, at the time this was written. Trusting task state alone reports an
    idle pool as busy, which is precisely backwards for the one question this
    command exists to answer.

    A JOB's ``active`` is not a task's: it means the container is open, and it
    stays open all day with nothing in it. That is why this is only half.
    """
    return (job.state or "").rsplit(".", 1)[-1].lower() == "active"


def is_live(job: Job) -> bool:
    """A job worth showing by default: real work, still in flight.

    ``IN_FLIGHT`` rather than a set spelled here: this asks "is anything
    happening", which includes a task queued for a node. That it differs from
    what cost accounting counts is deliberate and is decided once, in
    :mod:`~src.shared.task_states`.
    """
    if not is_active(job):
        return False
    return any(task.phase in task_states.IN_FLIGHT for task in job.tasks)


def select(jobs: list[Job], *, show_all: bool, limit: int) -> JobsPayload:
    """Order the jobs chronologically, then keep the NEWEST ``limit`` of them.

    The sort is the load-bearing half. Batch returns jobs in its own order,
    which is not chronological -- measured 09-01, the list opened at 08-31 and
    closed at 08-26 -- so `[-limit:]` over it kept the OLDEST 20 and hid the 52
    newest, burying today's job behind August 4th's. Ids are
    ``poker-YYYYMMDD[-pool]``, fixed-width and so lexicographically
    chronological.
    """
    jobs = sorted(jobs, key=lambda job: job.job)
    eligible = jobs if show_all else [job for job in jobs if is_live(job)]
    shown = eligible[-limit:] if limit > 0 else eligible
    return JobsPayload(
        jobs=shown,
        total_jobs=len(jobs),
        filtered_jobs=len(jobs) - len(eligible),
        hidden_jobs=len(eligible) - len(shown),
    )


def run(args: argparse.Namespace) -> JobsPayload:
    """List jobs and their tasks, newest last.

    Tasks are fetched only for the jobs that can survive the filter. Listing
    them costs ~0.39s per job against ~1.1s for the whole job list, so fetching
    all 44 to render the 2 that were active was the entire cost of this command
    -- measured at ~11s, and paid again by every caller, including `tasks` and
    the status screen.
    """
    client = batch.client(CloudConfig.load())
    jobs = batch.attach_tasks(
        client,
        batch.list_jobs(client),
        want=None if args.all else is_active,
        in_flight_only=not args.all,
    )
    return select(jobs, show_all=args.all, limit=args.limit)


def render(payload: JobsPayload) -> None:
    if not payload.jobs:
        print("Nothing running.")
    for job in payload.jobs:
        print(f"== {job.job}  [{(job.state or '').rsplit('.', 1)[-1].lower()}]")
        for task in job.tasks:
            # The OUTCOME once it has stopped, because `finished` says only that
            # it is over -- a cancellation and a crash both read `finished` with
            # an exit code beside them, and the code is the half nobody
            # remembers. `exit_meaning` explains the recurring ones; an
            # unfamiliar code stays a bare number rather than being guessed at.
            word = task.outcome or task.phase
            exit_code = "" if task.exit_code is None else f"  exit={task.exit_code}"
            because = f"  ({task.exit_meaning})" if task.exit_meaning else ""
            # The ATTEMPT, whenever it is not the first. Batch reruns a failed task
            # and the rerun reads `running`, so a crash-loop is indistinguishable
            # from progress on this screen without it -- which is exactly how four
            # OOM-looping scores looked healthy for 4.7 hours on 09-27.
            attempt = f"  ATTEMPT {task.retry_count + 1}" if task.retry_count else ""
            print(f"   {word:<10} {task.task}{attempt}{exit_code}{because}")
    if payload.hidden_jobs:
        listed = len(payload.jobs) + payload.hidden_jobs
        print(f"\n  showing {len(payload.jobs)} of {listed} job(s) — `--limit 0` for all")
    if payload.filtered_jobs:
        print(f"  {payload.filtered_jobs} finished job(s) hidden — show with --all")


COMMAND = Command(
    name="jobs",
    help="Every queued/running task on the pool (--all includes finished jobs).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
