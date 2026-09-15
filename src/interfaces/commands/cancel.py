"""The `cancel` subcommand: stop one task, keeping what it has produced.

Not a loss of work. The node wrapper publishes on exit, so everything up to the
last retained rung survives on the share and ``submit --run <id> --to <n>``
picks it back up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands import tasks
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver cancel`."""
    parser.add_argument("--task", required=True, help="Task id to terminate.")
    parser.add_argument(
        "--job",
        default="",
        help="Job id holding the task. Omit it and the record answers: a task's "
        "start leg names its job. Needed only for a task recorded before that field.",
    )


class CancelledPayload(BaseModel):
    """What `cancel` reports back.

    `job_id`/`task_id`, which is what this command has always returned. The
    hand-written Zod this replaced declared `job`/`task`, so the console's cancel
    button terminated the task and then threw a parse error reporting that it had
    failed -- the worst shape of bug this seam can produce, because the operator
    retries an action that already worked.
    """

    op: Literal["cancel"] = "cancel"
    job_id: str
    task_id: str


def job_of(task_id: str) -> str:
    """The job a task was queued under, from the record.

    Batch addresses a task by (job, task), but the task id is unique on its own
    and the record already holds the pair -- demanding `--job` cost eleven
    failed calls across seven sessions. Read through `tasks` so there is one
    join; reconcile is skipped because the answer is on the node's own leg.
    Never Batch enumeration: 0.39 s per job over every job in the account.
    """
    payload = tasks.COMMAND.invoke_as(tasks.TasksPayload, task=task_id, skip_reconcile=True)
    jobs = {row.job_id for row in payload.rows if row.job_id}
    if not payload.rows:
        raise CommandError(
            f"cancel: the record has no task {task_id!r}; pass --job to address it directly."
        )
    if not jobs:
        raise CommandError(
            f"cancel: the record holds {task_id} but not its job (recorded before the "
            "field existed); pass --job."
        )
    return max(jobs)


def run(args: argparse.Namespace) -> CancelledPayload:
    """Terminate one task."""
    job = args.job or job_of(args.task)
    config = CloudConfig.load()
    batch.cancel_task(batch.client(config), job, args.task)
    return CancelledPayload(job_id=job, task_id=args.task)


def render(payload: CancelledPayload) -> None:
    print(f"Terminated {payload.task_id} in {payload.job_id}.")
    print("  Partial progress up to the last retained rung is published to the store.")


COMMAND = Command(
    name="cancel",
    help="Terminate a running task; its partial progress is published first.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
