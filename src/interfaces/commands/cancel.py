"""The `cancel` subcommand: stop one task, keeping what it has produced.

Not a loss of work. The node wrapper publishes on exit, so everything up to the
last retained rung survives on the share and ``submit --run <id> --to <n>``
picks it back up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver cancel`."""
    parser.add_argument("--job", required=True, help="Job id holding the task.")
    parser.add_argument("--task", required=True, help="Task id to terminate.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Terminate one task."""
    config = CloudConfig.load()
    batch.cancel_task(batch.client(config), args.job, args.task)
    return {"op": "cancel", "job_id": args.job, "task_id": args.task}


def render(payload: dict[str, Any]) -> None:
    print(f"Terminated {payload['task_id']} in {payload['job_id']}.")
    print("  Partial progress up to the last retained rung is published on the share.")


COMMAND = Command(
    name="cancel",
    help="Terminate a running task; its partial progress is published first.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
