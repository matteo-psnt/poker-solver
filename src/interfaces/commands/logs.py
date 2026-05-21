"""The `logs` subcommand: a task's output, from the share or from the node.

Replaces three separate recipes (`job-log`, `task-log`, `task-logs`) because
they differ only in where they read from, and the choice between them is the
one thing a caller actually has to think about:

* ``--source share`` (default) reads the copy the node wrapper publishes on every
  checkpoint. It **survives node teardown**, which matters because the pool
  scales to zero within minutes of a task ending -- so the node copy is already
  gone for exactly the failed tasks most worth reading.
* ``--source node`` reads the live ``stdout.txt``/``stderr.txt`` while the task
  is still running. Fresher, but it answers ``NodeNotFound`` once the node is
  released.
"""

from __future__ import annotations

import argparse
from typing import Any

from azure.core.exceptions import ResourceNotFoundError

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import share
from src.interfaces.cloud.tasks import batch
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError

PROGRESS_PREFIX = "Training batches:"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver logs`."""
    parser.add_argument("--task", default=None, help="Task id to read.")
    parser.add_argument(
        "--list", action="store_true", help="List published task logs instead of reading one."
    )
    parser.add_argument(
        "--source",
        default="share",
        choices=["share", "node"],
        help="share = the published copy (survives teardown); node = live task output.",
    )
    parser.add_argument("--job", default=None, help="Job id. Required for --source node.")
    parser.add_argument(
        "--stream",
        default="stdout",
        choices=["stdout", "stderr"],
        help="Which node-side stream to read (--source node only).",
    )
    parser.add_argument("--lines", type=int, default=80, help="Tail this many lines (0 = all).")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Keep tqdm progress lines. They are stripped by default: a training "
        "stream is mostly carriage-returned progress bars, which bury the real output.",
    )


def _clean(text: str, *, raw: bool, lines: int) -> list[str]:
    """Split a log into displayable lines.

    Carriage returns are expanded first: tqdm redraws a progress bar in place
    with ``\\r``, so a raw read is a handful of enormous lines rather than the
    stream a human expects.
    """
    expanded = text.replace("\r", "\n").splitlines()
    kept = expanded if raw else [line for line in expanded if not line.startswith(PROGRESS_PREFIX)]
    return kept[-lines:] if lines > 0 else kept


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Read one task's log, or list what is published.

    Ordering:
        Everything the caller could have got right is refused first. `CloudConfig.load()`
        shells out to Terraform, so validating after it charges a round trip for a typo
        -- and made a "refused before any call" test pass only where `infra/.terraform`
        happened to be populated.
    """
    if not args.list and not args.task:
        raise CommandError("logs: --task is required unless --list is given.")

    if not args.list and args.source == "node" and not args.job:
        raise CommandError(
            "logs --source node needs --job: node-side files are addressed by "
            "(job, task). Use --source share to read by task id alone."
        )

    config = CloudConfig.load()
    if args.list:
        service = share.share_client(config)
        return {
            "op": "logs",
            "listing": share.task_log_names(service, config.share_name),
            "lines": None,
            "task": None,
        }

    if args.source == "node":
        try:
            text = batch.task_file(batch.client(config), args.job, args.task, f"{args.stream}.txt")
        except ResourceNotFoundError as error:
            # The normal end state, not an exotic one: the pool scales to zero
            # within minutes of a task ending, and Batch then answers "The
            # specified node does not exist." Uncaught it was a 40-line Azure
            # traceback at the exact moment someone was trying to find out why a
            # task died -- and it buried the fact that the answer is one flag
            # away, on the share, which is what the publish-on-exit trap is for.
            raise CommandError(
                f"No node-side files for {args.task}: {error.message.strip()}\n"
                "The node is released once the pool scales down, so this path only "
                "works while a task is running.\n"
                f"Read the published copy instead: logs --task {args.task}"
            ) from error
    else:
        found = share.read_task_log(share.share_client(config), config.share_name, args.task)
        if found is None:
            raise CommandError(
                f"No published log for {args.task}. It may still be running before its "
                "first publish, or the task id may be wrong — list them with --list."
            )
        text = found

    return {
        "op": "logs",
        "listing": None,
        "task": args.task,
        "lines": _clean(text, raw=args.raw, lines=args.lines),
    }


def render(payload: dict[str, Any]) -> None:
    if payload["listing"] is not None:
        for name in payload["listing"]:
            print(f"  {name}")
        if not payload["listing"]:
            print("  no published logs yet")
        return
    for line in payload["lines"]:
        print(line)


COMMAND = Command(
    name="logs",
    help="Read a task's log from the share (default) or live from its node.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
