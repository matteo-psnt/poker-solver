"""The `submit-migrate` subcommand: run the checkpoint migration on a node.

A task rather than something you type, because the bytes are on the SHARE: the
retained ladder is ~1,200 rungs of ~4,200 zarr chunk files each, roughly five
million reads. A node has that share mounted inside the region; a laptop has it
across the internet.

Expect to run it MORE THAN ONCE. The sweep is idempotent -- every rung is
checked against the container before it is read -- so a second dispatch costs
one HEAD per rung and uploads only what the first did not reach before its
deadline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from src.interfaces.cloud.tasks import dispatch, spec
from src.interfaces.commands._base import Command
from src.shared.cloudtask.kinds import TaskName

if TYPE_CHECKING:
    import argparse

# Long, because the sweep is bounded by SMB metadata round trips rather than by
# bytes, and it resumes cleanly. Shorter than Batch's own ceiling so the task
# reports a timeout rather than being killed as unresponsive.
MIGRATE_TIMEOUT = "6h"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver submit-migrate`."""
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Only these run ids (default: everything published).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many rungs. Useful for a first, small, provable sweep.",
    )
    parser.add_argument(
        "--recover-tars",
        action="store_true",
        help="Convert rungs that exist only as .tar objects; skip everything the share holds.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Upload nothing; report which published rungs the container lacks.",
    )
    parser.add_argument("--pool", default=None, help="Pool to run on (default: the training pool).")
    parser.add_argument("--timeout", default=MIGRATE_TIMEOUT, help="Wall-clock ceiling.")


def _flags(args: argparse.Namespace) -> tuple[str, ...]:
    """The sweep's own command line, carried verbatim on `eval_flags`.

    ONE `--runs` WITH MANY VALUES, not one flag per run. `--runs` takes
    `nargs="*"`, so a repeated flag keeps only the LAST -- `--runs a --runs b`
    parses to `["b"]`, and the sweep would migrate one run of two and report
    success.
    """
    flags: list[str] = []
    if args.verify:
        flags.append("--verify")
    if args.recover_tars:
        flags.append("--recover-tars")
    if args.limit:
        flags += ["--limit", str(args.limit)]
    if args.runs:
        flags += ["--runs", *args.runs]
    return tuple(flags)


class SubmitMigratePayload(dispatch.Dispatched):
    """A dispatch, plus the sweep's own command line."""

    op: Literal["submit-migrate"] = "submit-migrate"
    flags: list[str] = Field(default_factory=list)


def run(args: argparse.Namespace) -> SubmitMigratePayload:
    """Queue exactly one migration sweep."""
    flags = _flags(args)
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=TaskName.MIGRATE_CHECKPOINTS,
                eval_flags=flags,
                timeout=args.timeout,
            )
        ],
        pool=args.pool or "train",
    )
    return payload.extend(SubmitMigratePayload, flags=list(flags))


def render(payload: SubmitMigratePayload) -> None:
    dispatch.render_queued(payload)
    if payload.flags:
        print(f"  flags:         {' '.join(payload.flags)}")
    print("\nIdempotent: re-dispatch to continue where this one stopped.")
    print("Watch it with:  poker-solver logs --task <id>")


COMMAND = Command(
    name="submit-migrate",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Run the share -> container checkpoint migration on a pool node.",
)
