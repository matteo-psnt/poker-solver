"""The `mirror-legs` subcommand: put a task's own records into the database.

**Run BY THE NODE, on the node, about itself.** It exists because the wrapper
that writes those records cannot write them here: `infra/run_task.py` runs on
the pool's bootstrap interpreter rather than the venv `uv sync` builds, so the
database driver is not on its path -- and the first record is written before
that sync happens at all. Shelling out to this is the shortest path to a process
that has the driver, and it is the same way the wrapper runs the task itself.

Scoped to ONE task deliberately. The node knows which task it is, mirroring its
own handful of documents costs one round trip, and a node that swept the whole
directory would race every other node doing the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.adapters.postgres import connect, legs
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared import task_history
from src.shared.cloudtask import task_log

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver mirror-legs`."""
    parser.add_argument("--task", required=True, help="The task whose records to mirror.")
    parser.add_argument(
        "--legs-dir",
        required=True,
        help="Directory holding the leg documents -- the node's share mount.",
    )


class MirrorLegsPayload(BaseModel):
    """What was mirrored, so a node log says so rather than implying it."""

    op: Literal["mirror-legs"] = "mirror-legs"
    task_id: str
    documents: int
    """Zero when no DSN is set. NOT an error: dual-write is opt-in, and a task
    dispatched without one writes files only, which is the pre-migration
    behaviour and the rollback."""
    written: int
    enabled: bool


def run(args: argparse.Namespace) -> MirrorLegsPayload:
    """Mirror one task's leg documents into the database."""
    directory = Path(args.legs_dir)
    if not directory.is_dir():
        raise CommandError(f"No such legs directory: {directory}")

    # SCOPED BY NAME. Reading the directory and filtering afterwards reads and
    # parses all 13,600 documents on the share to find two, which cost a node
    # ~100s per call and put three and a half minutes on the end of a task whose
    # training took twenty seconds.
    rows = task_history.rows_from_documents(task_log.read_task_documents(directory, args.task))

    engine = connect.engine_from_environment(pre_ping=True)
    if engine is None:
        return MirrorLegsPayload(task_id=args.task, documents=len(rows), written=0, enabled=False)
    return MirrorLegsPayload(
        task_id=args.task,
        documents=len(rows),
        written=legs.record_legs(engine, rows),
        enabled=True,
    )


def render(payload: MirrorLegsPayload) -> None:
    if not payload.enabled:
        print(f"mirror-legs {payload.task_id}: no DSN set, {payload.documents} document(s) skipped")
        return
    print(f"mirror-legs {payload.task_id}: {payload.written} document(s) written")


COMMAND = Command(
    name="mirror-legs",
    help="Put one task's own records into the database (the node runs this about itself).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
