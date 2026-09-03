"""Write a leg document the way the node used to, for tests that READ them.

The node writes no files any more -- its account is rows -- but the file join
is still live for the legacy corpus on the share and for `--tasks-dir`, so the
tests that cover it need something to read. This builds the record through the
same `task_log.node_record` production uses and lays it out under the old
naming, so a fixture cannot drift from the shape the reader expects.
"""

from __future__ import annotations

from typing import Any

from src.shared import records
from src.shared.cloudtask import task_log


def write_leg(
    share, task_id: str = "", event: str = "", cause: str | None = None, **fields: Any
) -> dict:
    """One node record, on disk, in the layout `read_tasks` joins over.

    The attempt is counted from the start records already there -- which is what
    the node itself did before the number became a database claim. Kept here
    because a retry's fixture has to land beside the failed attempt rather than
    on top of it, which is the case these tests exist to cover.
    """
    directory = task_log.tasks_dir(share)
    directory.mkdir(parents=True, exist_ok=True)
    starts = len(list(directory.glob(f"{task_id}.*{task_log.START_SUFFIX}")))
    attempt = starts + 1 if event == task_log.EVENT_STARTED else max(starts, 1)
    record = task_log.node_record(
        task_id=task_id, attempt=attempt, event=event, cause=cause, **fields
    )
    suffix = task_log.START_SUFFIX if event == task_log.EVENT_STARTED else task_log.EXIT_SUFFIX
    records.write_snapshot(
        directory / f"{task_id}.{attempt}{suffix}", record, records.REGISTRY[f"legs/*{suffix}"]
    )
    return record
