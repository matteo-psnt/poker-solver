"""Writing back what Batch says happened to a task.

A READER'S write, and that is what separates it from `legmirror`: the node
records what it did, while this records what BATCH says happened to a task the
node never got to explain -- an OOM kill, a node loss, a wall-clock stop. The
reader is the only thing that can ask.

It exists because that observation has to reach BOTH stores. The share is the
source of truth and keeps its file; the database gets the same document, or the
next read re-asks Batch about a task the share can already explain, forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.dialects.postgresql import insert

from src.adapters.postgres import models
from src.shared.cloudtask import task_log

if TYPE_CHECKING:
    from collections.abc import Mapping

# `observed` is per TASK, not per attempt: Batch describes only a task's latest
# attempt, so there is one row per task and a retry OVERWRITES it.
_LEG = "observed"


def record_observations(engine: Any, documents: Mapping[str, dict[str, Any]]) -> int:
    """Store one Batch observation per task id. Returns how many were written.

    Rows come from `task_log.leg_row`, the same builder the node and the
    importer use, so an observation written here and the same one re-imported
    from the share are one row.
    """
    if not documents:
        return 0
    rows = [
        task_log.leg_row(task_id, task_log.TASK_SCOPED, _LEG, document)
        for task_id, document in documents.items()
    ]
    excluded = insert(models.Leg).excluded
    with engine.begin() as connection:
        connection.execute(
            insert(models.Leg)
            .values(rows)
            .on_conflict_do_update(
                index_elements=["task_id", "attempt", "leg"],
                set_={"body": excluded.body, "at": excluded.at},
            )
        )
    return len(rows)
