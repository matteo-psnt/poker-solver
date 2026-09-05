"""Storing what a task said about itself.

The node writes these to the share as files and CANNOT write them here: the
wrapper runs on the pool's bootstrap interpreter, not the venv `uv sync` builds,
so the driver is not on its path -- before or after the sync. It reaches this
code by running `poker-solver mirror-legs`, which is the same way it runs the
task itself.

Upsert, not insert-or-ignore. A task's `progress` leg is OVERWRITTEN as it runs:
the row has to move, or the bar it feeds freezes at whatever the first sample
said -- which is the exact failure that held the `tasks` reader on the share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.dialects.postgresql import insert

from src.adapters.postgres import models

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# When a leg happened, from whichever field its writer used. Not one field,
# because the writers are different programs: the node stamps `ts`, while
# `write_observed_document` stamps `observed_at` -- it is the READER saying when
# IT looked, not the node saying when something happened. Batch's own times come
# last so a record with neither is not dropped for want of a clock.
_INSTANT_FIELDS = ("ts", "observed_at", "end_time", "start_time")


def leg_values(task_id: str, attempt: int, leg: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """One leg document as the columns `legs` holds."""
    return {
        "task_id": task_id,
        "attempt": attempt,
        "leg": leg,
        "run_id": document.get("run_id") or None,
        "at": next((document[f] for f in _INSTANT_FIELDS if document.get(f)), None),
        "body": dict(document),
    }


def record_legs(engine: Any, rows: Sequence[tuple[str, int, str, dict[str, Any]]]) -> int:
    """Store leg documents, newest content winning. Returns how many were written."""
    if not rows:
        return 0
    values = [leg_values(*row) for row in rows]
    excluded = insert(models.Leg).excluded
    with engine.begin() as connection:
        connection.execute(
            insert(models.Leg)
            .values(values)
            .on_conflict_do_update(
                index_elements=["task_id", "attempt", "leg"],
                set_={"body": excluded.body, "at": excluded.at, "run_id": excluded.run_id},
            )
        )
    return len(values)
