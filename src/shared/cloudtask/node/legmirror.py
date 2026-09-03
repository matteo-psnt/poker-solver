"""Storing this task's own records in the database, from the wrapper itself.

The wrapper runs on the interpreter the pool's START TASK installs, not the venv
`uv sync` builds -- so what it can import is whatever that start task put there.
It installs `psycopg` beside the interpreter for exactly this, which is why this
is a function call rather than the `uv run poker-solver mirror-legs` subprocess
it replaces.

THE IMPORT IS LAZY AND GUARDED, and that is the whole safety argument. The node
loads this package BEFORE `uv sync`, and a module-level third-party import that
is missing kills the task at bootstrap -- before it can write the record that
would explain it, which `tasks` then reports as nothing at all. Imported inside
the function and caught, a node whose start task did not install the driver
mirrors nothing, which is the same as having no DSN and is already the policy.

Raw SQL rather than the ORM for the same reason: `shared` may not import
`adapters`, and the row shape both sides insert comes from `task_log.leg_row`
so there is still only one of it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.shared.cloudtask import task_log

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

# One statement, and the conflict target is the leg's own key. `progress` is
# OVERWRITTEN as a task runs, so the row has to move or the bar it feeds freezes
# at whatever the first sample said.
_UPSERT = """
    INSERT INTO legs (task_id, attempt, leg, run_id, at, body)
    VALUES (%(task_id)s, %(attempt)s, %(leg)s, %(run_id)s, %(at)s, %(body)s)
    ON CONFLICT (task_id, attempt, leg)
    DO UPDATE SET body = EXCLUDED.body, at = EXCLUDED.at, run_id = EXCLUDED.run_id
"""

# Short on purpose. Nothing waits on the answer, and the alternative to failing
# fast is holding the watcher thread while a task's work is what matters.
CONNECT_TIMEOUT_SECONDS = 15


def record(
    task_id: str,
    attempt: int,
    leg: str,
    document: Mapping[str, Any],
    *,
    dsn: str,
    log: Callable[[str], None] | None = None,
) -> None:
    """Store ONE record, at the moment it is made. NEVER FATAL.

    The direct half. :func:`publish` re-reads this task's files and upserts
    whatever it finds, which is self-healing and is why it stays -- but it can
    only heal a record that has a FILE, and progress is about to stop having
    one. Writing the row where the record is made covers what the re-read
    cannot, and costs one statement rather than a directory listing.
    """
    if not dsn:
        return
    try:
        _store(dsn, [task_log.leg_row(task_id, attempt, leg, document)])
    except Exception as exc:  # noqa: BLE001 -- a task must not die recording itself
        if log:
            log(f"leg {leg} not recorded: {type(exc).__name__}: {exc}".strip()[:200])


def publish(
    share: Path, task_id: str, *, dsn: str, log: Callable[[str], None] | None = None
) -> None:
    """Mirror this task's leg documents. NEVER FATAL.

    Reads the task's CURRENT records rather than a delta, so a call that fails
    is repaired by the next one instead of losing a document forever. An empty
    `dsn` writes nothing: that is the pre-migration behaviour, and the rollback.
    """
    if not dsn:
        return
    try:
        rows = [
            task_log.leg_row(*row)
            for row in task_log.rows_from_documents(
                task_log.read_task_documents(task_log.tasks_dir(share), task_id)
            )
        ]
        if rows:
            _store(dsn, rows)
    except Exception as exc:  # noqa: BLE001 -- the share has the record; this is the copy
        if log:
            log(f"legs not mirrored: {type(exc).__name__}: {exc}".strip()[:200])


def _store(dsn: str, rows: list[dict[str, Any]]) -> None:
    """Write the rows, importing the driver only if it is actually needed.

    `import psycopg` is INSIDE the function deliberately: see the module
    docstring. At module scope a node without it would die at bootstrap having
    recorded nothing.
    """
    import psycopg  # noqa: PLC0415 -- see the module docstring

    with psycopg.connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                _UPSERT,
                # `body` is a column of JSONB and psycopg will not infer that
                # from a dict; serialising here keeps the driver out of the row
                # shape, which `task_log.leg_row` owns for both writers.
                [{**row, "body": json.dumps(row["body"], default=str)} for row in rows],
            )
        connection.commit()
