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

# One statement, and the conflict target is the leg's own key. `progress` is
# OVERWRITTEN as a task runs, so the row has to move or the bar it feeds freezes
# at whatever the first sample said.
_UPSERT = """
    INSERT INTO legs (task_id, attempt, leg, run_id, at, body)
    VALUES (%(task_id)s, %(attempt)s, %(leg)s, %(run_id)s, %(at)s, %(body)s)
    ON CONFLICT (task_id, attempt, leg)
    DO UPDATE SET body = EXCLUDED.body, at = EXCLUDED.at, run_id = EXCLUDED.run_id
"""

# ONE STATEMENT, and it is what makes attempt numbering safe without a lock.
# `MAX(attempt) + 1` is computed inside the INSERT, so two writers racing both
# derive the same number and the PRIMARY KEY on (task_id, attempt, leg) makes
# one of them lose -- loudly, as a unique violation, rather than by overwriting
# the record of the failure that caused the retry. Batch retries one task
# sequentially, so the race is theoretical; the constraint is what makes it not
# matter.
# EVERY PARAMETER IS CAST, and that is not decoration. `task_id` appears twice
# -- once selected, once compared against the column -- and Postgres deduced
# `text` from one and `character varying` from the other, refusing the statement
# with `AmbiguousParameter` rather than guessing. Measured on a node: three
# retries, each exiting 44 in seconds, having recorded nothing. Nothing but a
# real server can catch this; the fake driver a unit test uses parses no SQL.
_CLAIM = """
    INSERT INTO legs (task_id, attempt, leg, run_id, at, body)
    SELECT %(task_id)s::varchar,
           COALESCE(MAX(attempt), 0) + 1,
           %(leg)s::varchar, %(run_id)s::varchar, %(at)s::timestamptz, %(body)s::jsonb
      FROM legs WHERE task_id = %(task_id)s::varchar AND attempt >= 0
    RETURNING attempt
"""

# `attempt >= 0` in both, and it is not defensive noise: a progress row is
# stored at `TASK_SCOPED` (-1), which is a real value in this column. Today the
# start row always lands first so MAX is never negative, but that is an argument
# about ORDERING, and the numbering of every subsequent record hangs on it. The
# predicate makes it an argument about the DATA instead.
_LATEST = """
    SELECT COALESCE(MAX(attempt), 0) FROM legs
     WHERE task_id = %(task_id)s AND attempt >= 0
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


def claim_attempt(task_id: str, document: Mapping[str, Any], *, dsn: str) -> int:
    """Reserve this attempt's number by WRITING the start row. RAISES.

    The one place in the node that is allowed to fail, and it has to be: the
    number it returns names the attempt every later record of this task belongs
    to, and inventing one on a failed write means the exit record and the
    progress samples overwrite a PREVIOUS attempt -- destroying the account of
    the failure that caused this retry, on what is now the only copy.

    It used to be counted by listing `.start.json` files on the share. Nothing
    writes those, so the count is the database's now, and the claim is the same
    statement as the write: `MAX + 1` inside the INSERT, refereed by the primary
    key. Fail-fast is not a new failure mode -- since the record moved, a task
    that cannot reach the database fails at its first training event anyway;
    this names the real reason minutes earlier.
    """
    row = task_log.leg_row(task_id, 0, "start", document)
    import psycopg  # noqa: PLC0415 -- see the module docstring

    with (
        psycopg.connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(_CLAIM, {**row, "body": json.dumps(row["body"], default=str)})
        claimed = cursor.fetchone()
        connection.commit()
    if not claimed:
        raise RuntimeError(f"could not claim an attempt number for {task_id}")
    return int(claimed[0])


def latest_attempt(task_id: str, *, dsn: str) -> int:
    """The attempt this task's terminal record belongs to. RAISES.

    Derived rather than carried: the exit trap may have lost anything the entry
    point computed, which is why this was a directory listing before and is a
    query now. 0 when nothing was ever claimed -- a task that died before its
    start row landed, which has no attempt to belong to.
    """
    import psycopg  # noqa: PLC0415 -- see the module docstring

    with (
        psycopg.connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(_LATEST, {"task_id": task_id})
        found = cursor.fetchone()
    return int(found[0]) if found else 0


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
