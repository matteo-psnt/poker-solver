"""Removing a run from the record: the one write the laptop performs on `runs`.

Everything a run owns hangs off its row by `ON DELETE CASCADE` -- events,
checkpoints, evals -- so one statement takes the record's whole claim on the
run. `progress` is keyed by scope and subject rather than by foreign key and
is swept in the same transaction. The task log (`legs`) is NOT touched: it is
the account of node time the cost screen is built from, and a task that ran
is a fact about the pool whether or not its run is still worth keeping.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa

_FORGET_RUN = sa.text("DELETE FROM runs WHERE run_id = :run_id")
_FORGET_PROGRESS = sa.text("DELETE FROM progress WHERE scope = 'run' AND subject_id = :run_id")


def forget_run(engine: Any, run_id: str) -> bool:
    """Delete one run and everything the record holds about it. False if absent."""
    with engine.begin() as connection:
        connection.execute(_FORGET_PROGRESS, {"run_id": run_id})
        result = connection.execute(_FORGET_RUN, {"run_id": run_id})
        return bool(result.rowcount)
