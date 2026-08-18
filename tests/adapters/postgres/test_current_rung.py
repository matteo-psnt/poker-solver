"""Only one rung is current, and it is the one the run last claimed.

`is_current` is a PARTIAL UNIQUE index -- one row per run where the flag is
set -- so every writer that sets it must clear the previous holder in the same
transaction. Two writers got this wrong in opposite ways, and neither failed
where anyone would see it:

* `sink.claim` RAISED on a run's second checkpoint, and `claim` is allowed to
  raise, so dual-write would have killed the run. Invisible until then, because
  a run with one rung works.
* the importer (since deleted) used a bare `on_conflict_do_nothing()`, which ignores a
  conflict on ANY index -- so the new current rung was silently dropped and the
  old one kept the flag. Three runs pointed 2,000 iterations behind the share,
  and a stale pointer RESOLVES, so nothing failed at all.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa

from src.adapters.postgres import models
from src.adapters.postgres.sink import PostgresSink


class _Engine:
    """Records the statements a claim issues, in order."""

    def __init__(self) -> None:
        self.statements: list[Any] = []

    def begin(self) -> _Engine:
        return self

    def __enter__(self) -> _Engine:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any) -> None:
        self.statements.append(statement)


def _claim() -> list[Any]:
    engine = _Engine()
    sink = PostgresSink(engine)
    sink.claim("run-a", 2000, "rungs/run-a/2000")
    return engine.statements


def _sql(statement: Any) -> str:
    """The statement as Postgres receives it.

    Compiled rather than introspected: `_post_values_clause` is SQLAlchemy's
    private shape and changed under this test once already, while the SQL is the
    thing the database actually enforces against.
    """
    from sqlalchemy.dialects import postgresql

    return str(statement.compile(dialect=postgresql.dialect())).replace("\n", " ")


def test_a_claim_clears_the_previous_current_rung_first():
    """The ORDER is the fix. Insert-then-clear either trips the same index or
    leaves the run pointing at a stale rung, which is worse than an error."""
    clear, insert = _claim()
    assert clear.is_update, "a claim must clear the old current rung"
    assert insert.is_insert


def test_the_clear_is_scoped_to_this_run():
    """`WHERE is_current` alone would unset every OTHER run's pointer too."""
    clear, _ = _claim()
    assert "run_id" in _sql(clear)
    assert "is_current" in _sql(clear)


def test_reclaiming_an_earlier_rung_moves_the_flag_back():
    """A resume points the run at where it restarts from, so the upsert has to
    set `is_current`, not only `blob_uri`. Verified against the real database:
    claiming 1000, 2000, 3000 leaves 3000 current, and re-claiming 2000 moves it
    back to 2000."""
    _, insert = _claim()
    conflict = _sql(insert).split("ON CONFLICT", 1)[1]
    assert "is_current" in conflict, "a re-claim that keeps the old flag is the stale pointer"


def test_the_conflict_target_is_the_rung_not_any_index():
    """A bare `on_conflict_do_nothing()` ignores conflicts on EVERY index, which
    is how the stale pointer survived an import that reported success."""
    conflict = _sql(_claim()[1]).split("ON CONFLICT", 1)[1]
    assert "run_id" in conflict
    assert "iteration" in conflict
    assert "DO UPDATE" in conflict, "DO NOTHING here drops the new current rung"


def test_the_checkpoint_table_still_declares_the_partial_index():
    """The tests above are only meaningful while this is the constraint."""
    index = next(
        arg
        for arg in models.Checkpoint.__table_args__
        if isinstance(arg, sa.Index) and "current" in str(arg.name)
    )
    assert index.unique
    assert index.dialect_options["postgresql"]["where"] is not None
