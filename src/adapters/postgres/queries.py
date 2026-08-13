"""Reads against the record database.

SQL rather than a builder wherever the query IS the idea -- a join, an
aggregate, a window. Single-table access goes through the models, where a
mistyped column is an attribute error rather than a runtime one.

These return plain rows. Turning them into the models a surface renders is the
COMMAND's job, because the model lives in `pipeline` and an adapter may not
import it.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@contextlib.contextmanager
def _read(engine: Any) -> Iterator[Any]:
    """A connection for ONE STATEMENT, outside a transaction.

    SQLAlchemy opens a transaction on first execute and rolls it back on close,
    so a plain `engine.connect()` spends three round trips on a statement that
    needs one. Against Sweden that is 554 ms to answer `SELECT 1` where the
    round trip is 175 ms.

    The one statement is the CONDITION, not a coincidence: under AUTOCOMMIT two
    statements see two snapshots, so a reader that asked for evals and then for
    the rung ladder could pair an eval against a ladder that moved between the
    round trips. A read that genuinely needs two gets a transaction and pays for
    it -- it does not get a waiver here.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        yield connection


# PAGE FIRST, THEN ENRICH. Measured at 10x the current record: a lateral over
# every run and then LIMIT is 52 ms; restricting to the page first and enriching
# only those rows is 4 ms. The naive form does the per-run work for 3,000 runs
# to return 200 of them.
_RUNS = sa.text("""
    WITH page AS (
        SELECT run_id, config_name, status, iterations, num_infosets,
               experiment_id, arm, git_commit, git_dirty, started_at
          FROM runs
         ORDER BY started_at DESC
         LIMIT :limit
    )
    -- `is_current`, NOT merely "has a rung". The share answers this by asking
    -- whether STATIC_CHECKPOINT.json exists, and that manifest is what NAMES
    -- the current rung -- so a run whose manifest is gone has snapshots nothing
    -- can resolve and is correctly unloadable. One run on the share is exactly
    -- that: three complete rungs, three markers, no manifest. Counting any rung
    -- reported it as loadable when a loader would start it from zero.
    --
    -- LEFT JOIN over a filtered set rather than a correlated EXISTS per row:
    -- the subquery form re-probes `checkpoints` once for every run in the page.
    SELECT p.*, (h.run_id IS NOT NULL) AS has_checkpoint
      FROM page p
      LEFT JOIN (SELECT run_id FROM checkpoints WHERE is_current) h USING (run_id)
     ORDER BY p.started_at DESC
""")


def describe_runs(engine: Any, *, limit: int = 1000) -> Sequence[Any]:
    """Every run, newest first, with whether it holds a checkpoint.

    `limit` is not a nicety: unbounded, this is the one query here whose cost
    grows with history rather than with the answer.
    """
    with _read(engine) as connection:
        return connection.execute(_RUNS, {"limit": limit}).all()


def run_ids(engine: Any) -> list[str]:
    """Every published run id.

    Whole rather than filtered in SQL: what a fragment identifies is decided by
    `interfaces.run_names.matching`, and a `LIKE` here would be a second
    implementation of that rule -- one that also has to think about `%` and `_`
    in what the user typed. 303 ids is one round trip and a few kilobytes; the
    rule stays where both surfaces already read it.
    """
    with _read(engine) as connection:
        return [row[0] for row in connection.execute(sa.text("SELECT run_id FROM runs"))]


# ORDER BY iteration, not `gseq`. `gseq` is arrival order and two processes
# write one run's events -- the node wrapper and the trainer -- while iteration
# is what the series MEANS. They agree across all 276 runs today and there is no
# duplicate (run, iteration) in the record, so this orders the same rows; it
# just does not depend on that staying true.
_CHECKPOINTS = sa.text("""
    SELECT body
      FROM run_events
     WHERE run_id = :run_id AND event = 'checkpoint'
     ORDER BY (body->>'iteration')::bigint, gseq
""")


def checkpoint_series(engine: Any, run_id: str) -> list[dict[str, Any]]:
    """One run's per-checkpoint events, oldest first.

    The stored body IS the event the share appends to `run.jsonl`, so what comes
    back here is what the file path parses -- same keys, same absences. A row
    written by an older version genuinely lacks fields a newer one carries, and
    that survives the crossing rather than being filled in.
    """
    with _read(engine) as connection:
        return [row[0] for row in connection.execute(_CHECKPOINTS, {"run_id": run_id})]
