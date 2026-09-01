"""Reads against the record database.

SQL rather than a builder wherever the query IS the idea -- a join, an
aggregate, a window. Single-table access goes through the models, where a
mistyped column is an attribute error rather than a runtime one.

These return plain rows. Turning them into the models a surface renders is the
COMMAND's job, because the model lives in `pipeline` and an adapter may not
import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    with engine.connect() as connection:
        return connection.execute(_RUNS, {"limit": limit}).all()
