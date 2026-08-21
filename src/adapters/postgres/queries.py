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


# No ORDER BY: the join keys off REBUILT FILENAMES and sorts them itself, which
# is what decides ties between two records claiming one slot. Ordering here
# would be a second, silently different, opinion about that.
_LEGS = sa.text("SELECT task_id, attempt, leg, body FROM legs")


def leg_rows(engine: Any) -> list[tuple[str, int, str, dict[str, Any]]]:
    """Every leg document, as `(task_id, attempt, leg, body)`.

    Whole rather than paged: the join needs every row to answer about any one of
    them -- `kinds.etas` estimates from the whole population, and which attempt
    of a task is latest is only knowable by seeing all of them.
    """
    with _read(engine) as connection:
        return [(row[0], row[1], row[2], row[3]) for row in connection.execute(_LEGS)]


def observed_legs(engine: Any) -> dict[str, dict[str, Any]]:
    """The stored Batch observation for each task, by task id."""
    with _read(engine) as connection:
        return {
            row[0]: row[1]
            for row in connection.execute(
                sa.text("SELECT task_id, body FROM legs WHERE leg = 'observed'")
            )
        }


# ORDER BY the recorded instant, which is what `read_records` sorts on: rows
# written by different machines arrive interleaved, so insertion order is
# "whose write landed last" and not "when the eval happened".
_EVALS = sa.text("SELECT payload FROM evals ORDER BY recorded_at, eval_id")

# The same order, for one run. `curve_series` and the unplaceable count both
# skip every record whose `run_id` is not the one asked about, so restricting
# here is the same answer -- and it is 2,238 documents against a handful.
_EVALS_FOR_RUN = sa.text("""
    SELECT payload FROM evals WHERE run_id = :run_id ORDER BY recorded_at, eval_id
""")


def eval_records(engine: Any, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every evaluation, as the DOCUMENT the readers already parse.

    `payload` is the whole document, so this returns what `read_records` returns
    and the filtering, tiering and curve maths above it are untouched. Tiering
    especially: it is ~30 knobs deep and a query that re-expressed it would be a
    second answer to which evals may be compared -- a five-column version once
    reported -100.0 mbb where the truth was -60.0.
    """
    with _read(engine) as connection:
        if run_id is None:
            return [row[0] for row in connection.execute(_EVALS)]
        return [row[0] for row in connection.execute(_EVALS_FOR_RUN, {"run_id": run_id})]


# The listing's filters are COLUMNS, so the page is cut here and the document
# ships only for the rows shown: `ledger` was pulling every eval -- 7.5 MB, 3 s
# on the wire from the laptop -- to print 25. `experiment_id` is the one filter
# the schema does not index; it stays an expression on the document rather
# than becoming a column for one reader. The window count is what the filters
# matched before the page, in the same statement and so the same snapshot.
# NULLIF turns a limit of 0 into no limit, which is what `--limit 0` means.
_LEDGER_PAGE = sa.text("""
    SELECT payload, count(*) OVER () AS matched
      FROM evals
     WHERE (CAST(:run_id AS text) IS NULL OR run_id = :run_id)
       AND (CAST(:method AS text) IS NULL OR method = :method)
       AND (CAST(:experiment_id AS text) IS NULL OR payload->>'experiment_id' = :experiment_id)
     ORDER BY recorded_at DESC, eval_id DESC
     LIMIT NULLIF(CAST(:limit AS integer), 0)
""")


def ledger_page(
    engine: Any,
    *,
    run_id: str | None,
    method: str | None,
    experiment_id: str | None,
    limit: int,
) -> tuple[int, list[dict[str, Any]]]:
    """The newest `limit` matching evaluations, oldest first, and how many matched."""
    with _read(engine) as connection:
        rows = connection.execute(
            _LEDGER_PAGE,
            {"run_id": run_id, "method": method, "experiment_id": experiment_id, "limit": limit},
        ).all()
    if not rows:
        return 0, []
    return int(rows[0][1]), [row[0] for row in reversed(rows)]


_SCORED = sa.text("""
    SELECT run_id, checkpoint_iteration FROM evals WHERE checkpoint_iteration IS NOT NULL
""")


def scored_rungs(engine: Any) -> dict[str, set[int]]:
    """Rungs an eval NAMES, per run. Deleting one makes its score unreproducible.

    Two columns rather than the document, because the only caller asks whether a
    rung may be deleted, not what it scored -- and that answer must not depend on
    a payload shape.
    """
    found: dict[str, set[int]] = {}
    with _read(engine) as connection:
        for run_id, iteration in connection.execute(_SCORED):
            found.setdefault(str(run_id), set()).add(int(iteration))
    return found


_LADDER = sa.text("""
    SELECT iteration FROM checkpoints WHERE run_id = :run_id ORDER BY iteration
""")


def rung_ladder(engine: Any, run_id: str) -> list[int]:
    """The retained rungs of one run, oldest first.

    What the checkpoint manifest names on the share. A curve reads it to say
    which rungs have NO eval -- a curve with holes and a curve that stops early
    look identical once plotted.
    """
    with _read(engine) as connection:
        return [int(row[0]) for row in connection.execute(_LADDER, {"run_id": run_id})]


# ORDER BY the event's OWN timestamp, then arrival. `gseq` alone is arrival
# order and two processes write one run's events -- the node wrapper and the
# trainer -- so a fold keyed on arrival can see an attempt end before it began.
_RUN_EVENTS = sa.text("""
    SELECT event, body FROM run_events WHERE run_id = :run_id ORDER BY at, gseq
""")


def run_event_bodies(engine: Any, run_id: str) -> list[dict[str, Any]]:
    """One run's events, as the LINES the log holds.

    `event` comes back merged into the body. It is a column here and a key
    there, and the fold looks it up by key -- so a body handed over as stored
    has no `created` event in it and a resume dies looking for one. Measured on
    a node: `ValueError: run log has no 'created' event`.
    """
    with _read(engine) as connection:
        return [
            {**(body or {}), "event": event}
            for event, body in connection.execute(_RUN_EVENTS, {"run_id": run_id})
        ]
