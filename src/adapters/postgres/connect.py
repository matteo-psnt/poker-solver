"""Building a sink from the environment, in the one place allowed to decide.

Separate from `sink.py` because construction is a different question from
behaviour: the sink knows how to write, this knows whether there is anywhere to
write to. Commands call this; nothing in `pipeline` can.
"""

from __future__ import annotations

import contextlib
import logging
import os
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.shared.ports.record import EvalSink, RecordSink, RecordSource

log = logging.getLogger(__name__)

DSN_ENV = "POKER_SOLVER_RECORD_DSN"


class NoRecordError(RuntimeError):
    """The record database is not configured, and nothing here guesses.

    A plain `RuntimeError` because `adapters` may not import the command
    layer's `CommandError`; both surfaces classify this one by name.
    """

    def __init__(self) -> None:
        super().__init__(
            f"No record database: {DSN_ENV} is unset and the store state could not "
            "be read. Apply infra/store, or export the DSN of the server to read."
        )


# A console view fans out over its panels at `max_workers=len(parts)`, so the
# pool has to be at least as wide as the widest screen or the fan-out serialises
# at one round trip each -- the exact failure `_compose.compose` warns about,
# and invisible, because the screen still renders.
READER_POOL = 8


# One engine per (DSN, role) FOR THE LIFE OF THE PROCESS. A SQLAlchemy engine
# owns a connection pool and is meant to be long-lived; building one per call
# throws the pool away and pays a fresh TLS handshake and SCRAM auth every time.
# Measured against Sweden from here: 3,806 ms per call rebuilding it against
# 701 ms reusing it, which was the whole of a listing's cost and had nothing to
# do with the query -- that runs in 2.8 ms server-side.
@cache
def _engine(dsn: str, pre_ping: bool, pool_size: int) -> Any:
    import sqlalchemy as sa  # noqa: PLC0415 -- only when a DSN says to

    return sa.create_engine(
        dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        pool_size=pool_size,
        max_overflow=pool_size,
        pool_pre_ping=pre_ping,
        # A firewall that drops the SYN looks like a hang without this; with it,
        # an unreachable server is a ten-second refusal `unreachable_hint` names.
        connect_args={"connect_timeout": 10},
        # Well inside Azure's idle timeout. A trainer holds this open for hours
        # between bursts, and a stale connection surfaces as a lost batch rather
        # than an error anyone sees.
        pool_recycle=280,
    )


def unreachable_hint(error: BaseException) -> str | None:
    """The one connection failure with a known cause, named for the operator.

    Checked by name so the surfaces that render it need not import the driver:
    a connect timeout against this server has meant, every time so far, that
    the laptop's address rotated out of the firewall rule.
    """
    if type(error).__name__ != "OperationalError":
        return None
    text = str(error).lower()
    if "timeout" not in text and "timed out" not in text:
        return None
    return (
        "The record database did not answer. Most likely this machine's public IP "
        "rotated out of its firewall rule: `poker-solver record-admit` re-admits it."
    )


def engine_for(dsn: str) -> Any:
    """A reader's engine for an explicit DSN -- the migration path, which runs
    before the environment is trusted."""
    return _engine(dsn, False, READER_POOL)


def engine_from_environment(*, pre_ping: bool = False) -> Any:
    """The record engine, or `NoRecordError`. Never a silent second answer.

    One answer to "is there a database", and one place that knows how to reach
    it -- but TWO engines, because the halves want opposite things. A reader
    wants width and no ping: it fans out across a screen's panels and uses a
    connection the moment it takes one. The sink wants one connection and a
    ping: it has a single writer thread, and its connection sits idle for hours
    between bursts, where a stale socket surfaces as a lost batch rather than as
    an error anyone sees. `pre_ping` is a FULL ROUND TRIP -- 175 ms from here,
    because the server is in Sweden and this is not.
    """
    dsn = os.environ.get(DSN_ENV, "").strip()
    if not dsn:
        raise NoRecordError
    return _engine(dsn, pre_ping, 1 if pre_ping else READER_POOL)


def sink_from_environment() -> RecordSink:
    """The writer's sink. A DSN that is set but unusable is NOT swallowed: it
    means someone intended to record and is not, which they need to be told at
    dispatch rather than discover in a query later."""
    engine = engine_from_environment(pre_ping=True)

    from src.adapters.postgres import schema  # noqa: PLC0415 -- alembic, only for a writer
    from src.adapters.postgres.sink import PostgresSink  # noqa: PLC0415

    schema.assert_current(engine)

    log.info("record sink attached")
    return PostgresSink(engine)


def eval_sink_from_environment() -> EvalSink:
    """Reuses the SINK engine, not the reader's: one connection with a pre-ping,
    which is what a process that scores for hours and then writes once needs."""
    engine = engine_from_environment(pre_ping=True)

    from src.adapters.postgres import schema  # noqa: PLC0415 -- alembic, only for a writer
    from src.adapters.postgres.evals import PostgresEvalSink  # noqa: PLC0415

    schema.assert_current(engine)

    return PostgresEvalSink(engine)


# The queue is nearly empty by the end of a run; this is a ceiling on a stall,
# not a budget. Long enough that a slow last batch still lands.
FLUSH_TIMEOUT_SECONDS = 30.0


@contextlib.contextmanager
def record_sink() -> Iterator[RecordSink]:
    """A sink for the duration, DRAINED on the way out.

    The drain is the point. `emit` queues and a background thread writes, and
    that thread is a daemon -- so whatever is still queued when the process
    exits dies with it. Nothing called `flush`, and the events lost were the
    LAST ones: measured on three node runs, every one of them reached the
    database without its `checkpoint` or its `status`. A missing terminal status
    is a finished run that goes on advertising itself as training, which is the
    zombie this migration exists to stop creating.

    A lossy drain is LOGGED, never raised. By this point the training has
    succeeded and the share has the whole record; the database being behind is
    something to say, not something to fail a run over.
    """
    sink = sink_from_environment()
    try:
        yield sink
    finally:
        if not sink.flush(FLUSH_TIMEOUT_SECONDS):
            log.warning("record sink dropped events; the database is behind the share")


def record_source_from_environment() -> RecordSource:
    """The READER'S engine, not the sink's: this answers one question at the
    start of a task and wants the pool the readers use, not the single
    pre-pinged connection a writer holds open for hours."""
    engine = engine_from_environment()

    from src.adapters.postgres.source import PostgresRecordSource  # noqa: PLC0415

    return PostgresRecordSource(engine)
