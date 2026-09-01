"""Building a sink from the environment, in the one place allowed to decide.

Separate from `sink.py` because construction is a different question from
behaviour: the sink knows how to write, this knows whether there is anywhere to
write to. Commands call this; nothing in `pipeline` can.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.ports.record import RecordSink

log = logging.getLogger(__name__)

DSN_ENV = "POKER_SOLVER_RECORD_DSN"


def engine_from_environment() -> Any | None:
    """An engine when a DSN is set, `None` when it is not.

    Shared by the sink and the readers so there is one answer to "is there a
    database", and one place that knows how to build a connection to it.
    """
    dsn = os.environ.get(DSN_ENV, "").strip()
    if not dsn:
        return None

    import sqlalchemy as sa  # noqa: PLC0415 -- only when a DSN says to

    return sa.create_engine(
        dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        # One connection, recycled well inside Azure's idle timeout. A trainer
        # holds this open for hours between bursts of events, and a stale
        # connection surfaces as a lost batch rather than an error anyone sees.
        pool_size=1,
        max_overflow=1,
        pool_pre_ping=True,
        pool_recycle=280,
    )


def sink_from_environment() -> RecordSink | None:
    """A sink when a DSN is set, `None` when it is not.

    `None` IS THE ROLLOUT. A task dispatched from a machine with no DSN, or one
    running a code snapshot from before this existed, writes files and nothing
    else -- exactly what every task did before the database. Dual-write is
    opt-in by setting one variable, and opting out is unsetting it.

    A DSN that is set but unusable is a different case and is NOT swallowed
    here: it means someone intended dual-write and it is not happening, which
    they need to be told at dispatch rather than discover in a query later.
    """
    engine = engine_from_environment()
    if engine is None:
        return None

    from src.adapters.postgres.sink import PostgresSink  # noqa: PLC0415

    log.info("record sink attached")
    return PostgresSink(engine)
