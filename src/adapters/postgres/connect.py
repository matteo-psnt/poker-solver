"""Building a sink from the environment, in the one place allowed to decide.

Separate from `sink.py` because construction is a different question from
behaviour: the sink knows how to write, this knows whether there is anywhere to
write to. Commands call this; nothing in `pipeline` can.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.ports.record import RecordSink

log = logging.getLogger(__name__)

DSN_ENV = "POKER_SOLVER_RECORD_DSN"


# One engine per (DSN, pre_ping) FOR THE LIFE OF THE PROCESS. A SQLAlchemy
# engine owns a connection pool and is meant to be long-lived; building one per
# call throws the pool away and pays a fresh TLS handshake and SCRAM auth every
# time. Measured against Sweden from here: 3,806 ms per call rebuilding it
# against 701 ms reusing it, which was the whole of a listing's cost and had
# nothing to do with the query -- that runs in 2.8 ms server-side.
@cache
def _engine(dsn: str, pre_ping: bool) -> Any:
    import sqlalchemy as sa  # noqa: PLC0415 -- only when a DSN says to

    return sa.create_engine(
        dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        pool_size=1,
        max_overflow=1,
        pool_pre_ping=pre_ping,
        # Well inside Azure's idle timeout. A trainer holds this open for hours
        # between bursts, and a stale connection surfaces as a lost batch rather
        # than an error anyone sees.
        pool_recycle=280,
    )


def engine_from_environment(*, pre_ping: bool = False) -> Any | None:
    """An engine when a DSN is set, `None` when it is not.

    Shared by the sink and the readers so there is one answer to "is there a
    database", and one place that knows how to build a connection to it.

    `pre_ping` sends a `SELECT 1` before handing out a pooled connection, and
    that is a FULL ROUND TRIP -- 175 ms from here, because the server is in
    Sweden and this is not. Worth it for the sink, whose connection sits idle
    for hours between bursts and would otherwise surface a stale socket as a
    lost batch. Not worth it for a reader, which opens a connection and uses it
    immediately.
    """
    dsn = os.environ.get(DSN_ENV, "").strip()
    if not dsn:
        return None
    return _engine(dsn, pre_ping)


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
    engine = engine_from_environment(pre_ping=True)
    if engine is None:
        return None

    from src.adapters.postgres.sink import PostgresSink  # noqa: PLC0415

    log.info("record sink attached")
    return PostgresSink(engine)
