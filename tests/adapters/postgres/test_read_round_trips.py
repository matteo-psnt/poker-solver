"""Two ways a correct answer arrives slowly, both measured against Sweden.

Neither failure raises, logs, or changes a row. A listing that took 3,854 ms
looked exactly like one that took 279 ms, so the only thing that can hold these
is a test that counts what crosses the wire rather than what comes back.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.adapters.postgres import connect, queries


class TestOneEnginePerProcess:
    """A SQLAlchemy engine owns a connection pool. Building one per call threw
    the pool away and paid a fresh TLS handshake and SCRAM auth every time:
    3,806 ms per call rebuilding against 701 ms reusing, none of it the query,
    which runs in 2.8 ms server-side.
    """

    def test_two_calls_share_one_engine(self, monkeypatch):
        monkeypatch.setenv(connect.DSN_ENV, "postgresql://u:p@h:5432/db")
        assert connect.engine_from_environment() is connect.engine_from_environment()

    def test_the_sinks_engine_is_not_the_readers(self, monkeypatch):
        """`pre_ping` is a full round trip and worth it only for the sink, whose
        connection sits idle for hours. Sharing one engine would force one
        policy onto both."""
        monkeypatch.setenv(connect.DSN_ENV, "postgresql://u:p@h:5432/db")
        assert connect.engine_from_environment(pre_ping=True) is not (
            connect.engine_from_environment(pre_ping=False)
        )

    def test_no_dsn_is_no_engine(self, monkeypatch):
        monkeypatch.setenv(connect.DSN_ENV, "   ")
        assert connect.engine_from_environment() is None


class _Recorder:
    """An engine that records the options a read asks for."""

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}
        self.statements: list[Any] = []

    def connect(self) -> _Recorder:
        return self

    def execution_options(self, **options: Any) -> _Recorder:
        self.options.update(options)
        return self

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any, _params: Any = None) -> _Recorder:
        self.statements.append(statement)
        return self

    def all(self) -> list[Any]:
        return []


def test_a_read_does_not_open_a_transaction():
    """SQLAlchemy begins on first execute and rolls back on close, so a plain
    `connect()` spends three round trips on a one-statement read -- 554 ms to
    answer `SELECT 1` where the round trip is 175 ms.
    """
    engine = _Recorder()
    queries.describe_runs(engine)
    assert engine.options.get("isolation_level") == "AUTOCOMMIT"


@pytest.mark.parametrize("reader", [queries.describe_runs])
def test_every_reader_goes_through_the_read_helper(reader):
    """One reader today. The cost is per-connection, not per-query, so a second
    one that calls `engine.connect()` directly pays the same three round trips
    again -- and is just as silent about it."""
    engine = _Recorder()
    reader(engine)
    assert engine.options.get("isolation_level") == "AUTOCOMMIT", (
        f"{reader.__name__} opens a transaction to read"
    )
