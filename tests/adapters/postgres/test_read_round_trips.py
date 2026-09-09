"""Two ways a correct answer arrives slowly, both measured against Sweden.

Neither failure raises, logs, or changes a row. A listing that took 3,854 ms
looked exactly like one that took 279 ms, so the only thing that can hold these
is a test that counts what crosses the wire rather than what comes back.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.adapters.postgres import connect, queries
from src.interfaces.commands import tasks as tasks_command
from src.interfaces.web import views
from src.shared.task_history import TaskRow


def _task_row(task_id: str) -> TaskRow:
    return TaskRow(task_id=task_id, attempt=1, cause="completed", cause_source="node")


def _widest_screen(monkeypatch) -> int:
    """How many parts the console's widest view fans out over.

    Measured by letting each view build its parts against a `compose` that
    records them instead of invoking them -- so a panel added to a screen
    widens this by existing, rather than by someone remembering to.
    """
    seen: list[int] = []

    def _record(_op, parts, join=None, invoke=None):
        seen.append(len(parts))
        return {"parts": {part.key: {} for part in parts}}

    monkeypatch.setattr(views, "compose", _record)
    monkeypatch.setattr(views, "_summarised", lambda part: part)
    monkeypatch.setattr(views, "_live_and_recent", lambda part: part)
    views.now()
    views.runs()
    views.run("run-a")
    return max(seen)


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

    def test_the_reader_pool_is_as_wide_as_the_widest_screen(self, monkeypatch):
        """A view fans out at `max_workers=len(parts)`. A pool narrower than
        that turns the fan-out into a queue at one round trip each -- 8
        concurrent reads measured at 303 ms against 278 ms for one, and at
        `pool_size=1` they would have cost 8 x 175 ms and still rendered.
        """
        monkeypatch.setenv(connect.DSN_ENV, "postgresql://u:p@h:5432/db")
        engine = connect.engine_from_environment()
        assert engine is not None
        widest = _widest_screen(monkeypatch)
        assert engine.pool.size() >= widest, (
            f"a {widest}-panel screen serialises through a pool of {engine.pool.size()}"
        )

    def test_no_dsn_is_a_refusal_not_a_second_answer(self, monkeypatch):
        monkeypatch.setenv(connect.DSN_ENV, "   ")
        with pytest.raises(connect.NoRecordError):
            connect.engine_from_environment()


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

    def scalar(self) -> int:
        return 0

    # A reader that builds its rows by iterating the result, which `leg_rows`
    # does. Empty: these count the STATEMENT, never what comes back.
    def __iter__(self):
        return iter(())


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


class TestABoundedReadFetchesABoundedNumberOfRows:
    """The third way a correct answer arrives slowly: fetching all of it.

    `--limit n` used to be a slice of the answer. The query returned every leg
    the log holds -- 16,895 rows, 10.6 MB -- and the command threw all but `n`
    away. Postgres reads every body in 0.02s; shipping them costs 1.65s, and the
    status bar polls the view that did it every 5 seconds from every page.

    None of that is visible in what comes back, which is why these count the
    statement rather than the rows.
    """

    def test_a_limit_bounds_the_query_and_not_just_the_answer(self):
        engine = _Recorder()
        queries.leg_rows(engine, recent_tasks=200)
        assert "LIMIT :tasks" in str(engine.statements[-1]), (
            "the whole log was fetched to return 200 rows"
        )

    def test_no_limit_still_reads_the_whole_log(self):
        """The run list needs it: only the whole log can say which runs have
        ever had a task, and a bounded read would call an old one abandoned."""
        engine = _Recorder()
        queries.leg_rows(engine)
        assert "LIMIT" not in str(engine.statements[-1])

    def test_the_bound_is_by_task_so_a_tasks_attempts_stay_together(self):
        """`join_documents` decides which attempt of a task is latest and can
        only do that seeing all of them -- and `attempt` is a sentinel (-1/0) on
        the `progress` and `observed` legs, so grouping on it would file a
        task's progress apart from its own start."""
        engine = _Recorder()
        queries.leg_rows(engine, recent_tasks=200)
        statement = " ".join(str(engine.statements[-1]).split())
        assert "GROUP BY task_id ORDER BY max(at) DESC LIMIT :tasks" in statement

    def test_the_live_view_asks_for_a_bound(self, monkeypatch):
        """The bound only pays if the view that polls every 5s uses it. This is
        the one that made the whole console slow: `now` asked for `--limit 0`."""
        asked: dict[str, dict] = {}

        def _record(_op, parts, join=None, invoke=None):
            asked.update({part.key: part.arguments for part in parts})
            return {"parts": {part.key: {} for part in parts}}

        monkeypatch.setattr(views, "compose", _record)
        monkeypatch.setattr(views, "_live_and_recent", lambda part: part)
        views.now()
        assert asked["tasks"].get("limit") == views.LIVE_WINDOW

    def test_a_bounded_read_reports_what_it_did_not_fetch(self):
        """`hidden_rows` counted the rows in hand. Once those are a window that
        is a fraction of what is hidden, and the Tasks page draws its
        `load all` affordance from it."""
        rows = [_task_row(f"t{i}") for i in range(5)]
        payload = tasks_command._result(rows, None, 2, total=6034)
        assert len(payload.rows) == 2
        assert payload.hidden_rows == 6032
