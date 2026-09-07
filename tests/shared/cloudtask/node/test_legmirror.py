"""The wrapper stores its own records, and cannot be hurt by failing to.

This replaces a subprocess. The wrapper runs on the interpreter the pool's start
task installs, which is why it can import a driver at all -- and why the import
is deferred and caught: the node loads this package BEFORE `uv sync`, and a
module-level third-party import that is missing kills the task at bootstrap,
before it can write the record that would explain it.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from src.shared.cloudtask import task_log
from src.shared.cloudtask.node import legmirror

DSN = "postgresql://u:p@h:5432/db"


class _Cursor:
    """Just enough psycopg to see what statement was sent and answer it."""

    def __init__(self, owner: _Connection) -> None:
        self._owner = owner

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, statement: str, params: dict[str, Any]) -> None:
        self._owner.executed.append((statement, params))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._owner.answer


class _Connection:
    def __init__(self, answer: tuple[Any, ...] | None = (3,)) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.answer = answer
        self.committed = False

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.committed = True


def _driver(monkeypatch, connection: _Connection) -> _Connection:
    """Stand in for the driver the START TASK installs beside the interpreter.

    Patched into `sys.modules` rather than onto the module, because the import
    is deliberately INSIDE the function -- see the module docstring -- and
    patching an attribute would test a seam that does not exist.
    """
    module = types.SimpleNamespace(connect=lambda _dsn, **_kw: connection)
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return connection


def _stored(monkeypatch) -> list[list[dict[str, Any]]]:
    calls: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: calls.append(rows))
    return calls


class TestClaimingAnAttempt:
    """The ONE thing on the node allowed to fail, and it has to be.

    The number it returns names the attempt every later record of this task
    belongs to. Inventing one after a failed write means the exit record and the
    progress samples overwrite a PREVIOUS attempt -- destroying the account of
    the failure that caused this retry, on what is now the only copy.
    """

    def test_it_returns_the_attempt_the_database_assigned(self, monkeypatch):
        _driver(monkeypatch, _Connection(answer=(3,)))
        assert legmirror.claim_attempt("task-a", {"task_id": "task-a"}, dsn=DSN) == 3

    def test_the_number_is_derived_inside_the_insert(self, monkeypatch):
        """Not read, then written. A SELECT-then-INSERT can interleave, and the
        loser overwrites rather than failing; `MAX(attempt) + 1` inside the
        INSERT makes the primary key the referee."""
        connection = _driver(monkeypatch, _Connection(answer=(1,)))
        legmirror.claim_attempt("task-a", {"task_id": "task-a"}, dsn=DSN)
        (statement, _params), *rest = connection.executed
        assert rest == [], "one statement, or the race is back"
        assert "INSERT INTO legs" in statement
        assert "MAX(attempt)" in statement
        assert "RETURNING attempt" in statement

    def test_the_task_scoped_progress_row_cannot_skew_it(self, monkeypatch):
        """A progress row lives at `TASK_SCOPED` (-1), which is a REAL value in
        this column. Aggregating over it would make the numbering depend on the
        start row happening to land first -- an argument about ordering, holding
        up the numbering of every record that follows."""
        connection = _driver(monkeypatch, _Connection(answer=(1,)))
        legmirror.claim_attempt("task-a", {"task_id": "task-a"}, dsn=DSN)
        assert "attempt >= 0" in connection.executed[0][0]

    def test_every_parameter_is_cast(self, monkeypatch):
        """MEASURED, on a node, three retries deep: `task_id` is both selected
        and compared against its column, so Postgres deduced `text` from one
        context and `character varying` from the other and refused the whole
        statement with `AmbiguousParameter`.

        Asserted as text because nothing else here can see it -- the fake driver
        below parses no SQL, which is exactly why this reached a node. The real
        check is `scratchpad/claim_probe.py` against a live server.
        """
        connection = _driver(monkeypatch, _Connection(answer=(1,)))
        legmirror.claim_attempt("task-a", {"task_id": "task-a"}, dsn=DSN)
        statement = connection.executed[0][0]
        for placeholder in ("task_id", "leg", "run_id", "at", "body"):
            assert f"%({placeholder})s::" in statement, f"{placeholder} is not cast"

    def test_it_commits(self, monkeypatch):
        connection = _driver(monkeypatch, _Connection(answer=(1,)))
        legmirror.claim_attempt("task-a", {"task_id": "task-a"}, dsn=DSN)
        assert connection.committed, "an uncommitted claim is not a claim"

    def test_an_unreachable_database_raises(self, monkeypatch):
        """Unlike every other write here. `record` logs and continues because
        losing one row costs a row; losing the CLAIM costs a previous attempt's
        whole account."""

        def _explode(_dsn, **_kw):
            raise RuntimeError("database is gone")

        monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(connect=_explode))
        with pytest.raises(RuntimeError):
            legmirror.claim_attempt("task-a", {"task_id": "task-a"}, dsn=DSN)

    def test_an_answer_of_nothing_raises(self, monkeypatch):
        """A statement that returned no row assigned no number, and proceeding
        would mean guessing one."""
        _driver(monkeypatch, _Connection(answer=None))
        with pytest.raises(RuntimeError):
            legmirror.claim_attempt("task-a", {"task_id": "task-a"}, dsn=DSN)

    def test_the_body_is_task_logs_row_and_not_a_second_one(self, monkeypatch):
        """One builder for the row, as everywhere else."""
        connection = _driver(monkeypatch, _Connection(answer=(1,)))
        document = {"task_id": "task-a", "ts": "t"}
        legmirror.claim_attempt("task-a", document, dsn=DSN)
        _statement, params = connection.executed[0]
        expected = task_log.leg_row("task-a", 0, "start", document)
        assert params["task_id"] == expected["task_id"]
        assert params["leg"] == "start"
        assert json.loads(params["body"]) == expected["body"]


class TestTheLatestAttempt:
    """What the terminal record belongs to. Derived rather than carried: the
    exit trap may have lost anything the entry point computed."""

    def test_it_reads_the_maximum(self, monkeypatch):
        connection = _driver(monkeypatch, _Connection(answer=(4,)))
        assert legmirror.latest_attempt("task-a", dsn=DSN) == 4
        assert "MAX(attempt)" in connection.executed[0][0]

    def test_it_ignores_the_task_scoped_progress_row(self, monkeypatch):
        connection = _driver(monkeypatch, _Connection(answer=(2,)))
        legmirror.latest_attempt("task-a", dsn=DSN)
        assert "attempt >= 0" in connection.executed[0][0]

    def test_nothing_claimed_is_zero(self, monkeypatch):
        """A task that died before its start row landed has no attempt to
        belong to, and 0 says so rather than claiming attempt 1's."""
        _driver(monkeypatch, _Connection(answer=None))
        assert legmirror.latest_attempt("task-a", dsn=DSN) == 0


class TestTheRowIsWrittenWhereTheRecordIsMade:
    """Every record writes its own row where it is made.

    There was a second half -- `publish` re-read this task's files and upserted
    whatever it found, which was self-healing. It could only heal a record that
    HAD a file, and none do now.
    """

    def test_one_record_becomes_one_row(self, tmp_path, monkeypatch):
        stored: list = []
        monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: stored.extend(rows))
        legmirror.record("task-a", 1, "start", {"task_id": "task-a", "ts": "t"}, dsn=DSN)
        assert [(r["task_id"], r["attempt"], r["leg"]) for r in stored] == [("task-a", 1, "start")]

    def test_it_builds_the_row_with_task_logs_builder(self, tmp_path, monkeypatch):
        """The node, the importer and the reader's write-back all go through
        `task_log.leg_row`. A second builder is a divergence `--verify` reports
        forever."""
        stored: list = []
        monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: stored.extend(rows))
        doc = {"task_id": "task-a", "done": 3.0, "ts": "t"}
        legmirror.record("task-a", task_log.TASK_SCOPED, "progress", doc, dsn=DSN)
        assert stored == [task_log.leg_row("task-a", task_log.TASK_SCOPED, "progress", doc)]

    def test_no_dsn_writes_nothing(self, tmp_path, monkeypatch):
        stored: list = []
        monkeypatch.setattr(legmirror, "_store", lambda _dsn, rows: stored.extend(rows))
        legmirror.record("task-a", 1, "start", {"task_id": "task-a"}, dsn="")
        assert stored == []

    def test_a_failure_cannot_cost_the_task(self, tmp_path, monkeypatch):
        def _explode(_dsn, _rows):
            raise RuntimeError("database is gone")

        monkeypatch.setattr(legmirror, "_store", _explode)
        logged: list[str] = []
        legmirror.record("task-a", 1, "exit", {"task_id": "task-a"}, dsn=DSN, log=logged.append)
        assert logged
        assert "not recorded" in logged[0]
