"""The node's disk, as a tmp_path.

Every module in the node package takes a :class:`NodePaths` rather than reading
the environment, which is exactly what lets the whole wrapper be exercised here
without a Batch node.
"""

from __future__ import annotations

import sys
import time

import pytest

from src.shared import task_history
from src.shared.cloudtask import task_log
from src.shared.cloudtask.node import blobstore, legmirror, progress
from src.shared.cloudtask.node.paths import NodePaths
from src.shared.cloudtask.node.process import TaskLogger


@pytest.fixture(autouse=True)
def _one_task_per_test():
    """The baseline and the rate window are module globals, because the wrapper
    process runs exactly ONE task. A test session is not one task."""
    progress._BASELINE.clear()
    progress._WINDOW.clear()


@pytest.fixture
def paths(tmp_path):
    return NodePaths(work=tmp_path / "work", share=tmp_path / "share", code=tmp_path / "code")


@pytest.fixture
def container(monkeypatch):
    """The checkpoint container, as a dict of `{object name: bytes}`.

    Autouse-adjacent by necessity: publishing is the node's ONLY durable output
    now that nothing lands on the share, so a test that asserted a published
    file has to read this instead. It MODELS the store rather than stubbing the
    calls -- `exists` answers for what was already put -- because the ordering
    the publisher guarantees (rungs, then metadata, then the manifest naming
    them) is only observable against a store that remembers.
    """
    store: dict[str, bytes] = {}

    def _put(_sas, name: str, body: bytes) -> int:
        # ASSIGNS. A PUT replaces whatever the name held, which is what makes
        # `force_publish` observable at all -- modelled with `setdefault`, an
        # overwrite silently kept the original and the test read as a refusal.
        store[name] = body
        return len(body)

    monkeypatch.setattr(blobstore, "exists", lambda _s, name: name in store)
    monkeypatch.setattr(
        blobstore, "put_object", lambda s, name, path: _put(s, name, path.read_bytes())
    )
    monkeypatch.setattr(blobstore, "put_bytes", _put)
    monkeypatch.setattr(blobstore, "read_object", lambda _s, name: store.get(name))
    return store


SAS = "https://acct.blob.core.windows.net/checkpoints?sig=test"
"""Any non-empty SAS. The node publishes only when it carries one, so a test
that omits this asserts against a publisher that deliberately did nothing."""


@pytest.fixture
def log(paths):
    logger = TaskLogger(paths.work / "task.log", paths.share)
    yield logger
    logger.close()


def python(*statements: str) -> list[str]:
    """A child process that is this interpreter, so no PATH lookup can vary."""
    return [sys.executable, "-c", "; ".join(statements)]


def eventually(predicate, attempts: int = 200) -> None:
    for _ in range(attempts):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")


class _Recorded:
    """The database, for a node test: what the task claimed and what it wrote.

    `RunTracker`-style file assertions are gone with the files. A node's whole
    account is now rows, so a test reads it back the way `tasks` does -- through
    `join_documents`, the ONE join both stores go through.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, int, str, dict]] = []
        self.attempts: dict[str, int] = {}

    def claim(self, task_id: str, document, *, dsn: str) -> int:
        attempt = self.attempts.get(task_id, 0) + 1
        self.attempts[task_id] = attempt
        self.rows.append((task_id, attempt, "start", {**dict(document), "attempt": attempt}))
        return attempt

    def latest(self, task_id: str, *, dsn: str) -> int:
        return self.attempts.get(task_id, 0)

    def record(self, task_id: str, attempt: int, leg: str, document, **_kw: object) -> None:
        self.rows.append((task_id, attempt, leg, dict(document)))

    def join(self):
        return task_history.join_documents(task_log.documents_from_rows(self.rows))


@pytest.fixture
def recorded(monkeypatch) -> _Recorded:
    """Stand in for the record database on the node path."""
    store = _Recorded()
    monkeypatch.setattr(legmirror, "claim_attempt", store.claim)
    monkeypatch.setattr(legmirror, "latest_attempt", store.latest)
    monkeypatch.setattr(legmirror, "record", store.record)
    return store
