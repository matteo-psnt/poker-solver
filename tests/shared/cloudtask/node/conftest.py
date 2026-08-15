"""The node's disk, as a tmp_path.

Every module in the node package takes a :class:`NodePaths` rather than reading
the environment, which is exactly what lets the whole wrapper be exercised here
without a Batch node.
"""

from __future__ import annotations

import sys
import time

import pytest

from src.shared.cloudtask.node import progress
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
