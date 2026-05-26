"""A warm-started task with no prior must fail, not quietly become a control.

This cost eight node-hours. Four 30M arms were submitted differing only in prior
strength; the prior had been pruned from the share in the meantime; the node
logged ``WARN ... is not on the share`` and trained anyway. All four came back
identical, and nothing said so until their coverage ladders turned out to match
to a tenth of a percent.

The failure is silent by construction: an unseeded warm arm IS a control, so it
trains fine, publishes fine, and completes with exit 0. Only the thing the
experiment was measuring is missing -- which is exactly the shape of failure a
warning cannot carry, because nobody reads a node log for a task that succeeded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.cloudtask.node import handlers, process, progress
from src.shared.cloudtask.node import paths as node_paths
from src.shared.cloudtask.node import plan as node_plan

ENV = {
    "RUN_OP": "train",
    "RUN_CONFIG": "production",
    "RUN_TO": "30000000",
    "AZ_BATCH_TASK_ID": "t1",
}


class _StubWatcher:
    def __init__(self, *_args, **_kwargs) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


@pytest.fixture
def logger(tmp_path: Path) -> process.TaskLogger:
    """A real TaskLogger, so the FATAL line is asserted where an operator would
    actually find it -- the task log -- rather than in a list only a test sees."""
    return process.TaskLogger(tmp_path / "task.log", tmp_path / "share")


@pytest.fixture
def paths(tmp_path: Path) -> node_paths.NodePaths:
    built = node_paths.NodePaths(
        work=tmp_path / "work", share=tmp_path / "share", code=tmp_path / "code"
    )
    built.archive.mkdir(parents=True, exist_ok=True)
    built.runs.mkdir(parents=True, exist_ok=True)
    return built


def _plan(**overrides: str) -> node_plan.TaskPlan:
    return node_plan.parse_environment({**ENV, **overrides})


def test_a_missing_prior_fails_the_task(
    paths: node_paths.NodePaths, logger: process.TaskLogger, monkeypatch
):
    def _never(*_args, **_kwargs):
        raise AssertionError("trained despite a missing prior")

    monkeypatch.setattr(handlers, "run_guarded", _never)
    code, outcome = handlers._train(
        _plan(RUN_WARM_START_FROM="vec-gone", RUN_WARM_START_WEIGHT="1000"), paths, logger
    )

    assert code != 0, "a warm arm with no prior must not report success"
    assert outcome == "missing-prior"
    written = logger.path.read_text()
    assert "FATAL" in written, written
    assert "vec-gone" in written, written


def test_a_present_prior_is_fetched(
    paths: node_paths.NodePaths, logger: process.TaskLogger, monkeypatch
):
    (paths.archive / "vec-here").mkdir(parents=True)
    fetched: list[str] = []
    monkeypatch.setattr(
        handlers.archive,
        "fetch_current_rung",
        lambda source, destination, log=None: (fetched.append(source.name), "")[1],
    )
    monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
    monkeypatch.setattr(progress, "LadderWatcher", _StubWatcher)

    code, _ = handlers._train(
        _plan(RUN_WARM_START_FROM="vec-here", RUN_WARM_START_WEIGHT="1000"), paths, logger
    )
    assert code == 0
    assert "vec-here" in fetched


def test_a_task_with_no_prior_requested_is_unaffected(
    paths: node_paths.NodePaths, logger: process.TaskLogger, monkeypatch
):
    """Only warm-started tasks are gated -- an ordinary control has no prior to
    miss, and gating it would break every scalar run."""
    monkeypatch.setattr(handlers, "run_guarded", lambda *a, **k: 0)
    monkeypatch.setattr(progress, "LadderWatcher", _StubWatcher)
    code, _ = handlers._train(_plan(), paths, logger)
    assert code == 0
