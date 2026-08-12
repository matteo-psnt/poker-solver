"""`runs` answers from either store, and the answers must be the same answer.

Not a speed test. The flip is only safe if the database reproduces what the
share says, and every defect this migration has produced was found by comparing
CONTENT after the counts already agreed.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from src.interfaces.commands import runs as runs_cmd


class _Row:
    """One row as the query returns it."""

    def __init__(self, **fields: Any):
        self.__dict__.update(fields)


def _row(**overrides: Any) -> _Row:
    base = {
        "run_id": "run-a",
        "config_name": "production",
        "status": "completed",
        "iterations": 30_000_000,
        "num_infosets": 900_275,
        "experiment_id": "exp-7",
        "arm": "control",
        "git_commit": "abc1234",
        "git_dirty": False,
        "started_at": "2026-01-01T00:00:00+00:00",
        "has_checkpoint": True,
    }
    return _Row(**(base | overrides))


def _summaries(monkeypatch, rows: list[_Row]):
    monkeypatch.setattr(runs_cmd.queries, "describe_runs", lambda _e, **_k: rows)
    monkeypatch.setattr(runs_cmd, "commits_ahead_of", lambda commit: 7 if commit else None)
    return runs_cmd._from_database(object())


class TestLoadableMeansACurrentRung:
    """The share answers this by asking whether STATIC_CHECKPOINT.json exists,
    and that manifest is what NAMES the current rung. A real run on the share
    has three complete rungs, three markers and no manifest -- counting any rung
    reported it loadable when a loader would resolve nothing and start at zero.
    """

    def test_a_run_with_a_current_rung_is_loadable(self, monkeypatch):
        (summary,) = _summaries(monkeypatch, [_row(has_checkpoint=True)])
        assert summary.loadable is True
        assert summary.blocker is None

    def test_a_run_with_no_current_rung_is_not(self, monkeypatch):
        (summary,) = _summaries(monkeypatch, [_row(has_checkpoint=False)])
        assert summary.loadable is False
        assert summary.blocker == "no checkpoint"


class TestTheGitAnswerIsPerCheckout:
    def test_commits_ago_is_computed_not_stored(self, monkeypatch):
        """It is a fact about THIS working copy, not about the run, so storing
        it would be storing someone else's answer."""
        (summary,) = _summaries(monkeypatch, [_row(git_commit="abc1234")])
        assert summary.commits_ago == 7

    def test_a_run_with_no_commit_reports_none(self, monkeypatch):
        (summary,) = _summaries(monkeypatch, [_row(git_commit=None)])
        assert summary.commits_ago is None

    def test_one_git_call_per_distinct_commit(self, monkeypatch):
        """303 runs share 72 commits, and each call shells out at ~24ms."""
        calls: list[Any] = []
        monkeypatch.setattr(runs_cmd.queries, "describe_runs", lambda _e, **_k: rows)
        monkeypatch.setattr(runs_cmd, "commits_ahead_of", lambda c: (calls.append(c), 1)[1])
        rows = [_row(run_id=f"run-{i}", git_commit="same") for i in range(50)]
        runs_cmd._from_database(object())
        assert len(calls) == 1, f"{len(calls)} git calls for one distinct commit"


class TestThePayloadSaysWhichStoreAnswered:
    def test_the_share_path_says_share(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runs_cmd.connect, "engine_from_environment", lambda: None)

        import contextlib

        @contextlib.contextmanager
        def _root(_args):
            yield tmp_path

        monkeypatch.setattr(runs_cmd, "records_root", _root)
        payload = runs_cmd.run(argparse.Namespace(limit=0, loadable_only=False, run=None))
        assert payload.source == "share"

    def test_the_database_path_says_database(self, monkeypatch):
        monkeypatch.setattr(runs_cmd.connect, "engine_from_environment", lambda: object())
        monkeypatch.setattr(runs_cmd.queries, "describe_runs", lambda _e, **_k: [_row()])
        monkeypatch.setattr(runs_cmd, "commits_ahead_of", lambda _c: 1)
        payload = runs_cmd.run(argparse.Namespace(limit=0, loadable_only=False, run=None))
        assert payload.source == "database", "a silent fallback hides a stale answer"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("iterations", 30_000_000),
        ("status", "completed"),
        ("config_name", "production"),
        ("experiment_id", "exp-7"),
        ("arm", "control"),
        ("num_infosets", 900_275),
    ],
)
def test_every_field_the_listing_shows_survives_the_crossing(monkeypatch, field, value):
    """Nine runs reached the database with a null arm and experiment because the
    importer read the `created` event instead of the fold. A field that arrives
    empty is not a visible failure -- it is a listing that is quietly wrong."""
    (summary,) = _summaries(monkeypatch, [_row(**{field: value})])
    assert getattr(summary, field) == value
