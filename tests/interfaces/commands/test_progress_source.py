"""`progress` answers from either store, and both must answer the same.

Compared across the whole record before this flip: 276 runs carry a checkpoint
series, all 276 identical between share and database, none present on one side
only. This holds the properties that comparison depended on.
"""

from __future__ import annotations

import argparse

import pytest

from src.adapters.postgres import queries
from src.interfaces.commands import _base, progress
from src.interfaces.errors import CommandError


def _args(run: str = "run-a", last: int = 0) -> argparse.Namespace:
    return argparse.Namespace(run=run, last=last)


def _series(n: int) -> list[dict[str, object]]:
    return [
        {"iteration": i * 1000, "coverage": 0.5 + i / 100, "event": "checkpoint"} for i in range(n)
    ]


class TestAnEmptySeriesRefuses:
    def test_a_run_with_no_series_refuses_rather_than_returning_nothing(self, monkeypatch):
        """An empty chart and "this run never checkpointed" are different facts."""
        monkeypatch.setattr(progress.connect, "engine_from_environment", lambda: object())
        monkeypatch.setattr(progress, "resolve_run_id", lambda run, _e: run)
        monkeypatch.setattr(progress.queries, "checkpoint_series", lambda _e, _r: [])
        with pytest.raises(CommandError, match="No checkpoint history"):
            progress.run(_args())


@pytest.fixture
def resolve(monkeypatch):
    """`resolve_run_id` against a given set of published ids."""

    def _resolve(fragment: str, names: list[str]) -> str:
        monkeypatch.setattr(queries, "run_ids", lambda _e: names)
        return _base.resolve_run_id(fragment, object())

    return _resolve


class TestOneRuleForWhatAFragmentIdentifies:
    """The share resolves a fragment against directory names, the database
    against ids. Both go through `run_names.matching`, because a reader that
    pulled one run and answered about another is the failure that rule exists
    for.
    """

    def test_a_fragment_resolves(self, resolve):
        assert resolve("1095", ["run-production-025433-1095", "run-other-7"]) == (
            "run-production-025433-1095"
        )

    def test_an_exact_id_wins_over_being_a_substring(self, resolve):
        """With `run-a` and `run-a-2` both published, `run-a` is unambiguous --
        otherwise the full id becomes unusable."""
        assert resolve("run-a", ["run-a", "run-a-2"]) == "run-a"

    def test_ambiguity_names_the_candidates(self, resolve):
        """Measured: `--run 1095` matches two runs. Taking the first would have
        answered about a different run than the one asked about."""
        with pytest.raises(CommandError, match="matches 2 runs"):
            resolve("run-", ["run-a", "run-b"])

    def test_no_match_refuses(self, resolve):
        with pytest.raises(CommandError, match="Run not found"):
            resolve("nope", ["run-a"])

    def test_an_empty_fragment_is_not_every_run(self, resolve):
        """`Path("")` is the current directory on the share path; here an empty
        fragment is a substring of every id. The blueprint host's systemd unit
        ships `RUN=` empty."""
        with pytest.raises(CommandError, match="No run given"):
            resolve("  ", ["run-a"])

    def test_a_path_is_not_accepted(self, resolve):
        """On the share a run IS a directory. Here there is nothing for a path
        to name, and resolving its basename would answer about whichever
        published run happened to share it."""
        with pytest.raises(CommandError, match="Run not found"):
            resolve("/tmp/somewhere/run-a", ["run-a"])
