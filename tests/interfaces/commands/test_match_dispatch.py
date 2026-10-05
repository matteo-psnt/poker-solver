"""The head-to-head gate reaches the pool, and an opponent is resolved here.

`match` had no cloud path at all: `score --method` listed three estimators and
none of them was the one the rules make gate #1. A match of two PUBLISHED runs
then had to be staged onto a laptop by hand, ~1 GB a checkpoint over a WAN,
which is why the gate every arm is supposed to pass had been run on two arms
once.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from src.interfaces.commands import score
from src.interfaces.errors import CommandError


@pytest.fixture
def published(monkeypatch):
    """Resolve fragments and verify rungs against a fixed set of published runs."""
    seen: dict[str, list[str]] = {}

    def _install(runs: dict[str, list[str]]):
        from src.interfaces.cloud.store import workspace

        def _resolve(fragment: str) -> str:
            matches = [run for run in runs if fragment in run]
            if len(matches) != 1:
                raise CommandError(f"no unique run for {fragment!r}")
            return matches[0]

        def _verify(run_id: str, rungs: list[str]) -> None:
            seen[run_id] = list(rungs)
            missing = [r for r in rungs if r and r not in runs[run_id]]
            if missing:
                raise CommandError(f"{run_id} does not hold {missing}")

        monkeypatch.setattr(workspace, "resolve_published_run", _resolve)
        monkeypatch.setattr(workspace, "verify_published_rungs", _verify)
        return seen

    return _install


class TestTheOpponentIsResolvedBeforeDispatch:
    """The node has no fragment matcher, and a pruned rung fails there only
    after a snapshot upload, a ~3-minute pool spin-up and three task retries."""

    def test_a_fragment_becomes_a_full_run_id_on_the_command_line(self, published):
        published({"run-pcs-plain-20108": ["4800"], "run-pcs-turnriver-7245": ["4000"]})
        flags = score._opponent_flags("blueprint_match", "7245", "4000")
        assert flags == ("--vs-run", "run-pcs-turnriver-7245", "--vs-at", "4000")

    def test_no_rung_asked_for_means_the_published_checkpoint(self, published):
        published({"run-b": ["4000"]})
        assert score._opponent_flags("blueprint_match", "run-b", "") == ("--vs-run", "run-b")

    def test_a_rung_the_opponent_does_not_hold_is_refused_here(self, published):
        published({"run-b": ["4000"]})
        with pytest.raises(CommandError, match="does not hold"):
            score._opponent_flags("blueprint_match", "run-b", "9999")

    def test_a_match_without_an_opponent_is_refused(self, published):
        published({"run-b": ["4000"]})
        with pytest.raises(CommandError, match="needs an opponent"):
            score._opponent_flags("blueprint_match", None, "")

    def test_another_estimator_needs_none(self, published):
        published({"run-b": ["4000"]})
        assert score._opponent_flags("exact_br", None, "") == ()

    def test_a_depth_duel_needs_an_opponent_too(self, published):
        published({"run-b": ["4000"]})
        with pytest.raises(CommandError, match="needs an opponent"):
            score._opponent_flags("depth_duel", None, "")


class TestLaddersAreResolvedBeforeDispatch:
    """An elimination names many runs; the node fetches only what the flags carry."""

    RUNS: ClassVar = {
        "run-d75-5790": ["4000"],
        "run-nobr-26596": ["8000"],
        "run-d50-26625": ["12000"],
    }

    def test_every_fragment_is_resolved_and_the_anchor_takes_its_rung(self, published):
        seen = published(self.RUNS)
        flags, rungs = score._ladder_flags(
            "elimination", "run-d75-5790", "5790:4000,26596:8000:0.35", "26596:8000:0.35", [""]
        )
        assert flags == (
            "--ladder-a",
            "run-d75-5790:4000:0,run-nobr-26596:8000:0.35",
            "--ladder-b",
            "run-nobr-26596:8000:0.35",
        )
        assert rungs == ["4000"]
        assert seen["run-nobr-26596"] == ["8000"]

    def test_the_scored_run_must_be_in_ladder_a(self, published):
        published(self.RUNS)
        with pytest.raises(CommandError, match="not a rung of --ladder-a"):
            score._ladder_flags("elimination", "run-d50-26625", "5790:4000", "26596:8000", [""])

    def test_a_conflicting_at_is_refused(self, published):
        published(self.RUNS)
        with pytest.raises(CommandError, match="disagrees"):
            score._ladder_flags("elimination", "run-d75-5790", "5790:4000", "26596:8000", ["2000"])

    def test_a_ladder_on_another_method_is_refused(self, published):
        published(self.RUNS)
        with pytest.raises(CommandError, match="elimination"):
            score._ladder_flags("exact_br", "run-d75-5790", "5790:4000", "", [""])
