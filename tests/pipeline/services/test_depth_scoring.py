"""The cross-depth gates as recorded: what they refuse, and which rows may pair.

`depth_duel` exists to let the runs' STACKS differ, so the only game difference
it may refuse is the BLINDS -- the one thing the replay cannot convert.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.pipeline.evaluation.ledger import tiers
from src.pipeline.services.scoring import depth
from tests.test_helpers import build_test_solver, build_trained_test_solver, make_test_config

BB = 100


@dataclass
class _Metadata:
    run_id: str
    card_abstraction_hash: str = "h"
    action_config_hash: str = "a"
    config_name: str = "production"
    git_commit: str = "c"


def _loaded(solver, run_id: str, rung: int = 400) -> depth._Loaded:
    return depth._Loaded(solver, _Metadata(run_id), rung, 0)  # ty: ignore[invalid-argument-type]


@pytest.fixture
def stub_loads(monkeypatch):
    """`_load` keyed by run name, so a test names the blueprint behind each run."""
    table: dict[str, Any] = {}

    def fake(run_dir: Path, at, threshold, side, source):
        return _loaded(table[run_dir.name], run_dir.name)

    monkeypatch.setattr(depth, "_load", fake)
    return table


class TestDepthDuel:
    def test_different_stacks_play(self, stub_loads):
        stub_loads["a"] = build_trained_test_solver(4, starting_stack=4 * BB)
        stub_loads["b"] = build_trained_test_solver(4, starting_stack=12 * BB)
        out = depth.evaluate_depth_duel(Path("a"), Path("b"), table_bb=8, num_deals=4)
        assert out.results["table_bb"] == 8.0
        assert (out.results["depth_a_bb"], out.results["depth_b_bb"]) == (4.0, 12.0)

    def test_different_blinds_are_refused(self, stub_loads):
        stub_loads["a"] = build_trained_test_solver(1, starting_stack=4 * BB)
        stub_loads["b"], _ = build_test_solver(
            make_test_config(seed=42, small_blind=5, big_blind=10, starting_stack=400)
        )
        with pytest.raises(ValueError, match="Blinds differ"):
            depth.evaluate_depth_duel(Path("a"), Path("b"), table_bb=8, num_deals=2)

    def test_a_shared_strategy_table_is_refused(self, stub_loads):
        solver = build_trained_test_solver(1)
        stub_loads["a"] = stub_loads["b"] = solver
        with pytest.raises(ValueError, match="share one strategy table"):
            depth.evaluate_depth_duel(Path("a"), Path("b"), table_bb=4, num_deals=2)


class TestElimination:
    def test_the_row_must_be_filed_under_a_rung_of_ladder_a(self):
        with pytest.raises(ValueError, match="not a rung of ladder A"):
            depth.evaluate_elimination(
                Path("elsewhere"),
                ladder_a=[(Path("a"), 400, 0.0)],
                ladder_b=[(Path("b"), 400, 0.0)],
            )

    def test_one_load_serves_both_ladders(self, stub_loads, monkeypatch):
        """Ladders share most rungs and a 100 bb blueprint is gigabytes."""
        stub_loads["a"] = build_trained_test_solver(2, starting_stack=8 * BB)
        calls: list[str] = []
        fake = depth._load
        monkeypatch.setattr(depth, "_load", lambda run, *a: calls.append(run.name) or fake(run, *a))
        out = depth.evaluate_elimination(
            Path("a"),
            ladder_a=[(Path("a"), 400, 0.0)],
            ladder_b=[(Path("a"), 400, 0.0)],
            num_pairs=2,
            starting_bb=8,
        )
        assert calls == ["a"]
        assert out.results["a_win_rate"] == 0.5, "an identical ladder must split every pair"
        assert out.results["ladder_a"] == ["a:400:0"]


class TestTheTierSplits:
    @staticmethod
    def _duel() -> dict[str, Any]:
        return {"run_b": "b", "rung_b": 8000, "num_deals": 20000, "seed": 1, "table_bb": 75.0}

    @staticmethod
    def _digest(method: str, knobs: dict[str, Any]) -> str:
        return tiers.tier_digest({"method": method, "knobs": knobs})

    def test_a_duel_at_another_table_depth_is_another_tier(self):
        at75 = tiers.build_depth_duel_knobs(self._duel())
        at80 = tiers.build_depth_duel_knobs(self._duel() | {"table_bb": 80.0})
        assert self._digest("depth_duel", at75) != self._digest("depth_duel", at80)
        assert tiers.tier_mismatches(
            {"method": "depth_duel", "knobs": at75}, {"method": "depth_duel", "knobs": at80}
        )

    def test_a_served_side_is_another_tier(self):
        raw = tiers.build_depth_duel_knobs(self._duel())
        cut = tiers.build_depth_duel_knobs(self._duel() | {"policy_threshold_b": 0.35})
        assert self._digest("depth_duel", raw) != self._digest("depth_duel", cut)

    @staticmethod
    def _elimination(**overrides: Any) -> dict[str, Any]:
        base = {
            "ladder_a": ["a:4000:0.35", "d50:12000:0.35"],
            "ladder_b": ["b:8000:0.35", "d50:12000:0.35"],
            "num_pairs": 500,
            "starting_bb": 100.0,
            "escalate_every": 10,
            "escalate_by": 1.5,
            "seed": 1,
        }
        return base | overrides

    @pytest.mark.parametrize(
        "change",
        [
            {"ladder_b": ["b:4000:0.35", "d50:12000:0.35"]},
            {"escalate_every": 20},
            {"escalate_by": 2.0},
            {"starting_bb": 50.0},
        ],
    )
    def test_an_elimination_splits_on_each_ladder_and_the_schedule(self, change):
        base = tiers.build_elimination_knobs(self._elimination())
        moved = tiers.build_elimination_knobs(self._elimination(**change))
        assert self._digest("elimination", base) != self._digest("elimination", moved)
        assert "escalate_every=10" in tiers.tier_label({"method": "elimination", "knobs": base})
