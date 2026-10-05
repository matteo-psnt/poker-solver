"""A match can purify one side, which is how the SEAT plays what the gate picked.

Every head-to-head gate to date matched raw tables while the fielded seat serves
each rung thresholded at 0.20, so the configuration actually in production had
never been measured on chips. Thresholding one side of a self-match isolates
that knob: duplicate deals share their sampling stream, so identical tables give
provably all-zero pairs and every non-zero pair is a decision the cut changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from src.engine.solver.policy.threshold import apply_policy_threshold
from src.pipeline.evaluation.estimators.blueprint_match import play_blueprint_match
from src.pipeline.evaluation.ledger import tiers
from src.pipeline.services.scoring import matches
from tests.test_helpers import build_trained_test_solver

if TYPE_CHECKING:
    from src.engine.solver.mccfr.static_solver import StaticTreeSolver

# 0.60 over 30 deals, MEASURED: 0.20 and 0.45 both cut rows (11 of 246) that no
# deal in a short match reaches, so the edge is a true zero and the test would
# pass with the threshold wired to nothing. This is the smallest pairing that
# actually fires on the toy tree.
PROBE_THRESHOLD = 0.60
PROBE_DEALS = 30
PROBE_SEED = 3


def _pair(threshold: float) -> tuple[StaticTreeSolver, StaticTreeSolver]:
    """Two strategy-identical blueprints in distinct memory, the first purified."""
    thresholded = build_trained_test_solver(40)
    raw = build_trained_test_solver(40)
    assert not np.shares_memory(thresholded.storage.strategy_sum, raw.storage.strategy_sum)
    changed, _ = apply_policy_threshold(thresholded.storage, thresholded.tree, threshold)
    assert changed > 0, "threshold moved no row, so the match measures nothing"
    return thresholded, raw


class TestThresholdIsTheOnlyDifference:
    """The knob has to reach the table, and reach the side it was asked for."""

    def test_purifying_one_side_moves_chips_and_the_sign_follows_the_side(self):
        thresholded, raw = _pair(PROBE_THRESHOLD)

        purified_is_a = play_blueprint_match(
            thresholded, raw, num_deals=PROBE_DEALS, seed=PROBE_SEED
        )
        purified_is_b = play_blueprint_match(
            raw, thresholded, num_deals=PROBE_DEALS, seed=PROBE_SEED
        )

        assert purified_is_a.a_mbb_per_hand != 0.0
        # The sign test is what catches a threshold applied to the wrong solver:
        # a symmetric bug leaves the magnitude right and the attribution wrong.
        assert purified_is_b.a_mbb_per_hand == pytest.approx(-purified_is_a.a_mbb_per_hand)

    def test_two_raw_builds_of_one_blueprint_are_all_zero(self):
        """The control the guard above rests on: without a threshold there is no
        divergence at all, so any non-zero edge is the knob and not the rebuild."""
        result = play_blueprint_match(
            build_trained_test_solver(40),
            build_trained_test_solver(40),
            num_deals=PROBE_DEALS,
            seed=PROBE_SEED,
        )
        assert all(sample == 0.0 for sample in result.pair_samples_mbb)


class TestSharedTableIsRefused:
    """One table behind both sides would purify both and report a provable zero."""

    def test_a_shared_strategy_table_raises(self, monkeypatch):
        solver = build_trained_test_solver(1)

        @dataclass
        class _Metadata:
            """A DATACLASS, like the real `RunMetadata`: the resolver overrides copy
            it with `dataclasses.replace`. A bare `type(...)` stub satisfied that
            call by accident and then a pydantic-shaped stub satisfied a WRONG one --
            `ty` caught `model_copy` on a class that has no such method."""

            config: object
            run_id: str = "r"
            card_abstraction_hash: str = "h"

        metadata = _Metadata(config=solver.config)
        monkeypatch.setattr(matches, "load_run_metadata", lambda *a, **k: metadata)
        monkeypatch.setattr(matches, "effective_abstraction_hash", lambda *a, **k: "h")
        monkeypatch.setattr(
            matches,
            "build_blueprint_for",
            lambda *a, **k: (solver, solver.storage, {}),
        )

        with pytest.raises(ValueError, match="share one strategy table"):
            matches.evaluate_blueprint_match(
                Path("a"), Path("b"), num_deals=1, policy_threshold_a=0.2
            )


class TestTheTierSplits:
    """A purified row must not pair with the raw row that picked the rung."""

    @staticmethod
    def _results(**overrides: Any) -> dict[str, Any]:
        base = {
            "run_b": "run-b",
            "rung_b": 12000,
            "num_deals": 20000,
            "seed": 1,
        }
        return base | overrides

    def test_raw_rows_keep_the_knobs_they_already_had(self):
        assert tiers.build_blueprint_match_knobs(self._results()) == {
            "run_b": "run-b",
            "rung_b": 12000,
            "num_deals": 20000,
            "base_seed": 1,
        }

    def test_a_purified_side_is_its_own_tier(self):
        raw = tiers.build_blueprint_match_knobs(self._results())
        served = tiers.build_blueprint_match_knobs(
            self._results(policy_threshold_a=0.2, policy_threshold_b=0.0)
        )

        assert served["policy_threshold_a"] == 0.2
        # Zero is ABSENT rather than recorded: every raw-vs-raw row ever written
        # has to keep the digest it already has.
        assert "policy_threshold_b" not in served
        assert tiers.tier_digest(self._record(raw)) != tiers.tier_digest(self._record(served))

    def test_a_deployed_side_b_is_its_own_tier(self):
        """Two DIFFERENT action trees can only meet with B translating; that row
        must not pair with a bare-B match of the same runs."""
        raw = tiers.build_blueprint_match_knobs(self._results())
        translated = tiers.build_blueprint_match_knobs(self._results(opponent_b="deployed"))

        assert translated["opponent_b"] == "deployed"
        assert tiers.tier_digest(self._record(raw)) != tiers.tier_digest(self._record(translated))

    def test_a_blended_side_is_its_own_tier(self):
        """An ensemble side A is a different player; its row must not pair with the
        unblended match of the same run, and the label must say what was blended."""
        raw = tiers.build_blueprint_match_knobs(self._results())
        blended = tiers.build_blueprint_match_knobs(
            self._results(mix_run_a="run-c", mix_at_a=4000, mix_weight_a=0.5)
        )

        assert blended["mix_run"] == "run-c"
        assert tiers.tier_digest(self._record(raw)) != tiers.tier_digest(self._record(blended))
        assert "mix_run=run-c" in tiers.tier_label(self._record(blended))

    def test_the_label_names_the_threshold(self):
        """`tier_label` must name every knob the digest splits on, or the operator
        sees two different tiers rendered as the same string."""
        served = tiers.build_blueprint_match_knobs(self._results(policy_threshold_a=0.2))

        assert "policy_threshold_a=0.2" in tiers.tier_label(self._record(served))

    @staticmethod
    def _record(knobs: dict[str, Any]) -> dict[str, Any]:
        return {"method": "blueprint_match", "knobs": knobs}
