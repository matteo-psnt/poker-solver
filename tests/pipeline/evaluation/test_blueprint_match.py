"""Tests for the duplicate-deal blueprint-vs-blueprint match harness."""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.pipeline.evaluation.blueprint_match import play_blueprint_match
from tests.test_helpers import build_trained_test_solver


def _build_solver(iterations: int) -> StaticTreeSolver:
    return build_trained_test_solver(iterations)


class TestBlueprintMatch:
    """End-to-end: the match runs, pairs deals, and reports sane statistics."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_match_runs_and_reports(self):
        solver_a = _build_solver(3)
        solver_b = _build_solver(3)
        result = play_blueprint_match(solver_a, solver_b, num_deals=4, seed=5)

        assert result.num_deals == 4
        assert result.num_hands == 8
        assert len(result.pair_samples_mbb) == 4
        assert np.isfinite(result.a_mbb_per_hand)
        assert np.isfinite(result.se_mbb)
        lo, hi = result.confidence_95_mbb
        assert lo <= result.a_mbb_per_hand <= hi

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_self_match_is_zero_sum_symmetric(self):
        """A blueprint against itself has an exactly antisymmetric pair: the two
        seat-swapped games are identical decisions on identical cards, so each
        pair sample must be zero."""
        solver = _build_solver(3)
        result = play_blueprint_match(solver, solver, num_deals=4, seed=7)
        assert result.a_mbb_per_hand == 0.0
        assert all(sample == 0.0 for sample in result.pair_samples_mbb)
