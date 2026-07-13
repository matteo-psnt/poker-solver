"""Tests for the duplicate-deal resolver-vs-blueprint match harness."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.actions import call
from src.core.game.state import Card, Street
from src.engine.solver.mccfr import MCCFRSolver
from src.pipeline.evaluation.resolver_match import (
    _deal_from_stack,
    play_resolver_match,
)
from tests.test_helpers import build_trained_test_solver


def _build_solver(iterations: int) -> MCCFRSolver:
    return build_trained_test_solver(iterations, session_id="resolver-match")


class TestDuplicateDealing:
    """Board cards must come off fixed deck positions, street by street."""

    def test_streets_deal_from_fixed_positions(self):
        solver = _build_solver(0)
        state = solver.deal_initial_state()
        # A preflop call closes the street in this engine, leaving the flop chance node.
        state = state.apply_action(call(), solver.rules)
        assert solver.is_chance_node(state)

        deck = Card.get_full_deck()
        known = {c.mask for hand in state.hole_cards for c in hand}
        stack = [c for c in deck if c.mask not in known][:5]

        flop_state = _deal_from_stack(state, stack)
        assert flop_state.street == Street.FLOP
        assert flop_state.board == tuple(stack[:3])
        assert not solver.is_chance_node(flop_state)


class TestResolverMatch:
    """End-to-end: the gate runs, pairs deals, and reports sane statistics."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_match_runs_and_reports(self):
        solver = _build_solver(3)
        result = play_resolver_match(solver, num_deals=2, time_budget_ms=20, seed=5)

        assert result.num_deals == 2
        assert result.num_hands == 4
        assert len(result.pair_samples_mbb) == 2
        assert result.resolver_decisions > 0
        assert result.resolver_fallbacks <= result.resolver_decisions
        assert np.isfinite(result.resolver_mbb_per_hand)
        assert np.isfinite(result.se_mbb)
        lo, hi = result.confidence_95_mbb
        assert lo <= result.resolver_mbb_per_hand <= hi
