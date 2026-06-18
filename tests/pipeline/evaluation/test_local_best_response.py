"""Bridge tests for Local Best Response against exact best response.

LBR is a *lower bound* on exploitability. On Kuhn and Leduc — where the exact
best response is computable — we can assert the defining relationships:
``local_exploitability <= exploitability`` (LBR never over-claims) and
``LBR detects a positive lower bound on an exploitable strategy``. This transfers
trust from the exact harness to the LBR algorithm, which is what the HUNL LBR
(that has no exact reference) reuses.
"""

from __future__ import annotations

import pytest

from src.pipeline.evaluation.best_response import best_response_value, exploitability
from src.pipeline.evaluation.game_tree import TabularStrategy
from src.pipeline.evaluation.local_best_response import (
    local_best_response_value,
    local_exploitability,
)
from src.pipeline.evaluation.tabular_cfr import TabularCFRSolver
from tests.pipeline.evaluation.kuhn_poker import KuhnPoker
from tests.pipeline.evaluation.leduc_poker import LeducPoker

GAMES = {"kuhn": KuhnPoker, "leduc": LeducPoker}
_TOL = 1e-9


@pytest.mark.parametrize("game_name", ["kuhn", "leduc"])
class TestLbrBridge:
    def test_lbr_never_exceeds_exact_br(self, game_name):
        """For every player, the LBR value is <= the exact best-response value."""
        game = GAMES[game_name]()
        uniform = TabularStrategy()
        for player in range(game.num_players):
            lbr = local_best_response_value(game, player, uniform)
            exact = best_response_value(game, player, uniform)
            assert lbr <= exact + _TOL

    def test_lbr_exploitability_is_valid_lower_bound(self, game_name):
        """LBR exploitability is non-negative and <= exact exploitability."""
        game = GAMES[game_name]()
        uniform = TabularStrategy()
        lbr_exploit = local_exploitability(game, uniform)
        exact_exploit = exploitability(game, uniform)
        assert lbr_exploit >= -_TOL
        assert lbr_exploit <= exact_exploit + _TOL

    def test_lbr_detects_exploitable_strategy(self, game_name):
        """A uniform-random strategy yields a strictly positive LBR bound."""
        game = GAMES[game_name]()
        assert local_exploitability(game, TabularStrategy()) > 0.0


class TestLbrAtEquilibrium:
    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_lbr_small_near_equilibrium(self):
        """Against a converged strategy the LBR bound is small (nothing to exploit)."""
        game = LeducPoker()
        solver = TabularCFRSolver(game)
        solver.train(100)
        average = solver.average_strategy()
        # LBR is a lower bound, so it is at most the exact exploitability (~0.12).
        assert local_exploitability(game, average) < 0.2
