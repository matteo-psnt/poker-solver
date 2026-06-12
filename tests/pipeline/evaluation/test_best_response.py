"""Validation of the exact best-response / exploitability harness on Kuhn poker.

These anchor the evaluation machinery to ground truth before it is trusted on
HUNL: an exactly hand-computable best response, and CFR converging to Kuhn's
known game value of -1/18 at ~0 exploitability. A best response that violated
information-set consistency (peeking at the opponent's card) would report
positive exploitability against the equilibrium and fail these tests.
"""

from __future__ import annotations

import pytest

from src.pipeline.evaluation.best_response import (
    best_response_value,
    exploitability,
    on_policy_value,
)
from src.pipeline.evaluation.game_tree import TabularStrategy
from src.pipeline.evaluation.tabular_cfr import TabularCFRSolver
from tests.pipeline.evaluation.kuhn_poker import KuhnPoker

KUHN_GAME_VALUE_P0 = -1.0 / 18.0


def _always_pass_strategy() -> TabularStrategy:
    """A strategy that passes with probability 1 at every information set."""

    class _AlwaysPass(TabularStrategy):
        def __call__(self, info_key, legal_actions):
            return [1.0 if action == "p" else 0.0 for action in legal_actions]

    return _AlwaysPass()


class TestExactBestResponseGroundTruth:
    def test_always_pass_best_response_is_exact(self):
        """Against an all-pass opponent, betting always wins exactly 1 chip.

        BR for either player is to bet: the opponent then either folds to the
        bet (P0 bets -> "bp") or is bet into after checking (P1 bets -> "pbp"),
        winning 1 chip regardless of cards.
        """
        game = KuhnPoker()
        policy = _always_pass_strategy()
        assert best_response_value(game, 0, policy) == pytest.approx(1.0)
        assert best_response_value(game, 1, policy) == pytest.approx(1.0)

    def test_always_pass_on_policy_value_is_zero(self):
        """Both players checking to showdown is symmetric -> zero EV."""
        game = KuhnPoker()
        policy = _always_pass_strategy()
        assert on_policy_value(game, 0, policy) == pytest.approx(0.0)
        assert on_policy_value(game, 1, policy) == pytest.approx(0.0)

    def test_always_pass_exploitability_is_one(self):
        """NashConv per player = ((1 - 0) + (1 - 0)) / 2 = 1."""
        game = KuhnPoker()
        assert exploitability(game, _always_pass_strategy()) == pytest.approx(1.0)


class TestExploitabilityProperties:
    def test_uniform_random_is_exploitable(self):
        """A uniform-random strategy is far from equilibrium."""
        game = KuhnPoker()
        uniform = TabularStrategy()  # empty table -> uniform everywhere
        assert exploitability(game, uniform) > 0.05

    def test_exploitability_is_non_negative(self):
        """Best response can never do worse than on-policy: NashConv >= 0."""
        game = KuhnPoker()
        uniform = TabularStrategy()
        assert exploitability(game, uniform) >= -1e-12


@pytest.fixture(scope="module")
def converged_strategy() -> TabularStrategy:
    """CFR average strategy for Kuhn, trained once and shared across tests."""
    solver = TabularCFRSolver(KuhnPoker())
    solver.train(20000)
    return solver.average_strategy()


@pytest.mark.slow
class TestHarnessConvergence:
    @pytest.mark.timeout(30)
    def test_cfr_converges_to_kuhn_equilibrium(self, converged_strategy):
        """Full-tree CFR should reach ~0 exploitability and the -1/18 value.

        This exercises the entire harness together: CFR, on-policy evaluation,
        and the exact best response. Reaching the analytically known game value
        (-1/18 for the first player) is a strong end-to-end correctness check.
        """
        game = KuhnPoker()
        assert exploitability(game, converged_strategy) < 5e-3
        assert on_policy_value(game, 0, converged_strategy) == pytest.approx(
            KUHN_GAME_VALUE_P0, abs=5e-3
        )

    @pytest.mark.timeout(30)
    def test_best_response_matches_on_policy_at_equilibrium(self, converged_strategy):
        """At equilibrium, a best responder gains nothing over on-policy play."""
        game = KuhnPoker()
        for player in (0, 1):
            br = best_response_value(game, player, converged_strategy)
            on_policy = on_policy_value(game, player, converged_strategy)
            assert br == pytest.approx(on_policy, abs=5e-3)
