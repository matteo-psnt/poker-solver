"""Validate the evaluation harness on Leduc — the mid-tree-chance ground truth.

Kuhn has only a root deal and one betting round. Leduc adds a public card dealt
mid-tree and a second round, so these tests exercise counterfactual reach and
value propagation through a non-root chance node for both the exact best
response and the reference CFR.
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
from tests.pipeline.evaluation.leduc_poker import LeducPoker

# Card ids: rank = id // 2, so 0,1=J  2,3=Q  4,5=K.
JACK, QUEEN, KING = 0, 2, 4


def _play(game: LeducPoker, deal, actions):
    """Apply a private deal followed by a sequence of (bet or public-card) actions."""
    state = game.next_state(game.initial_state(), deal)
    for action in actions:
        state = game.next_state(state, action)
    return state


class TestDeterministicPayoffs:
    def test_checkdown_higher_card_wins(self):
        """Ante-only showdown: higher private rank wins the pot (+1 / -1)."""
        game = LeducPoker()
        # P0=K, P1=J, public=Q. Check-check, deal public, check-check -> showdown.
        state = _play(game, (KING, JACK), ["c", "c", QUEEN, "c", "c"])
        assert game.is_terminal(state)
        assert tuple(game.returns(state)) == (1.0, -1.0)

    def test_fold_to_bet_forfeits_ante(self):
        """P0 bets round 0, P1 folds: P0 wins P1's committed (the ante, +1)."""
        game = LeducPoker()
        state = _play(game, (JACK, KING), ["r", "f"])
        assert game.is_terminal(state)
        assert tuple(game.returns(state)) == (1.0, -1.0)

    def test_pair_beats_higher_card(self):
        """A pair with the public card beats a higher unpaired card."""
        game = LeducPoker()
        # P0=J, P1=K, public=J -> P0 pairs and wins despite lower rank.
        state = _play(game, (JACK, KING), ["c", "c", 1, "c", "c"])
        assert game.is_terminal(state)
        assert tuple(game.returns(state)) == (1.0, -1.0)

    def test_raise_amounts_accumulate_into_pot(self):
        """A round-0 bet (2) called, then a round-1 bet (4) called, sizes the pot."""
        game = LeducPoker()
        # P0=K, P1=J. r0: P0 bet/called; public Q; r1: P0 bet/called; showdown.
        state = _play(game, (KING, JACK), ["r", "c", QUEEN, "r", "c"])
        assert game.is_terminal(state)
        # P0 contributed ante 1 + 2 + 4 = 7; wins P1's equal contribution.
        assert tuple(game.returns(state)) == (7.0, -7.0)


class TestExploitabilityProperties:
    def test_uniform_random_is_exploitable(self):
        game = LeducPoker()
        assert exploitability(game, TabularStrategy()) > 0.1

    def test_exploitability_non_negative(self):
        game = LeducPoker()
        assert exploitability(game, TabularStrategy()) >= -1e-9


# Uniform-random exploitability is ~2.37; a short CFR run must fall far below it.
_UNIFORM_EXPLOITABILITY = 2.37


@pytest.fixture(scope="module")
def converged_strategy() -> TabularStrategy:
    # 100 full-tree iterations cut exploitability ~19x (2.37 -> ~0.12), enough to
    # show CFR converges through the mid-tree chance node while staying fast.
    solver = TabularCFRSolver(LeducPoker())
    solver.train(100)
    return solver.average_strategy()


@pytest.mark.slow
class TestHarnessConvergence:
    @pytest.mark.timeout(60)
    def test_cfr_reduces_exploitability(self, converged_strategy):
        """CFR through the mid-tree chance node drives exploitability well down.

        If counterfactual reach or value propagation mishandled the non-root
        chance node, CFR would not approach equilibrium here.
        """
        game = LeducPoker()
        exploit = exploitability(game, converged_strategy)
        assert exploit < 0.2
        assert exploit < _UNIFORM_EXPLOITABILITY / 10

    @pytest.mark.timeout(60)
    def test_best_response_approaches_on_policy(self, converged_strategy):
        """Near equilibrium a best responder gains little over on-policy play."""
        game = LeducPoker()
        for player in (0, 1):
            gain = best_response_value(game, player, converged_strategy) - on_policy_value(
                game, player, converged_strategy
            )
            assert gain < 0.2
