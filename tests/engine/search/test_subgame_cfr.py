"""Tests for the range-vs-range subgame CFR.

The masses machinery (per-combo win/tie/alive vs a reach vector, with exact
card removal) is validated against an O(n^2) brute-force reference — that is
the correctness anchor for every terminal valuation in the solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.actions import call, check
from src.core.game.evaluator import get_evaluator
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street
from src.engine.search.range_inference import ALL_COMBOS, COMBO_MASKS, NUM_COMBOS
from src.engine.search.subgame_cfr import RunoutEvaluator, solve_subgame
from src.engine.search.tree_builder import build_local_tree
from tests.test_helpers import make_test_config


def _combo_index(a: str, b: str) -> int:
    target = Card.new(a).mask | Card.new(b).mask
    for idx in range(NUM_COMBOS):
        if int(COMBO_MASKS[idx]) == target:
            return idx
    raise AssertionError(f"combo {a}{b} not found")


class TestRunoutEvaluatorMasses:
    """masses() must match a brute-force pairwise computation exactly."""

    BOARD = (Card.new("Kh"), Card.new("8d"), Card.new("3c"), Card.new("Qs"), Card.new("2d"))

    def _brute_force(self, board, reach):
        evaluator = get_evaluator()
        board_mask = 0
        for card in board:
            board_mask |= card.mask
        alive = [i for i in range(NUM_COMBOS) if not (int(COMBO_MASKS[i]) & board_mask)]
        ranks = {i: evaluator.evaluate(ALL_COMBOS[i], board) for i in alive}
        support = [i for i in alive if reach[i] > 0]

        win = np.zeros(NUM_COMBOS)
        tie = np.zeros(NUM_COMBOS)
        total = np.zeros(NUM_COMBOS)
        for h in alive:
            for c in support:
                if c == h or (int(COMBO_MASKS[h]) & int(COMBO_MASKS[c])):
                    continue
                total[h] += reach[c]
                if ranks[c] > ranks[h]:  # larger rank = worse hand
                    win[h] += reach[c]
                elif ranks[c] == ranks[h]:
                    tie[h] += reach[c]
        return win, tie, total

    def test_matches_brute_force_on_sparse_reach(self):
        rng = np.random.default_rng(7)
        reach = np.zeros(NUM_COMBOS)
        board_mask = 0
        for card in self.BOARD:
            board_mask |= card.mask
        candidates = [i for i in range(NUM_COMBOS) if not (int(COMBO_MASKS[i]) & board_mask)]
        support = rng.choice(len(candidates), size=90, replace=False)
        for pos in support:
            reach[candidates[int(pos)]] = float(rng.uniform(0.1, 1.0))

        evaluator = RunoutEvaluator(self.BOARD)
        win, tie, alive = evaluator.masses(reach)
        ref_win, ref_tie, ref_total = self._brute_force(self.BOARD, reach)

        np.testing.assert_allclose(win, ref_win, atol=1e-9)
        np.testing.assert_allclose(tie, ref_tie, atol=1e-9)
        np.testing.assert_allclose(alive, ref_total, atol=1e-9)

    def test_blocked_combos_are_excluded(self):
        # Reach concentrated on one combo: any combo sharing a card sees zero mass.
        evaluator = RunoutEvaluator(self.BOARD)
        reach = np.zeros(NUM_COMBOS)
        held = _combo_index("As", "Td")
        reach[held] = 1.0
        win, tie, alive = evaluator.masses(reach)

        blocker = _combo_index("As", "9c")  # shares the As
        assert alive[blocker] == 0.0
        assert win[blocker] == 0.0 and tie[blocker] == 0.0

        independent = _combo_index("Jh", "Jc")
        assert alive[independent] == pytest.approx(1.0)

    def test_incomplete_board_rejected(self):
        with pytest.raises(ValueError, match="complete board"):
            RunoutEvaluator(self.BOARD[:4])


def _river_state(pot: int = 200, stack: int = 900) -> tuple[GameState, GameRules]:
    """Hand-built river node, first to act, no pending bet."""
    rules = GameRules(small_blind=50, big_blind=100)
    board = (Card.new("Kh"), Card.new("Kd"), Card.new("7s"), Card.new("2c"), Card.new("9h"))
    state = GameState(
        street=Street.RIVER,
        pot=pot,
        stacks=(stack, stack),
        board=board,
        hole_cards=((Card.new("Ks"), Card.new("Kc")), (Card.new("As"), Card.new("7h"))),
        betting_history=(call(), check()),
        button_position=0,
        current_player=0,
        is_terminal=False,
        to_call=0,
        last_aggressor=None,
        blind_to_call=100,
        _skip_validation=True,
    )
    return state, rules


class TestSolveSubgame:
    """CFR over a river tree: sane strategies, hand-strength monotonicity."""

    @pytest.mark.timeout(60)
    def test_nuts_bet_more_than_air(self):
        state, rules = _river_state()
        config = make_test_config(seed=42)
        tree = build_local_tree(state, action_model=ActionModel(config), rules=rules, max_depth=2)

        quads = _combo_index("Ks", "Kc")  # board KKxxx -> quad kings, the nuts
        air = _combo_index("5d", "4d")  # no pair, beats nothing
        hero_range = np.zeros(NUM_COMBOS)
        hero_range[quads] = 0.5
        hero_range[air] = 0.5
        opp_range = np.zeros(NUM_COMBOS)
        opp_range[_combo_index("As", "7h")] = 0.5  # kings and sevens
        opp_range[_combo_index("Th", "9c")] = 0.5  # kings and nines

        np.random.seed(11)
        solution = solve_subgame(
            tree,
            hero=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            rules=rules,
            budget_ms=1000,
            max_iterations=200,
        )

        assert solution.iterations == 200
        # Strategies are distributions.
        sums = solution.root_strategy[[quads, air]].sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

        aggressive = [i for i, action in enumerate(solution.root_actions) if action.amount > 0]
        assert aggressive, "river root should offer bet actions"
        bet_quads = solution.root_strategy[quads, aggressive].sum()
        bet_air = solution.root_strategy[air, aggressive].sum()
        assert bet_quads > bet_air
        assert bet_quads > 0.5  # the nuts must bet for value most of the time

    @pytest.mark.timeout(60)
    def test_deterministic_with_fixed_iterations(self):
        state, rules = _river_state()
        config = make_test_config(seed=42)
        action_model = ActionModel(config)

        def _solve():
            np.random.seed(3)
            tree = build_local_tree(state, action_model=action_model, rules=rules, max_depth=2)
            hero_range = np.zeros(NUM_COMBOS)
            hero_range[_combo_index("Ks", "Kc")] = 1.0
            opp_range = np.zeros(NUM_COMBOS)
            opp_range[_combo_index("As", "7h")] = 1.0
            return solve_subgame(
                tree,
                hero=0,
                hero_range=hero_range,
                opponent_range=opp_range,
                rules=rules,
                budget_ms=50,
                max_iterations=25,
            )

        first, second = _solve(), _solve()
        np.testing.assert_array_equal(first.root_strategy, second.root_strategy)
        np.testing.assert_array_equal(first.root_values, second.root_values)
