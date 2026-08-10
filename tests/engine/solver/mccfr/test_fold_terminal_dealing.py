"""Folds must not be dealt a runout — and the payoff must not care.

``_terminal_utility`` routes every terminal through ``deal_remaining_cards``.
For a fold that used to complete the board at random, which is unobservable in
the payoff (``get_payoff`` resolves a fold from the folder's identity) but is
paid for at 34% of all terminal evaluations.

Skipping it is only safe if the payoff is genuinely board-independent there, so
that is what these tests assert directly rather than trusting the reading of
``get_payoff``.

Note this changes RNG consumption: seeded runs no longer draw cards at fold
terminals, so a given seed produces a different (equally valid) sample path than
before. The invariant preserved is the payoff, not the stream.
"""

from __future__ import annotations

import random

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from tests.test_helpers import make_test_config

COUNTS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}


class Buckets:
    def get_bucket(self, hole_cards, board, street):
        return (hole_cards[0].rank_eval7() + board[0].rank_eval7()) % COUNTS[street]

    def num_buckets(self, street):
        return COUNTS[street]


@pytest.fixture(scope="module")
def setup():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    action_model = ActionModel(config)
    rules = GameRules(small_blind=config.game.small_blind, big_blind=config.game.big_blind)
    abstraction = Buckets()
    tree = build_betting_tree(
        rules, action_model, abstraction, starting_stack=config.game.starting_stack
    )
    return config, action_model, rules, abstraction, tree


def _collect_fold_terminals(config, action_model, rules, limit=40):
    """Play random hands, keeping fold terminals that still have an open board."""
    found = []
    rng = random.Random(5)
    deck = list(FULL_DECK)
    trial = 0
    while len(found) < limit and trial < 4000:
        trial += 1
        rng.shuffle(deck)
        state = rules.create_initial_state(
            config.game.starting_stack, ((deck[0], deck[1]), (deck[2], deck[3])), button=trial % 2
        )
        board_pool = deck[4:9]
        guard = 0
        while not state.is_terminal and guard < 40:
            guard += 1
            needed = state.street.board_card_count
            if len(state.board) < needed:
                state = state.replace(board=tuple(board_pool[:needed]), validate=False)
                continue
            actions = rules.get_legal_actions(state, action_model=action_model)
            if not actions:
                break
            state = rules.apply_action(state, rng.choice(actions))
        if state.is_terminal and state.ended_by_fold and len(state.board) < 5:
            found.append(state)
    return found


class TestFoldPayoffIsBoardIndependent:
    def test_fold_terminals_exist_in_the_sample(self, setup):
        config, action_model, rules, _, _ = setup
        folds = _collect_fold_terminals(config, action_model, rules)
        assert len(folds) >= 10, "sample produced too few fold terminals to prove anything"

    def test_payoff_identical_under_every_runout(self, setup):
        """The load-bearing claim: no completion of the board moves the payoff."""
        config, action_model, rules, _, _ = setup
        folds = _collect_fold_terminals(config, action_model, rules)
        assert folds, "no fold terminals collected — this test would prove nothing"
        rng = random.Random(11)

        for state in folds:
            used = {repr(c) for c in state.board}
            for hole in state.hole_cards:
                used.update(map(repr, hole))
            available = [c for c in FULL_DECK if repr(c) not in used]
            needed = 5 - len(state.board)

            baseline = [rules.get_payoff(state, p) for p in (0, 1)]
            for _ in range(15):
                completed = state.replace(
                    street=Street.RIVER,
                    board=(*state.board, *rng.sample(available, needed)),
                    is_terminal=True,
                    to_call=0,
                    validate=False,
                )
                assert [rules.get_payoff(completed, p) for p in (0, 1)] == baseline


class TestSolverSkipsTheDeal:
    def test_deal_remaining_cards_leaves_folds_alone(self, setup):
        config, action_model, rules, abstraction, tree = setup
        solver = StaticTreeSolver(
            action_model, abstraction, StaticArrayStorage(tree), config, tree=tree
        )
        try:
            folds = _collect_fold_terminals(config, action_model, rules)
            for state in folds:
                assert solver.deal_remaining_cards(state) is state
        finally:
            solver.storage.close()

    def test_showdowns_are_still_completed(self, setup):
        """Guard the other direction: real showdowns must still get a full board."""
        config, action_model, rules, abstraction, tree = setup
        solver = StaticTreeSolver(
            action_model, abstraction, StaticArrayStorage(tree), config, tree=tree
        )
        try:
            rng = random.Random(3)
            deck = list(FULL_DECK)
            rng.shuffle(deck)
            allin = rules.create_initial_state(
                config.game.starting_stack, ((deck[0], deck[1]), (deck[2], deck[3])), button=0
            )
            allin = allin.replace(is_terminal=True, to_call=0, validate=False)
            assert not allin.ended_by_fold
            completed = solver.deal_remaining_cards(allin)
            assert len(completed.board) == 5
        finally:
            solver.storage.close()

    def test_training_still_runs_and_writes(self, setup):
        config, action_model, _, abstraction, tree = setup
        solver = StaticTreeSolver(
            action_model, abstraction, StaticArrayStorage(tree), config, tree=tree
        )
        try:
            random.seed(1)
            for _ in range(80):
                solver.train_iteration()
            assert solver.num_infosets() > 0
            assert solver.storage.regrets.any()
        finally:
            solver.storage.close()
