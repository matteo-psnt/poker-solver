"""The scalar arm has to solve the SAME game as the vector arms.

The whole comparison rests on it. The shipped kernel samples a fresh runout from
the deck every iteration; the vector kernels solve a fixed K-runout mixture. Left
alone they are answering different questions, and whichever wins would have won
on the easier problem rather than the better kernel.

So what is pinned here is not the CFR math — that is inherited untouched and
tested elsewhere — but the two properties that make the comparison fair: the
board comes from the given set, and one hand uses one board all the way down.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.vector.fixed_board_scalar import FixedBoardStaticSolver
from tests.test_helpers import make_test_config

COUNTS = {Street.FLOP: 4, Street.TURN: 5, Street.RIVER: 6}
STACK = 12


class StubAbstraction:
    """Cheap deterministic bucketing; the abstraction is not what is under test."""

    def get_bucket(self, hole_cards, board, street):
        if street == Street.PREFLOP:
            return abs(hash(repr(hole_cards[0]))) % 169
        return abs(hash((repr(hole_cards[0]), repr(board[0]), street.name))) % COUNTS[street]

    def num_buckets(self, street):
        return 169 if street == Street.PREFLOP else COUNTS[street]


@pytest.fixture(scope="module")
def parts():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(1, 2)
    action_model = ActionModel(config)
    tree = BettingTree(rules, action_model, starting_stack=STACK, buckets_per_street=COUNTS)
    rng = np.random.default_rng(3)
    runouts = [tuple(FULL_DECK[int(c)] for c in rng.choice(52, 5, replace=False)) for _ in range(4)]
    return config, action_model, tree, runouts


def _solver(parts):
    config, action_model, tree, runouts = parts
    storage = StaticArrayStorage(tree)
    solver = FixedBoardStaticSolver(
        action_model, StubAbstraction(), storage, config, tree=tree, runouts=runouts
    )
    return solver, storage, runouts


def test_it_only_ever_deals_the_given_runouts(parts):
    solver, storage, runouts = _solver(parts)
    try:
        random.seed(0)
        seen = set()
        for _ in range(300):
            solver.train_iteration()
            seen.add(solver._current)
        assert seen <= set(runouts)
        assert len(seen) > 1, "a solver stuck on one runout is not sampling chance at all"
    finally:
        storage.close()


def test_hole_cards_never_collide_with_the_board(parts):
    """A hand dealt a card that is on the board is not a legal deal.

    The shipped dealer draws four cards from the whole deck because it draws the
    board later; pinning the board first inverts that, and the exclusion has to
    move with it.
    """
    solver, storage, _ = _solver(parts)
    try:
        random.seed(1)
        for _ in range(200):
            state = solver.deal_initial_state()
            held = {card for hand in state.hole_cards for card in hand}
            assert not held & set(solver._current)
    finally:
        storage.close()


def test_one_hand_uses_one_board_on_every_street(parts):
    """Streets must reveal prefixes of the same runout, not fresh cards.

    Sampling per street would deal boards that are not in the set at all — a
    far larger chance layer than the one the vector arms solve, which is exactly
    the unfairness this class exists to remove.
    """
    solver, storage, _ = _solver(parts)
    try:
        random.seed(2)
        state = solver.deal_initial_state()
        runout = solver._current
        for street, size in ((Street.FLOP, 3), (Street.TURN, 4), (Street.RIVER, 5)):
            advanced = solver.sample_chance_outcome(
                state.replace(street=street, board=(), validate=False)
            )
            assert advanced.board == runout[:size]
    finally:
        storage.close()


def test_it_writes_the_table_the_vector_kernels_write(parts):
    """Same flat (node, bucket, action) storage, so one scorer reads either."""
    solver, storage, _ = _solver(parts)
    try:
        random.seed(3)
        for _ in range(200):
            solver.train_iteration()
        assert storage.strategy_sum.shape == (solver.tree.num_slots,)
        assert (storage.strategy_sum != 0).any()
        assert solver.dropped_unknown_id_updates == 0
    finally:
        storage.close()


def test_an_empty_runout_set_is_refused(parts):
    config, action_model, tree, _ = parts
    storage = StaticArrayStorage(tree)
    try:
        with pytest.raises(ValueError, match="at least one runout"):
            FixedBoardStaticSolver(
                action_model, StubAbstraction(), storage, config, tree=tree, runouts=[]
            )
    finally:
        storage.close()


def test_it_deals_the_tree_s_stack_not_the_config_s(parts):
    """A state dealt at the wrong stack has no node in the tree.

    The betting tree is enumerated for ONE starting stack and every infoset id
    derives from it. Dealing a different one produced `Illegal action ALL_IN`
    from deep inside a traversal -- a failure that names neither the stack nor
    the tree, and only appeared once a real config disagreed with --stack.
    """
    config, action_model, tree, runouts = parts
    storage = StaticArrayStorage(tree)
    try:
        mismatched = config.model_copy(
            update={"game": config.game.model_copy(update={"starting_stack": 200})}
        )
        solver = FixedBoardStaticSolver(
            action_model, StubAbstraction(), storage, mismatched, tree=tree, runouts=runouts
        )
        random.seed(4)
        state = solver.deal_initial_state()
        assert sum(state.stacks) + state.pot == 2 * tree.starting_stack
        for _ in range(50):
            solver.train_iteration()
    finally:
        storage.close()
