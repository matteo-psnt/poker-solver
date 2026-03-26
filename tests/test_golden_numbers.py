"""Exact numbers a refactor is not allowed to move.

The rest of the suite asks "does it work"; these ask "does it give the SAME
ANSWER". A failure here is a LINEAGE BREAK, not a stale constant: every score
recorded before the change is incomparable with every score after. Update a
value only alongside a note on why the shift is correct, and re-baseline.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline.evaluation.public_tree_br import PublicBRConfig, compute_public_tree_br
from tests.test_helpers import make_test_config

BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}


class GoldenBuckets:
    """Keys on rank, not ``hash()`` -- which is randomised per process."""

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        return (hole_cards[0].rank_eval7() + board[0].rank_eval7()) % BUCKETS[street]

    def num_buckets(self, street: Street) -> int:
        return BUCKETS[street]


def _trained_solver(iterations: int) -> StaticTreeSolver:
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    action_model = ActionModel(config)
    abstraction = GoldenBuckets()
    tree = build_betting_tree(GameRules(1, 2), action_model, abstraction, starting_stack=20)
    solver = StaticTreeSolver(
        action_model, abstraction, StaticArrayStorage(tree), config, tree=tree
    )
    random.seed(1)
    np.random.seed(1)
    for _ in range(iterations):
        solver.train_iteration()
    return solver


def test_tree_shape_is_pinned():
    """The tree IS the infoset space; a change here silently rescales coverage."""
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    tree = build_betting_tree(
        GameRules(1, 2), ActionModel(config), GoldenBuckets(), starting_stack=20
    )
    assert len(tree) == 2140
    assert tree.num_rows == 9432


def test_training_reaches_the_same_state():
    """Pins the traversal: action ordering and chance sampling move these
    long before they move a rounded score."""
    solver = _trained_solver(400)
    try:
        assert solver.storage.num_touched_infosets() == 3114
        assert int(solver.storage.reach_counts.sum()) == 2842
    finally:
        solver.storage.close()


@pytest.mark.timeout(120)
def test_exact_br_is_bit_stable():
    """The zero-variance scorer. Every convergence curve is denominated in this."""
    solver = _trained_solver(400)
    try:
        result = compute_public_tree_br(
            solver,
            PublicBRConfig(num_flops=2, num_turns=1, num_rivers=1, board_seed=7),
            starting_stack=20,
        )
        assert result.exploitability_mbb == pytest.approx(1658.5536976402323, abs=1e-9)
        assert result.missing_policy_mass == pytest.approx(0.19440688553774604, abs=1e-12)
    finally:
        solver.storage.close()
