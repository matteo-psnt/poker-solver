"""The core promise, and the exact numbers a refactor may not move.

Two kinds of guard, both about the thing that matters most -- that this still
trains a poker solver:

* MORE TRAINING MUST LOWER EXPLOITABILITY. If that stops holding, the solver is
  not solving, however green everything else is.
* The scorers must give the SAME ANSWER. A failure there is a LINEAGE BREAK, not
  a stale constant: every score recorded before the change is incomparable with
  every score after. Update a value only alongside a note on why the shift is
  correct, and expect to re-baseline.

The kernel itself is proven separately, against games with known analytic
equilibria, in ``tests/engine/solver/mccfr/test_kernel_conformance.py``. These
cover the HUNL path that harness cannot reach.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline.evaluation.estimators.lbr.hunl_local_best_response import (
    LBRConfig,
    compute_lbr_exploitability,
)
from src.pipeline.evaluation.estimators.public_tree_br import PublicBRConfig, compute_public_tree_br
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
    # Re-pinned 2026-08-23: the game gained the big blind's option behind a
    # limp (lineage break, deliberate). Was 2140 nodes / 9432 rows.
    assert len(tree) == 2570
    assert tree.num_rows == 12634


def test_training_reaches_the_same_state():
    """Pins the traversal: action ordering and chance sampling move these
    long before they move a rounded score."""
    solver = _trained_solver(400)
    try:
        # Re-pinned 2026-08-23 with the BB-option tree (was 3114 / 2842).
        assert solver.storage.num_touched_infosets() == 3930
        assert int(solver.storage.reach_counts.sum()) == 3341
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
        # Re-pinned 2026-08-23 with the BB-option tree (was 1658.5536976402323 /
        # 0.19440688553774604).
        assert result.exploitability_mbb == pytest.approx(1735.5097100061157, abs=1e-9)
        assert result.missing_policy_mass == pytest.approx(0.2195317024184147, abs=1e-12)
    finally:
        solver.storage.close()


@pytest.mark.timeout(120)
def test_more_training_lowers_exploitability():
    """The core promise: this trains a poker solver.

    Measured with exact BR because it has zero evaluation variance -- a sampled
    scorer would need many more hands before a real improvement outran its own
    noise, and a flaky version of THIS test is worse than none.
    """
    config = PublicBRConfig(num_flops=2, num_turns=1, num_rivers=1, board_seed=7)
    scores = []
    # 400 up, not 100: on the BB-option tree 100 and 400 iterations are both
    # inside the fallback-dominated regime (53% / 22% uniform mass) and read
    # 1730.9 / 1735.5 -- noise, not training. From 400 the ladder falls
    # 1735.5 -> 1611.0 -> 1461.8 (-> 1315.2 at 4000).
    for iterations in (400, 1000, 2000):
        solver = _trained_solver(iterations)
        try:
            scores.append(compute_public_tree_br(solver, config, starting_stack=20))
        finally:
            solver.storage.close()

    mbb = [s.exploitability_mbb for s in scores]
    # STRICTLY decreasing. Non-strict would pass on a solver that learns
    # nothing: zero the regret updates and every budget scores an identical
    # 1762.9 (uniform everywhere), which is "sorted descending" and tells you
    # only that the tree got more covered, not that the strategy got better.
    assert all(a > b for a, b in itertools.pairwise(mbb)), (
        f"exploitability must FALL with training, got {mbb}"
    )
    # Coverage rises too. Not evidence of learning on its own -- it rises under
    # the sabotage above as well -- but a score that fell while coverage shrank
    # would mean the gain came from exploring less, not from playing better.
    assert scores[-1].missing_policy_mass < scores[0].missing_policy_mass


@pytest.mark.timeout(120)
def test_lbr_is_bit_stable():
    """LBR is the project's default metric and had no pin.

    Only a stability pin, not a convergence one: at this hand count LBR's
    sampling noise swamps the improvement, so it cannot say training helped --
    ``test_more_training_lowers_exploitability`` is what says that.
    """
    solver = _trained_solver(400)
    try:
        result = compute_lbr_exploitability(
            solver, LBRConfig(num_hands=12, equity_runouts=2, seed=7)
        )
        # Re-pinned 2026-08-23 with the BB-option tree (was 947.4978980029194).
        assert result.exploitability_mbb == pytest.approx(574.0163194436639, abs=1e-9)
        assert result.num_hands == 12
    finally:
        solver.storage.close()
