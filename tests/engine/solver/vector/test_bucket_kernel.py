"""The board-free kernel, where chance is a matrix instead of a card.

The risky part is not the CFR math — that is shared with the hand-space kernel
and already tested. It is the street boundary, where a range is carried forward
by a transition matrix and a counterfactual value is carried back by its
transpose. Get the transpose wrong and nothing crashes: values stay finite,
regrets still move, and the solver converges to the wrong game.

So the boundary is tested as what it actually is — an adjoint pair — and the
whole kernel is tested by zero-sum, which is a joint statement about terminal
signs, transitions and value propagation at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.vector import bucket_game, compile_tree
from src.engine.solver.vector.bucket_kernel import BucketVectorCFR
from tests.engine.solver.vector.contexts import ordered_context
from tests.test_helpers import make_test_config

COUNTS = {Street.FLOP: 4, Street.TURN: 5, Street.RIVER: 6}
ALL_COUNTS = {Street.PREFLOP: 169, **COUNTS}
STACK = 12


def _context(rng):
    """Strength-ordered, so showdowns carry signal — see ``contexts``."""
    return ordered_context(rng, ALL_COUNTS)


def _build_solver() -> BucketVectorCFR:
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(small_blind=1, big_blind=2)
    tree = BettingTree(rules, ActionModel(config), starting_stack=STACK, buckets_per_street=COUNTS)
    compiled = compile_tree(tree, rules)

    rng = np.random.default_rng(31)
    game = bucket_game.derive([_context(rng) for _ in range(6)], ALL_COUNTS)
    return BucketVectorCFR(compiled, game, cfr_plus=True)


@pytest.fixture(scope="module")
def solver():
    return _build_solver()


@pytest.fixture(scope="module")
def initial(solver):
    return np.ones(solver.compiled.tree.num_buckets(Street.PREFLOP), dtype=np.float32)


class TestStreetBoundary:
    def test_advance_and_retreat_are_adjoint(self, solver):
        """<advance(range), value> == <range, retreat(value)>.

        This is the exact condition under which the backward pass computes the
        counterfactual value of the forward pass it is paired with. A transposed
        or inverted transition satisfies neither side of it, and no other check
        in this suite would notice.
        """
        rng = np.random.default_rng(2)
        for source, target in (
            (Street.PREFLOP, Street.FLOP),
            (Street.FLOP, Street.RIVER),
            (Street.TURN, Street.TURN),
        ):
            width_in = solver.compiled.tree.num_buckets(source)
            width_out = solver.compiled.tree.num_buckets(target)
            ranges = rng.random((3, width_in)).astype(np.float32)
            values = rng.random((3, width_out)).astype(np.float32)

            forward = float((solver._advance(ranges, source, target) * values).sum())
            backward = float((ranges * solver._retreat(values, source, target)).sum())
            assert forward == pytest.approx(backward, rel=1e-4)

    def test_advancing_conserves_range_mass(self, solver):
        """Transitions are row-stochastic, so no range mass is created or lost."""
        rng = np.random.default_rng(4)
        width = solver.compiled.tree.num_buckets(Street.FLOP)
        ranges = rng.random((5, width)).astype(np.float32)
        moved = solver._advance(ranges, Street.FLOP, Street.RIVER)
        assert np.allclose(moved.sum(axis=1), ranges.sum(axis=1), rtol=1e-4)


class TestKernel:
    def test_root_values_cancel(self, solver, initial):
        """Zero-sum, jointly over terminal signs, transitions and propagation.

        The tolerance is relative to the total *absolute* value at the root, not
        to the signed sum — the signed sum is the thing being asserted to be zero
        and cancels almost entirely, so scaling by it would be circular.

        Both precisions are pinned, and the gap between them is the whole reason
        ``dtype`` is a parameter: the root value is a near-total cancellation of
        intermediates four orders larger, which costs float32 about four
        significant digits. A genuine sign error leaves a residual the size of
        the values themselves, so either bound catches it by orders of magnitude
        — but only float64 can tell a subtle asymmetry from rounding.
        """
        width = solver.compiled.tree.num_buckets(Street.PREFLOP)
        for dtype, tolerance in ((np.float32, 1e-3), (np.float64, 1e-9)):
            kernel = BucketVectorCFR(solver.compiled, solver.game, cfr_plus=True, dtype=dtype)
            for _ in range(4):
                kernel.iterate(initial)
                button = kernel.value[0, 0, :width]
                other = kernel.value[1, 0, :width]
                residual = abs(float(button.sum()) + float(other.sum()))
                scale = float(np.abs(button).sum() + np.abs(other).sum())
                assert residual <= tolerance * max(scale, 1.0), dtype

    def test_the_strategy_is_a_distribution_everywhere(self, solver, initial):
        solver.iterate(initial)
        for group in solver.groups:
            block = solver._strategy_block(group, slice(0, group.node_ids.shape[0]))
            assert np.allclose(block.sum(axis=-1), 1.0, atol=1e-4)
            assert (block >= 0).all()

    def test_regrets_stay_finite(self, solver, initial):
        for _ in range(3):
            solver.iterate(initial)
        assert np.isfinite(solver.regrets).all()
        assert np.isfinite(solver.strategy_sum).all()

    # 5.8s single-process for 200 iterations, and the suite runs 12-way
    # parallel, so the default 5s fires on a loaded box. Bounded rather than
    # marked slow: this is the check that the kernel solves the game it plays,
    # and it should run on every commit.
    @pytest.mark.timeout(60)
    def test_abstract_game_exploitability_falls_toward_zero(self, initial):
        """The kernel must solve the game it is actually playing.

        Distinct from the real-game floor: the averaging in ``bucket_game``
        stops the *real* exploitability below ~0.29, but exploitability inside
        the abstract game has no such floor and must decay. If it did not, the
        floor would be a solver defect rather than an abstraction cost, and the
        whole conclusion would change.

        A FRESH solver: the shared one has been iterated by earlier tests in
        file order, which moves the iteration-0 reading the ratio is taken from
        and made this pass or fail with test ordering.
        """
        solver = _build_solver()
        history = []
        target = 0
        for target in (0, 10, 50, 200):
            while solver.iteration < target:
                solver.iterate(initial)
            history.append(solver.exploitability(initial))

        assert history == sorted(history, reverse=True)
        assert history[-1] < history[0] / 8
        assert all(gain > 0 for gain in history)

    def test_it_writes_the_same_table_the_hand_space_kernel_reads(self, solver):
        """Storage is the shared ragged layout, which is what makes the
        cross-evaluation in the results writeup possible at all."""
        assert solver.regrets.shape == (solver.compiled.tree.num_slots,)
        assert solver.strategy_sum.shape == (solver.compiled.tree.num_slots,)
