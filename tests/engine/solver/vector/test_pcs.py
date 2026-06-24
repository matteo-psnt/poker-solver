"""Public chance sampling is the fixed-board kernel, one sampled board at a time.

Three things have to hold or the trainer is solving something else: a one-board
iteration IS the fixed-board kernel's iteration on that board; the expected
update over the boards it samples IS the enumerating mixture's update; and the
regret and average-strategy bookkeeping is the production scalar kernel's
DCFR, not the vector kernel's CFR+. The last is pinned against ``numba_ops``
directly, so the two trainers cannot drift apart in what a stored number means.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.numba_ops import apply_regret_updates, compute_dcfr_strategy_weight
from src.engine.solver.vector.compiled_tree import compile_tree
from src.engine.solver.vector.hand_context import HandContext
from src.engine.solver.vector.kernel import VectorCFR
from src.engine.solver.vector.mixture import BoardMixtureCFR
from src.engine.solver.vector.pcs import PublicChanceSamplingCFR, dcfr_discount
from tests.engine.solver.vector.contexts import (
    MIN_SHOWDOWN_SIGNAL,
    ordered_context,
    showdown_signal,
)
from tests.test_helpers import make_test_config

DECK = 36
BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}
ALL_COUNTS = {Street.PREFLOP: 169, **BUCKETS}
STACK = 12


@pytest.fixture(scope="module")
def parts():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(small_blind=1, big_blind=2)
    tree = BettingTree(rules, ActionModel(config), starting_stack=STACK, buckets_per_street=BUCKETS)
    compiled = compile_tree(tree, rules)
    rng = np.random.default_rng(23)
    contexts = [ordered_context(rng, ALL_COUNTS, num_cards=DECK) for _ in range(3)]
    assert showdown_signal(contexts, ALL_COUNTS) > MIN_SHOWDOWN_SIGNAL
    pairs = float(np.mean([(~c.blocks).sum() for c in contexts]))
    return compiled, contexts, pairs


def _pcs(compiled, **options):
    slots = compiled.tree.num_slots
    regrets = np.zeros(slots, dtype=np.float32)
    strategy_sum = np.zeros(slots, dtype=np.float32)
    return (
        PublicChanceSamplingCFR(compiled, regrets, strategy_sum, **options),
        regrets,
        strategy_sum,
    )


class TestOneBoardIsTheFixedBoardKernel:
    def test_a_cfr_plus_iteration_matches_the_kernel_bit_for_bit(self, parts):
        compiled, contexts, _ = parts
        pcs, regrets, strategy_sum = _pcs(
            compiled, weighting="none", cfr_plus=True, showdown="matmul"
        )
        pcs.iterate([contexts[0]], 0)

        reference = VectorCFR(compiled, contexts[0], cfr_plus=True)
        reference.iterate(np.ones(contexts[0].num_hands, dtype=np.float32))

        assert np.array_equal(regrets, reference.regrets)
        assert np.array_equal(strategy_sum, reference.strategy_sum)

    def test_the_rank_walk_showdown_agrees_with_the_matrix_product(self, parts):
        """Same terminal values to float rounding; the walk is O(H), the product O(H^2)."""
        compiled, contexts, _ = parts
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)
        product = VectorCFR(compiled, contexts[0], cfr_plus=True, showdown="matmul")
        walk = VectorCFR(compiled, contexts[0], cfr_plus=True, showdown="walk")
        for kernel in (product, walk):
            kernel.forward(initial)
            kernel.evaluate_terminals()
        scale = float(np.abs(product.terminal_value).max())
        assert scale > 0
        assert np.allclose(walk.terminal_value, product.terminal_value, atol=1e-4 * scale)


class TestSamplingIsTheMixtureInExpectation:
    def test_summed_one_board_updates_equal_the_enumerated_update(self, parts):
        """Each board read from the same zero table: the sum of the single-board
        increments is the mixture's increment, so a uniform draw is unbiased."""
        compiled, contexts, _ = parts
        summed = np.zeros(compiled.tree.num_slots, dtype=np.float64)
        for context in contexts:
            pcs, regrets, _ = _pcs(compiled, weighting="none", cfr_plus=False, showdown="matmul")
            pcs.iterate([context], 0)
            summed += regrets

        mixture = BoardMixtureCFR(compiled, contexts)
        mixture.iterate(np.ones(contexts[0].num_hands, dtype=np.float32))
        assert np.allclose(np.maximum(summed, 0.0), mixture.regrets, rtol=1e-4, atol=1e-3)

    def test_runouts_under_one_flop_average_before_the_floor(self, parts):
        """K contexts in one iteration is the mixture's joint increment over K."""
        compiled, contexts, _ = parts
        pcs, regrets, _ = _pcs(compiled, weighting="none", cfr_plus=True, showdown="matmul")
        pcs.iterate(contexts, 0)

        mixture = BoardMixtureCFR(compiled, contexts)
        mixture.iterate(np.ones(contexts[0].num_hands, dtype=np.float32))
        assert np.allclose(regrets, mixture.regrets / len(contexts), rtol=1e-4, atol=1e-3)


class TestProductionBookkeeping:
    def test_the_discount_is_the_scalar_kernels(self):
        """One row through ``apply_regret_updates`` and through ``dcfr_discount``."""
        for iteration in (0, 1, 2, 7, 5000):
            stored = np.array([3.0, -2.0, 0.0], dtype=np.float64)
            row = stored.copy()
            apply_regret_updates(
                row, np.arange(3), np.zeros(3), 0.0, 1.0, False, iteration, 2, 1.5, 0.0
            )
            factors = dcfr_discount(iteration, 1.5, 0.0)
            positive, negative = (1.0, 1.0) if factors is None else factors
            expected = np.where(stored > 0, stored * positive, stored * negative)
            assert np.allclose(row, expected)

    def test_a_dcfr_iteration_discounts_occupied_rows_then_adds(self, parts):
        """Against the kernel's own increment: stored rows this board occupies
        are discounted, rows it does not are left alone, then the increment
        lands with no floor and the strategy sum carries t^gamma."""
        compiled, contexts, _ = parts
        # Vacate river bucket 2 on this board, so the mask has a row to spare.
        buckets = contexts[0].bucket_of_hand.copy()
        river = buckets[Street.RIVER.value - 1]
        river[river == 2] = 3
        context = HandContext(
            contexts[0].hand_cards, buckets, contexts[0].showdown_rank, contexts[0].blocks
        )
        iteration = 9
        rng = np.random.default_rng(5)
        start = rng.standard_normal(compiled.tree.num_slots).astype(np.float32)

        pcs, regrets, strategy_sum = _pcs(
            compiled, weighting="dcfr", cfr_plus=False, showdown="matmul"
        )
        regrets[:] = start
        pcs.iterate([context], iteration)

        reference = VectorCFR(compiled, context, cfr_plus=False)
        reference.regrets[:] = start
        delta = np.zeros(compiled.tree.num_slots, dtype=np.float32)
        reference.regret_target = delta
        reference.iterate(np.ones(context.num_hands, dtype=np.float32))

        factors = dcfr_discount(iteration, 1.5, 0.0)
        assert factors is not None
        positive, negative = factors
        discounted = np.where(start > 0, start * positive, start * negative).astype(np.float32)
        occupied = np.zeros(compiled.tree.num_slots, dtype=bool)
        tree = compiled.tree
        for node in tree.nodes:
            present = np.unique(context.buckets_for(node.street))
            width = int(tree.num_actions[node.node_id])
            base = int(tree.slot_offset[node.node_id])
            for bucket in present:
                occupied[base + bucket * width : base + (bucket + 1) * width] = True
        expected = np.where(occupied, discounted, start) + delta
        assert np.allclose(regrets, expected, rtol=1e-4, atol=1e-3)
        assert not occupied.all(), "the fixture occupies every row, so the mask is untested"

        weight = compute_dcfr_strategy_weight(iteration, 2.0)
        assert weight == iteration**2
        assert np.allclose(strategy_sum, reference.strategy_sum * weight, rtol=1e-4)

    def test_alternating_updates_leave_the_other_players_rows_alone(self, parts):
        compiled, contexts, _ = parts
        pcs, regrets, strategy_sum = _pcs(
            compiled, weighting="none", alternating=True, showdown="matmul"
        )
        pcs.iterate([contexts[0]], 1)

        actor_of_slot = np.repeat(
            [0 if node.actor_is_button else 1 for node in compiled.tree.nodes],
            compiled.tree.buckets_per_node * compiled.tree.num_actions,
        )
        assert not regrets[actor_of_slot == 0].any()
        assert not strategy_sum[actor_of_slot == 0].any()
        assert regrets[actor_of_slot == 1].any()


class TestConvergence:
    @pytest.mark.timeout(60)
    def test_sampling_from_a_small_universe_drives_its_exploitability_down(self, parts):
        """The mixture over three boards is the game; sampling them drives the
        in-abstraction exploitability of that game down under production DCFR."""
        compiled, contexts, pairs = parts
        pcs, _, strategy_sum = _pcs(compiled, weighting="dcfr", cfr_plus=False)
        scorer = BoardMixtureCFR(compiled, contexts)
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)
        rng = np.random.default_rng(11)

        def score() -> float:
            scorer.strategy_sum[:] = strategy_sum
            return scorer.exploitability(initial, pairs)

        pcs.iterate([contexts[rng.integers(len(contexts))]], 0)
        early = score()
        for iteration in range(1, 96):
            pcs.iterate([contexts[rng.integers(len(contexts))]], iteration)
        later = score()
        assert 0 < later < early / 3
