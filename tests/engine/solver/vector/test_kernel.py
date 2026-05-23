"""Does the vector kernel actually minimise regret, and is its collapse exact?

Three claims, in increasing order of how much they'd cost to get wrong:

1. The hand→bucket collapse is a plain segment sum. It is the one place the
   imperfect-recall abstraction enters the math, and ``reduceat`` over a sorted
   axis is an optimisation of a scatter-add — so it is checked against one.
2. The tree is zero-sum at the root. This is a joint statement about terminal
   signs, reach propagation and value propagation; a fold paying the wrong
   player, or a showdown matrix used untransposed, breaks it and almost nothing
   else would notice.
3. Exploitability inside the abstraction falls toward zero. This is the claim
   that the kernel is CFR and not merely arithmetic that runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.vector.compiled_tree import compile_tree
from src.engine.solver.vector.kernel import VectorCFR, build_segments
from tests.engine.solver.vector.contexts import ordered_context
from tests.test_helpers import make_test_config

# A 36-card deck: 465 holdings instead of 1,081, which roughly halves the cost of
# every pass. The showdown signal stays at 0.88 (floor 0.5), so the game these
# tests solve is still a real one — the deck size is a speed knob, the bucket
# ORDERING is what carries the meaning.
DECK = 36
BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}
STACK = 12


@pytest.fixture(scope="module")
def solver_parts():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(small_blind=1, big_blind=2)
    tree = BettingTree(rules, ActionModel(config), starting_stack=STACK, buckets_per_street=BUCKETS)
    compiled = compile_tree(tree, rules)

    # Strength-ordered, so showdowns carry signal — see ``contexts``.
    context = ordered_context(
        np.random.default_rng(5), {Street.PREFLOP: 169, **BUCKETS}, num_cards=DECK
    )
    return compiled, context, float((~context.blocks).sum())


@pytest.fixture(scope="module")
def trained(solver_parts):
    """One solver trained once, shared by the tests that only read it.

    128 iterations, not the 32 an earlier revision used. Buckets here now track
    hand strength (see ``contexts``), which makes this a real game rather than
    one where every showdown is a coin flip — and a real game takes longer to
    solve. The old budget cleared the same bar only because the game was
    trivial.
    """
    compiled, context, pairs = solver_parts
    solver = VectorCFR(compiled, context, cfr_plus=True)
    initial = np.ones(context.num_hands, dtype=np.float32)
    history = []
    for target in (1, 8, 32, 128):
        while solver.iteration < target:
            solver.iterate(initial)
        history.append(solver.exploitability(initial, pairs))
    return solver, history


class TestBucketCollapse:
    def test_segment_sum_equals_a_scatter_add(self):
        rng = np.random.default_rng(1)
        bucket_of_hand = rng.integers(0, 7, 50)
        values = rng.standard_normal((3, 50, 2)).astype(np.float32)

        segments = build_segments(bucket_of_hand)
        collapsed = np.add.reduceat(
            values[:, segments.hand_order, :], segments.segment_start, axis=1
        )
        fast = np.zeros((3, 7, 2), dtype=np.float32)
        fast[:, segments.segment_bucket, :] = collapsed

        naive = np.zeros((3, 7, 2), dtype=np.float32)
        for hand, bucket in enumerate(bucket_of_hand.tolist()):
            naive[:, bucket, :] += values[:, hand, :]
        assert np.allclose(fast, naive, atol=1e-4)

    def test_every_bucket_present_in_the_map_gets_a_segment(self):
        segments = build_segments(np.array([2, 0, 2, 5, 0]))
        assert segments.segment_bucket.tolist() == [0, 2, 5]
        assert segments.segment_start.tolist() == [0, 2, 4]

    def test_no_bucket_is_ever_read_without_having_been_written(self):
        """The best-response tables are filled at ``segment_bucket`` and read at
        ``buckets_for(street)``. If the read set could exceed the write set, an
        unoccupied bucket would silently return action 0 — a wrong answer, not a
        crash. It cannot, because both derive from the same array; this pins that
        so a future change to either side has to keep it true.

        Deliberately sparse: far more buckets than hands, which is the regime a
        production 600-bucket river hits on any single board.
        """
        rng = np.random.default_rng(0)
        for _ in range(200):
            num_buckets = int(rng.integers(2, 600))
            bucket_of_hand = rng.integers(0, num_buckets, int(rng.integers(1, 60)))
            written = set(build_segments(bucket_of_hand).segment_bucket.tolist())
            assert set(bucket_of_hand.tolist()) <= written


class TestZeroSum:
    def test_root_values_cancel(self, solver_parts):
        """Both players' root counterfactual values must sum to zero.

        Summed over hands with an all-ones range, each side is the same set of
        ordered holdings scored from opposite seats, so the totals are exact
        negations up to float32 rounding.

        Deliberately not sharing the trained fixture: a best-response pass
        leaves one player maximising rather than following a strategy, and the
        root values are then *correctly* not zero-sum. Only a plain iteration
        establishes the invariant.
        """
        compiled, context, _ = solver_parts
        solver = VectorCFR(compiled, context, cfr_plus=True)
        initial = np.ones(context.num_hands, dtype=np.float32)
        for _ in range(4):
            solver.iterate(initial)
            button = float(solver.value[0, 0].sum())
            other = float(solver.value[1, 0].sum())
            assert abs(button + other) <= 1e-4 * max(abs(button), 1.0)


class TestConvergence:
    def test_exploitability_falls(self, trained):
        _, history = trained
        assert history == sorted(history, reverse=True)
        assert history[-1] < history[0] / 8

    def test_best_response_is_never_worse_than_the_average_strategy(self, trained, solver_parts):
        """A best response cannot lose to the strategy it responds to.

        Exploitability is a *gain* over the average strategy's own value, so a
        negative figure would mean the responder found nothing — which for an
        unconverged strategy would indicate the maximisation is not maximising.
        """
        _, history = trained
        assert all(gain > 0 for gain in history)

    def test_untrained_play_is_more_exploitable_than_trained(self, solver_parts, trained):
        compiled, context, pairs = solver_parts
        initial = np.ones(context.num_hands, dtype=np.float32)
        untrained = VectorCFR(compiled, context, cfr_plus=True)
        untrained.iterate(initial)

        _, history = trained
        assert untrained.exploitability(initial, pairs) > history[-1]
