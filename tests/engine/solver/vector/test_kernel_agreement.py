"""Do the two kernels describe the same game?

They are independent implementations over the same betting tree: one carries a
range over private hands with exact card removal and exact showdown ranks, the
other carries a range over buckets with both of those averaged into matrices.
Nothing forces them to agree, and if they disagree then every cross-evaluation
in the results writeup is comparing two different games rather than measuring
the cost of an approximation.

The uniform strategy is where they *must* agree closely: no strategy has been
learned, so the only thing being compared is the game's own structure — the
tree, the payoffs, the blocking, the showdown. Any residual gap there is the
averaging itself, which is the quantity under study.

They are deliberately not asserted equal. Averaging card removal and showdown
ranks over buckets is lossy by construction, so the assertion is that the two
agree to within a fraction of the difference training makes.
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
from src.engine.solver.vector.mixture import BoardMixtureCFR
from tests.engine.solver.vector.contexts import (
    MIN_SHOWDOWN_SIGNAL,
    ordered_context,
    showdown_signal,
)
from tests.test_helpers import make_test_config

COUNTS = {Street.FLOP: 4, Street.TURN: 5, Street.RIVER: 6}
ALL_COUNTS = {Street.PREFLOP: 169, **COUNTS}
STACK = 12


@pytest.fixture(scope="module")
def games():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(small_blind=1, big_blind=2)
    tree = BettingTree(rules, ActionModel(config), starting_stack=STACK, buckets_per_street=COUNTS)
    compiled = compile_tree(tree, rules)

    rng = np.random.default_rng(77)
    contexts = [ordered_context(rng, ALL_COUNTS) for _ in range(8)]
    pairs = float(np.mean([(~c.blocks).sum() for c in contexts]))
    game = bucket_game.derive(contexts, ALL_COUNTS)
    mass = np.mean(
        [np.bincount(c.buckets_for(Street.PREFLOP), minlength=169) for c in contexts], axis=0
    )
    return compiled, contexts, pairs, game, mass


def test_both_kernels_agree_on_the_untrained_strategy(games):
    """Uniform play must be about as exploitable in one game as the other."""
    compiled, contexts, pairs, game, mass = games

    hand_space = BoardMixtureCFR(compiled, contexts, cfr_plus=True)
    hand_value = hand_space.exploitability(np.ones(contexts[0].num_hands, dtype=np.float32), pairs)
    board_free = BucketVectorCFR(compiled, game, cfr_plus=True)
    bucket_value = board_free.exploitability(mass)

    assert hand_value > 0
    # Within 25%: the two agree on the shape of the game, and the residual is
    # the averaging. Training moves this quantity by more than an order of
    # magnitude, so a disagreement large enough to matter would blow this open.
    assert abs(bucket_value - hand_value) / hand_value < 0.25


def test_the_two_kernels_share_a_storage_layout(games):
    """A strategy trained in one is readable by the other with no translation.

    This is what makes the cross-evaluation possible, and it is a property of
    the layout rather than of either kernel, so it is pinned here rather than in
    either kernel's own tests.
    """
    compiled, contexts, _, game, _ = games
    hand_space = BoardMixtureCFR(compiled, contexts, cfr_plus=True)
    board_free = BucketVectorCFR(compiled, game, cfr_plus=True)

    assert hand_space.strategy_sum.shape == board_free.strategy_sum.shape
    assert hand_space.regrets.shape == board_free.regrets.shape
    assert board_free.strategy_sum.shape == (compiled.tree.num_slots,)


def test_the_shared_fixtures_carry_showdown_signal(games):
    """The suite must not quietly lose the ability to see what it tests.

    Every kernel test builds its boards with ``ordered_context``. If that ever
    stopped ordering buckets by strength, the showdown matrix would collapse
    toward zero, every game in the suite would become a sequence of coin flips,
    and the tests would go on passing while checking almost nothing — which is
    exactly how a showdown sign error survived an entire green suite once.

    So the premise is asserted rather than assumed, in one place, for all of it.
    """
    _, contexts, _, _, _ = games
    signal = showdown_signal(contexts, ALL_COUNTS)
    assert signal > MIN_SHOWDOWN_SIGNAL, (
        f"ordered fixtures peak at {signal:.3f}; buckets no longer track "
        "strength, so the suite cannot observe showdown behaviour"
    )
