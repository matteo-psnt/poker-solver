"""Which way does the showdown matrix point?

Both kernels settle a showdown by multiplying the opponent's range against an
antisymmetric sign matrix, and both write ``@ S.T``. Nothing pinned that ``.T``.
Dropping it in either kernel passes the entire suite, because every check the
suite makes is invariant under ``S -> -S``:

    zero-sum            -S is still antisymmetric, so the root still cancels
    exploitability      the kernel happily solves the inverted game instead
    kernel agreement    compares the UNTRAINED strategy, which is uniform

Measured cost of that blind spot, with strength-ordered buckets and 200
iterations: the flipped kernel converges to 0.2239 in its own game -- visually
indistinguishable from the correct 0.2241 -- while scoring **4.34** in the real
one, 19x worse, having learned that losing hands win.

The invariant that catches it is polarity, and it needs no reference
implementation: order the holdings by showdown rank and the value a player
collects at a showdown must RISE with strength. A sign error inverts it exactly.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.vector import bucket_game, compile_tree
from src.engine.solver.vector.bucket_kernel import BucketVectorCFR
from src.engine.solver.vector.compiled_tree import TerminalKind
from src.engine.solver.vector.hand_context import (
    HandContext,
    blocking_matrix,
    enumerate_live_hands,
)
from src.engine.solver.vector.kernel import VectorCFR
from tests.test_helpers import make_test_config

COUNTS = {Street.FLOP: 4, Street.TURN: 5, Street.RIVER: 6}
ALL_COUNTS = {Street.PREFLOP: 169, **COUNTS}
STACK = 12
# Below this the matrix carries too little signal for polarity to be observable.
# The kernels' own fixtures bucket at random and sit near 0.02, which is why a
# sign flip is invisible to them; a real equity abstraction orders buckets.
MIN_SHOWDOWN_SIGNAL = 0.5


def _ordered_context(rng: np.random.Generator) -> HandContext:
    """A context whose bucket index tracks hand strength, as a real one does."""
    hand_cards = enumerate_live_hands(rng.choice(52, 5, replace=False))
    num_hands = hand_cards.shape[0]
    ranks = rng.permutation(num_hands)
    buckets = np.zeros((4, num_hands), dtype=np.int64)
    for street, count in ALL_COUNTS.items():
        buckets[street.value - 1] = np.minimum(ranks * count // num_hands, count - 1)
    return HandContext(hand_cards, buckets, ranks, blocking_matrix(hand_cards))


@pytest.fixture(scope="module")
def compiled():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(small_blind=1, big_blind=2)
    tree = BettingTree(rules, ActionModel(config), starting_stack=STACK, buckets_per_street=COUNTS)
    return compile_tree(tree, rules)


def _showdown_terminals(compiled, street: Street | None = None) -> np.ndarray:
    kind = np.asarray(compiled.terminal_kind)
    selected = kind == TerminalKind.SHOWDOWN
    if street is not None:
        selected &= np.asarray(compiled.terminal_street) == street.value
    return np.flatnonzero(selected)


def test_bucket_kernel_showdown_value_rises_with_bucket_strength(compiled):
    rng = np.random.default_rng(31)
    game = bucket_game.derive([_ordered_context(rng) for _ in range(6)], ALL_COUNTS)

    signal = float(np.abs(game.showdown).max())
    assert signal > MIN_SHOWDOWN_SIGNAL, (
        f"showdown matrix peaks at {signal:.3f}; buckets do not track strength, "
        "so this test could not observe a sign error"
    )

    kernel = BucketVectorCFR(compiled, game, cfr_plus=True, dtype=np.float64)
    initial = np.ones(compiled.tree.num_buckets(Street.PREFLOP), dtype=np.float64)
    kernel.iterate(initial)

    width = compiled.tree.num_buckets(Street.RIVER)
    rows = _showdown_terminals(compiled, Street.RIVER)
    profile = kernel.terminal_value[0, rows, :width].mean(axis=0)
    assert np.all(np.diff(profile) > 0), (
        f"showdown value must rise with bucket strength, got {profile}"
    )


def test_hand_space_kernel_showdown_value_rises_with_hand_strength(compiled):
    rng = np.random.default_rng(5)
    context = _ordered_context(rng)
    solver = VectorCFR(compiled, context, cfr_plus=True)
    solver.iterate(np.ones(context.num_hands, dtype=np.float32))

    rows = _showdown_terminals(compiled)
    # Hands ordered weakest to strongest, then binned so the comparison is
    # between groups rather than between two adjacent near-ties.
    by_strength = np.argsort(context.showdown_rank)
    profile = solver.terminal_value[0, rows][:, by_strength].mean(axis=0)
    binned = [float(part.mean()) for part in np.array_split(profile, 8)]
    assert all(a < b for a, b in itertools.pairwise(binned)), (
        f"showdown value must rise with hand strength, got {binned}"
    )
