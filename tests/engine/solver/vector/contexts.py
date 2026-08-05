"""Board contexts for the vector-CFR tests, and why they are ordered.

A real equity abstraction numbers its buckets by strength: bucket 0 is the
weakest holdings, the last bucket the strongest. Bucketing at *random* instead
produces a game that technically runs and says almost nothing — every bucket
holds a uniform mix of strengths, so the showdown matrix averages to near zero
and no showdown ever meaningfully favours anyone.

That is not a hypothetical weakness. Measured on these fixtures: random buckets
peak at ``max|S| ~ 0.02`` against ``~0.92`` for ordered ones. A suite built on
random buckets is exercising the kernels at roughly **2% of production
amplitude**, and a sign error in the showdown matrix moved the trained strategy
by 2e-06 — invisible — while costing 19x in the real game. See
``test_showdown_polarity``, which found exactly that.

So ordered contexts are the default here, and
:data:`MIN_SHOWDOWN_SIGNAL` is the guard: any test whose power depends on
showdowns actually mattering should assert the signal is present *before*
asserting anything about behaviour, so it cannot quietly lose its own power.

This module is a helper, not a test module, so pytest does not collect it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.game.state import Street
from src.engine.solver.vector import bucket_game
from src.engine.solver.vector.hand_context import (
    HandContext,
    blocking_matrix,
    enumerate_live_hands,
)

# Below this the showdown matrix carries too little signal for a test to observe
# polarity or anything else that scales with it. Ordered buckets clear it
# comfortably; random buckets sit near 0.02.
MIN_SHOWDOWN_SIGNAL = 0.5


def ordered_context(
    rng: np.random.Generator, counts: dict[Street, int], *, num_cards: int = 52
) -> HandContext:
    """A context whose bucket index tracks hand strength, as a real one does."""
    hand_cards = enumerate_live_hands(rng.choice(num_cards, 5, replace=False), num_cards)
    num_hands = hand_cards.shape[0]
    ranks = rng.permutation(num_hands)

    buckets = np.zeros((4, num_hands), dtype=np.int64)
    for street, count in counts.items():
        buckets[street.value - 1] = np.minimum(ranks * count // num_hands, count - 1)
    return HandContext(hand_cards, buckets, ranks, blocking_matrix(hand_cards))


def showdown_signal(contexts: Sequence[HandContext], counts: dict[Street, int]) -> float:
    """Peak magnitude of the **bucket-space** showdown matrix these contexts imply.

    Measured after collapsing to buckets, which is the only place the number
    means anything. In hand space the matrix is ``sign(rank_i - rank_j)``, so its
    peak is 1.0 for *any* bucketing including a random one — measuring there
    would report full signal on a game that has none.

    It is the bucket average that collapses: with random buckets each bucket
    holds a uniform mix of strengths, every bucket pair averages to about
    nothing, and no showdown favours anyone.
    """
    return float(np.abs(bucket_game.derive(contexts, counts).showdown).max())


__all__: Sequence[str] = ("MIN_SHOWDOWN_SIGNAL", "ordered_context", "showdown_signal")
