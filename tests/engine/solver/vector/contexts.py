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

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Street
from src.engine.solver.vector import bucket_game
from src.engine.solver.vector.hand_context import (
    HandContext,
    blocking_matrix,
    enumerate_live_hands,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Below this the showdown matrix carries too little signal for a test to observe
# polarity or anything else that scales with it. Ordered buckets clear it
# comfortably; random buckets sit near 0.02.
MIN_SHOWDOWN_SIGNAL = 0.5


def ordered_context(
    rng: np.random.Generator,
    counts: dict[Street, int],
    *,
    num_cards: int = 52,
    board: Sequence[int] | np.ndarray | None = None,
) -> HandContext:
    """A context whose bucket index tracks hand strength, as a real one does.

    ``board`` pins the five cards instead of drawing them, which is what lets a
    test build runouts that deliberately share a prefix — the case that decides
    whether an unconstrained best response is reading cards that are still face
    down. See ``test_mixture``.
    """
    cards = rng.choice(num_cards, 5, replace=False) if board is None else board
    hand_cards = enumerate_live_hands(cards, num_cards)
    num_hands = hand_cards.shape[0]
    ranks = rng.permutation(num_hands)

    buckets = np.zeros((4, num_hands), dtype=np.int64)
    for street, count in counts.items():
        buckets[street.value - 1] = np.minimum(ranks * count // num_hands, count - 1)
    return HandContext(hand_cards, buckets, ranks, blocking_matrix(hand_cards))


def prefix_consistent_contexts(
    boards: Sequence[Sequence[int] | np.ndarray],
    counts: dict[Street, int],
    *,
    num_cards: int = 52,
) -> list[HandContext]:
    """Contexts obeying the rule production buckets obey: a street sees a prefix.

    ``build_hand_context`` asks the abstraction for a street's bucket with
    ``cards[:seen]`` — three cards on the flop, four on the turn, five on the
    river, none preflop. So a bucket at street ``s`` is a function of the hand
    and the cards *face up* at ``s``, and two runouts sharing that prefix hand
    the same holding the same bucket. A player genuinely cannot tell them apart.

    :func:`ordered_context` draws a fresh permutation per board and so breaks
    that rule: it gives one hand unrelated turn buckets on two boards with the
    same turn. Every test that treats boards independently is unaffected — but a
    test about what a responder may DISTINGUISH is measuring nothing on such a
    fixture, because the abstraction itself leaks the runout there.

    Buckets are keyed on the global two-card id rather than a board's own hand
    index, since boards remove different cards and index their live hands
    differently.
    """
    visible = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
    span = num_cards * num_cards
    contexts = []
    for board in boards:
        hand_cards = enumerate_live_hands(board, num_cards)
        global_id = hand_cards[:, 0] * num_cards + hand_cards[:, 1]
        buckets = np.zeros((4, hand_cards.shape[0]), dtype=np.int64)
        strength = None
        for street, count in counts.items():
            prefix = sorted(int(card) for card in board[: visible[street]])
            # A stable integer seed, NOT hash() of anything: Python randomises
            # string hashing per process and a fixture that moved between runs
            # would be worse than one that is merely unrealistic.
            seed = street.value
            for card in prefix:
                seed = seed * 53 + card + 1
            rank = np.random.default_rng(seed).permutation(span)[global_id]
            buckets[street.value - 1] = np.minimum(rank * count // span, count - 1)
            if street is Street.RIVER:
                strength = rank
        if strength is None:
            raise ValueError("counts must include Street.RIVER: showdown ranks come from it.")
        # Showdown is settled on the full board, which the river prefix already
        # is, so ranking by the river ordering keeps buckets strength-ordered.
        contexts.append(HandContext(hand_cards, buckets, strength, blocking_matrix(hand_cards)))
    return contexts


def sparse_context(
    rng: np.random.Generator,
    counts: dict[Street, int],
    *,
    occupancy: float = 0.2,
    num_cards: int = 52,
    board: Sequence[int] | np.ndarray | None = None,
) -> HandContext:
    """A context whose board occupies only PART of each street's bucket space.

    :func:`ordered_context` spreads a board's hands over every bucket, which the
    production abstraction does not: measured against the 100/300/600 artifact,
    one runout occupies 48 of 100 flop, 58 of 300 turn and 81 of 600 river
    buckets. A row is therefore written by a MINORITY of boards, and the
    per-visit DCFR discount reaches it at a rate that varies by street -- the
    regime the fixtures above cannot express and the one production trains in.

    Each board draws its own sorted subset of the bucket space and ranks its
    hands into it, so buckets stay strength-ordered (the showdown signal
    survives) while different boards share an overlapping, non-identical set.

    Inherits :func:`ordered_context`'s per-board draw, so it BREAKS the prefix
    rule :func:`prefix_consistent_contexts` exists to keep: two boards sharing a
    flop give one hand unrelated flop buckets. Fine for a sampler that draws one
    full board per iteration, where every board is alone in its partition —
    never for a mixture or CFR-BR test, which would read the runout off it.
    """
    cards = rng.choice(num_cards, 5, replace=False) if board is None else board
    hand_cards = enumerate_live_hands(cards, num_cards)
    num_hands = hand_cards.shape[0]
    ranks = rng.permutation(num_hands)

    buckets = np.zeros((4, num_hands), dtype=np.int64)
    for street, count in counts.items():
        width = max(1, min(count, round(count * occupancy)))
        occupied = np.sort(rng.choice(count, width, replace=False))
        buckets[street.value - 1] = occupied[np.minimum(ranks * width // num_hands, width - 1)]
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


__all__: Sequence[str] = (
    "MIN_SHOWDOWN_SIGNAL",
    "ordered_context",
    "prefix_consistent_contexts",
    "showdown_signal",
    "sparse_context",
)
