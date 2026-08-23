"""Showdown values in O(H) per terminal, by walking hands in rank order.

``VectorCFR.evaluate_terminals`` values every showdown as one ``(T, H) @ (H, H)``
product against the board's win/lose sign matrix. That is 358 GFLOP per
iteration on the production tree -- invisible behind Apple's AMX, several
seconds on an EPYC core, and it is the arithmetic a worker on the pool would
spend most of its time in. The same numbers come from two prefix sums
(Johanson et al. 2012, "public chance sampling"): walk hands from weakest to
strongest carrying the opponent mass seen so far in total and per card, and a
hand's "mass I beat" is the total minus the mass holding either of its cards.
Ties contribute nothing, exactly as the sign matrix's zero diagonal blocks.
"""

from __future__ import annotations

import numpy as np
from numba import jit

# Width of the per-card running totals: a standard deck.
NUM_CARDS = 52


@jit(nopython=True, cache=True)
def _walk(reach, order, rank_sorted, cards, out):
    """``out[t, h] = beaten mass - losing mass`` for every terminal row ``t``.

    ``order`` lists hands weakest first and ``rank_sorted`` their ranks in that
    order; equal ranks form a tie group, valued against the state BEFORE the
    group is folded in so no hand sees its own tie partners. Accumulates in
    float64: the output is a difference of two sums of similar size.
    """
    num_rows, num_hands = reach.shape
    card = np.zeros(NUM_CARDS, dtype=np.float64)
    for row in range(num_rows):
        for direction in range(2):
            total = 0.0
            for c in range(NUM_CARDS):
                card[c] = 0.0
            start = 0
            while start < num_hands:
                stop = start
                while stop < num_hands and rank_sorted[stop] == rank_sorted[start]:
                    stop += 1
                for k in range(start, stop):
                    h = order[k]
                    mass = total - card[cards[h, 0]] - card[cards[h, 1]]
                    if direction == 0:
                        out[row, h] = mass
                    else:
                        out[row, h] -= mass
                for k in range(start, stop):
                    h = order[k]
                    v = reach[row, h]
                    total += v
                    card[cards[h, 0]] += v
                    card[cards[h, 1]] += v
                start = stop
            # Second direction walks strongest first: reverse the same arrays.
            order = order[::-1]
            rank_sorted = rank_sorted[::-1]


class RankWalk:
    """One board's rank order, so the walk costs a sort once per board."""

    __slots__ = ("cards", "order", "rank_sorted")

    def __init__(self, showdown_rank: np.ndarray, hand_cards: np.ndarray):
        self.order = np.argsort(showdown_rank, kind="stable").astype(np.int64)
        self.rank_sorted = np.asarray(showdown_rank, dtype=np.int64)[self.order]
        self.cards = np.ascontiguousarray(hand_cards, dtype=np.int64)

    def values(self, reach: np.ndarray) -> np.ndarray:
        """``(T, H)``: for each row, the opponent mass each hand beats minus loses to."""
        out = np.zeros(reach.shape, dtype=np.float64)
        _walk(
            np.ascontiguousarray(reach, dtype=np.float32),
            self.order,
            self.rank_sorted,
            self.cards,
            out,
        )
        return out


__all__ = ("RankWalk",)
