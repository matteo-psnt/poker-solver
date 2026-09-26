"""Showdown values in O(H) per terminal, by walking hands in rank order.

``VectorCFR`` could value every showdown as one ``(T, H) @ (H, H)``
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


# Rows valued together: each row keeps its own running totals, laid out so the
# inner loop runs across rows and vectorises. Per row the arithmetic, and so
# every output byte, is exactly the single-row walk's.
ROW_BLOCK = 16


@jit(nopython=True, cache=True)
def _walk(reach, order, group_start, cards, out):
    """``out[t, h] = beaten mass - losing mass`` for every terminal row ``t``.

    ``order`` lists hands weakest first; ``group_start`` marks where each run of
    equal rank begins, closed by ``num_hands``. A tie group is valued against
    the state BEFORE it is folded in, so no hand sees its own tie partners; the
    second direction walks the same groups strongest first. Accumulates in
    float64: the output is a difference of two sums of similar size.
    """
    num_rows, num_hands = reach.shape
    groups = group_start.shape[0] - 1
    card = np.zeros((NUM_CARDS, ROW_BLOCK), dtype=np.float64)
    total = np.zeros(ROW_BLOCK, dtype=np.float64)
    local = np.empty((num_hands, ROW_BLOCK), dtype=np.float64)
    result = np.empty((num_hands, ROW_BLOCK), dtype=np.float64)
    for first in range(0, num_rows, ROW_BLOCK):
        width = min(ROW_BLOCK, num_rows - first)
        for h in range(num_hands):
            for r in range(width):
                local[h, r] = reach[first + r, h]
        for direction in range(2):
            card[:, :] = 0.0
            total[:] = 0.0
            for step in range(groups):
                g = step if direction == 0 else groups - 1 - step
                lo = group_start[g]
                hi = group_start[g + 1]
                for k in range(lo, hi):
                    h = order[k]
                    c0 = cards[h, 0]
                    c1 = cards[h, 1]
                    for r in range(width):
                        mass = total[r] - card[c0, r] - card[c1, r]
                        if direction == 0:
                            result[h, r] = mass
                        else:
                            result[h, r] -= mass
                # Folding the group in walks it in the direction's own order.
                for j in range(hi - lo):
                    k = lo + j if direction == 0 else hi - 1 - j
                    h = order[k]
                    c0 = cards[h, 0]
                    c1 = cards[h, 1]
                    for r in range(width):
                        v = local[h, r]
                        total[r] += v
                        card[c0, r] += v
                        card[c1, r] += v
        for r in range(width):
            for h in range(num_hands):
                out[first + r, h] = result[h, r]


class RankWalk:
    """One board's rank order, so the walk costs a sort once per board."""

    __slots__ = ("cards", "group_start", "order")

    def __init__(self, showdown_rank: np.ndarray, hand_cards: np.ndarray):
        self.order = np.argsort(showdown_rank, kind="stable").astype(np.int64)
        ranks = np.asarray(showdown_rank, dtype=np.int64)[self.order]
        boundaries = np.flatnonzero(np.diff(ranks)) + 1
        self.group_start = np.concatenate([[0], boundaries, [ranks.shape[0]]]).astype(np.int64)
        self.cards = np.ascontiguousarray(hand_cards, dtype=np.int64)

    def values(self, reach: np.ndarray) -> np.ndarray:
        """``(T, H)``: for each row, the opponent mass each hand beats minus loses to."""
        out = np.zeros(reach.shape, dtype=np.float64)
        _walk(
            np.ascontiguousarray(reach, dtype=np.float32),
            self.order,
            self.group_start,
            self.cards,
            out,
        )
        return out


__all__ = ("RankWalk",)
