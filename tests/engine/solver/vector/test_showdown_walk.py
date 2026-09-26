"""The row-blocked walk must give the one-row walk's BYTES.

It reorders only WHICH rows run together; each row's float64 sums must stay in
the single-row walk's order, including the within-group order of the second,
strongest-first direction. The oracle is that single-row walk, kept here.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import jit

from src.engine.solver.vector.showdown import ROW_BLOCK, RankWalk


@jit(nopython=True)
def _one_row_walk(reach, order, rank_sorted, cards, out):
    num_rows, num_hands = reach.shape
    card = np.zeros(52, dtype=np.float64)
    for row in range(num_rows):
        o, rs = order, rank_sorted
        for direction in range(2):
            total = 0.0
            card[:] = 0.0
            start = 0
            while start < num_hands:
                stop = start
                while stop < num_hands and rs[stop] == rs[start]:
                    stop += 1
                for k in range(start, stop):
                    h = o[k]
                    mass = total - card[cards[h, 0]] - card[cards[h, 1]]
                    if direction == 0:
                        out[row, h] = mass
                    else:
                        out[row, h] -= mass
                for k in range(start, stop):
                    h = o[k]
                    v = reach[row, h]
                    total += v
                    card[cards[h, 0]] += v
                    card[cards[h, 1]] += v
                start = stop
            o, rs = o[::-1], rs[::-1]


@pytest.mark.parametrize(("rows", "ranks"), [(1, 50), (ROW_BLOCK + 1, 300), (40, 1081), (7, 3)])
def test_blocked_walk_matches_the_one_row_walk(rows, ranks):
    rng = np.random.default_rng(rows * 1000 + ranks)
    cards = np.array([(a, b) for a in range(47) for b in range(a + 1, 47)], dtype=np.int64)
    rank = rng.integers(0, ranks, cards.shape[0])
    reach = rng.random((rows, cards.shape[0]), dtype=np.float32)
    reach[:, ::9] = 0.0

    order = np.argsort(rank, kind="stable").astype(np.int64)
    expected = np.zeros(reach.shape, dtype=np.float64)
    _one_row_walk(reach, order, rank.astype(np.int64)[order], cards, expected)

    assert RankWalk(rank, cards).values(reach).tobytes() == expected.tobytes()
