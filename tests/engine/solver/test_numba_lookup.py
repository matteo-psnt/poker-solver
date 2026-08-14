"""Does the compiled lookup address the abstraction exactly as DenseBucketer does?

`numba_lookup` exists because a `nopython` kernel cannot call a Python object
with LRU caches and dicts, and `objmode` for ~48 lookups an iteration costs
more than compiling the walk saves. It reproduces the ADDRESSING, not the
abstraction: the artifact's own arrays are passed in.

The bar is the bucket that comes out, checked against `DenseBucketer` on the
same artifact. Disagreement would not crash — it would train the right strategy
against the wrong hand class, forever, silently — so the flop sweep covers
every canonical board rather than a sample, and the boards are SHUFFLED,
because the suit-character tie-break is invisible in deck order (that is how
the same rule went in wrong in `suit_isomorphism`).

The artifact here is synthetic: bucket VALUES are arbitrary, since what is
under test is which cell gets read, and a value that is a function of the cell
makes a misaddressed read visible.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from src.core.game.state import Card, Street
from src.engine.solver.numba_lookup import postflop_bucket, suit_labels
from src.pipeline.abstraction.postflop.bucketer import (
    N_HAND_COLUMNS,
    DenseBucketer,
    build_hand_column_index,
)
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    canonical_board_id,
)

DECK = [Card.new(rank + suit) for rank in "23456789TJQKA" for suit in "cdhs"]

# The flop sweep walks all 22,100 canonical boards and pays numba's compile on
# a cold cache, which is ~4s alone and past the 5s default under twelve xdist
# workers. Tight enough to still catch a hang, loose enough not to fail on load.
pytestmark = pytest.mark.timeout(60)
NUM_BUCKETS = 250


def _arrays(cards):
    return (
        np.array([c.rank_eval7() for c in cards], dtype=np.int64),
        np.array([c.suit_eval7() for c in cards], dtype=np.int64),
    )


def _artifact(boards, street):
    """A DenseBucketer covering exactly ``boards``, with cell-derived values."""
    ids = sorted({canonical_board_id(board)[0] for board in boards})
    board_ids = np.array(ids, dtype=np.int64)
    rng = np.random.default_rng(11)
    matrix = rng.integers(0, NUM_BUCKETS, size=(board_ids.size, N_HAND_COLUMNS), dtype=np.uint16)
    bucketer = DenseBucketer(
        {street: NUM_BUCKETS}, {street: board_ids}, {street: matrix}, build_hand_column_index()
    )
    return bucketer, board_ids, matrix


@pytest.fixture(scope="module")
def flop_artifact():
    shuffler = random.Random(5)
    boards = []
    for combo in itertools.combinations(DECK, 3):
        board = list(combo)
        shuffler.shuffle(board)
        boards.append(tuple(board))
    return (*_artifact(boards, Street.FLOP), boards)


class TestSuitLabelsMatchThePythonLabelling:
    def test_every_canonical_flop_shape(self, flop_artifact):
        *_, boards = flop_artifact
        labels = np.empty(4, dtype=np.int64)
        for board in boards:
            ranks, suits = _arrays(board)
            count = suit_labels(ranks, suits, len(board), labels)
            _, want = canonical_board_id(board)
            got = {"cdhs"[s]: int(labels[s]) for s in range(4) if labels[s] >= 0}
            assert got == want, board
            assert count == len(want)

    def test_a_river_sample(self):
        rng = random.Random(6)
        labels = np.empty(4, dtype=np.int64)
        for _ in range(20_000):
            board = tuple(rng.sample(DECK, 5))
            ranks, suits = _arrays(board)
            suit_labels(ranks, suits, 5, labels)
            _, want = canonical_board_id(board)
            assert {"cdhs"[s]: int(labels[s]) for s in range(4) if labels[s] >= 0} == want, board


class TestBucketsMatchDenseBucketer:
    def test_every_canonical_flop_with_a_random_hand(self, flop_artifact):
        """One hand per flop, over all 22,100 — the addressing, end to end."""
        bucketer, board_ids, matrix, boards = flop_artifact
        rng = random.Random(7)
        checked = 0
        for board in boards:
            available = [c for c in DECK if c not in board]
            hole = tuple(rng.sample(available, 2))
            want = bucketer.get_bucket(hole, board, Street.FLOP)
            hole_ranks, hole_suits = _arrays(hole)
            board_ranks, board_suits = _arrays(board)
            got = postflop_bucket(
                hole_ranks,
                hole_suits,
                board_ranks,
                board_suits,
                board_ids,
                matrix,
                build_hand_column_index(),
                np.iinfo(np.uint16).max,
            )
            assert got == want, (board, hole)
            checked += 1
        assert checked == 22_100

    def test_many_hands_on_one_board(self, flop_artifact):
        """Every legal hand on a fixed board — the column map, exhaustively."""
        bucketer, board_ids, matrix, _ = flop_artifact
        board = (Card.new("Ah"), Card.new("Kh"), Card.new("2c"))
        hand_to_col = build_hand_column_index()
        board_ranks, board_suits = _arrays(board)
        available = [c for c in DECK if c not in board]
        checked = 0
        for hole in itertools.combinations(available, 2):
            want = bucketer.get_bucket(hole, board, Street.FLOP)
            hole_ranks, hole_suits = _arrays(hole)
            got = postflop_bucket(
                hole_ranks,
                hole_suits,
                board_ranks,
                board_suits,
                board_ids,
                matrix,
                hand_to_col,
                np.iinfo(np.uint16).max,
            )
            assert got == want, hole
            checked += 1
        assert checked == len(available) * (len(available) - 1) // 2

    def test_hole_card_order_does_not_change_the_bucket(self, flop_artifact):
        _, board_ids, matrix, boards = flop_artifact
        hand_to_col = build_hand_column_index()
        rng = random.Random(8)
        for board in boards[:2_000]:
            available = [c for c in DECK if c not in board]
            a, b = rng.sample(available, 2)
            board_ranks, board_suits = _arrays(board)
            first = postflop_bucket(
                *_arrays((a, b)),
                board_ranks,
                board_suits,
                board_ids,
                matrix,
                hand_to_col,
                np.iinfo(np.uint16).max,
            )
            second = postflop_bucket(
                *_arrays((b, a)),
                board_ranks,
                board_suits,
                board_ids,
                matrix,
                hand_to_col,
                np.iinfo(np.uint16).max,
            )
            assert first == second

    def test_a_board_outside_the_artifact_reports_absence(self):
        """DenseBucketer raises; a kernel has no exceptions, so it returns -1.

        The absent board has to be the SAME WIDTH as the artifact's. Board ids
        are a base-52 fold of the cards, so they are only unique WITHIN a
        street — a four-card id lands in the same integer range as a
        three-card one and can collide. Production is safe by construction
        (`_board_row` selects the street's own array first), but it means
        "a turn board cannot be in a flop artifact" is false.
        """
        covered = [
            (Card.new("2c"), Card.new("7d"), Card.new("9h")),
            (Card.new("As"), Card.new("Ks"), Card.new("Qs")),
        ]
        bucketer, board_ids, matrix = _artifact(covered, Street.FLOP)
        hand_to_col = build_hand_column_index()
        hole = (Card.new("4d"), Card.new("5d"))

        present = postflop_bucket(
            *_arrays(hole),
            *_arrays(covered[0]),
            board_ids,
            matrix,
            hand_to_col,
            np.iinfo(np.uint16).max,
        )
        assert present == bucketer.get_bucket(hole, covered[0], Street.FLOP)

        absent = (Card.new("3c"), Card.new("8d"), Card.new("Th"))
        with pytest.raises(KeyError):
            bucketer.get_bucket(hole, absent, Street.FLOP)
        assert (
            postflop_bucket(
                *_arrays(hole),
                *_arrays(absent),
                board_ids,
                matrix,
                hand_to_col,
                np.iinfo(np.uint16).max,
            )
            == -1
        )
