"""Bucket lookup as pure array arithmetic, for a compiled traversal.

``DenseBucketer`` is a Python object holding two LRU caches, dicts of suit
labels and a memory-mapped matrix. A ``nopython`` kernel can index the arrays
but cannot touch the rest, and calling back out through ``objmode`` for ~48
lookups an iteration costs more than compiling the walk saves. So the lookup
itself has to come inside.

Nothing here re-derives the abstraction. The artifact's three arrays are passed
straight in — sorted canonical board ids, the bucket matrix, and the static
hand-id-to-column map — and this reproduces the addressing that reaches them:

    canonicalise the board  ->  board id  ->  binary search  ->  row
    canonicalise the hand    ->  hand id   ->  column map     ->  column

THE TIE-BREAK IS THE WHOLE RISK. Two suits with identical rank lists are
interchangeable on the board but not for a hand read against the resulting
labels, so they are ordered by suit CHARACTER. eval7 encodes suits c=0, d=1,
h=2, s=3, which is that same alphabetical order, so the integer suit index is
the tie-break directly. Getting this wrong re-buckets hands silently — see
``suit_isomorphism._suit_labels``, where the same rule survived every flop and
every turn while wrong.

Ranks arrive in eval7's encoding (0=2 .. 12=A) and are converted to the
abstraction's (0=A .. 12=2) on the way in, so callers pass card attributes
straight through.
"""

from __future__ import annotations

import numpy as np
from numba import jit

# Wide enough for a rank index (0..12) or the sentinel that marks "this suit has
# no more cards", which must sort above every rank — see `_suit_labels`.
_SLOT_BITS = 7
_SENTINEL = 99


@jit(nopython=True, cache=True)
def suit_labels(ranks, suits, width, labels):
    """Canonical label per suit, or -1 for a suit absent from the board.

    Fills ``labels`` (length 4, indexed by eval7 suit) and returns how many
    suits were present, which is the next free label for a hand that brings a
    suit the board never showed.

    Each suit's sort key packs its ranks ASCENDING (so the highest card first,
    since 0 is the ace here), pads to the board width with a sentinel above
    every rank, and appends the suit index. That is exactly the
    "shorter-is-greater, then by suit character" order the Python
    implementation sorts tuples by.
    """
    keys = np.empty(4, dtype=np.int64)
    present = np.zeros(4, dtype=np.int64)
    for suit in range(4):
        labels[suit] = -1

    for suit in range(4):
        mask = 0
        for i in range(ranks.shape[0]):
            if suits[i] == suit:
                mask |= 1 << (12 - ranks[i])
        if mask == 0:
            keys[suit] = 0
            continue
        present[suit] = 1
        key = 0
        taken = 0
        for rank in range(13):
            if mask & (1 << rank):
                key = (key << _SLOT_BITS) | rank
                taken += 1
        for _ in range(taken, width):
            key = (key << _SLOT_BITS) | _SENTINEL
        keys[suit] = (key << 2) | suit

    # At most four entries, so a selection sort is the cheapest thing that is
    # obviously right.
    count = 0
    for _ in range(4):
        best = -1
        for suit in range(4):
            if present[suit] == 1 and labels[suit] < 0 and (best < 0 or keys[suit] < keys[best]):
                best = suit
        if best < 0:
            break
        labels[best] = count
        count += 1
    return count


@jit(nopython=True, cache=True)
def board_id(ranks, suits, labels):
    """The canonical board id: cards as ``rank * 4 + label``, folded base-52."""
    width = ranks.shape[0]
    codes = np.empty(width, dtype=np.int64)
    for i in range(width):
        codes[i] = (12 - ranks[i]) * 4 + labels[suits[i]]
    codes.sort()

    packed = 0
    for i in range(width):
        packed = packed * 52 + codes[i]
    return packed


@jit(nopython=True, cache=True)
def hand_id(hole_ranks, hole_suits, labels, next_label):
    """``hand_id_of`` over a two-card array pair."""
    return hand_id_of(
        hole_ranks[0], hole_suits[0], hole_ranks[1], hole_suits[1], labels, next_label
    )


@jit(nopython=True, cache=True)
def hand_id_of(rank_first, suit_first, rank_second, suit_second, labels, next_label):
    """The canonical hand id against a board's labels.

    A hole-card suit the board never showed takes the next free label, and
    those are handed out in RANK order (high card first) so that the two hole
    cards cannot canonicalise differently depending on which was passed first.
    ``labels`` is not modified.
    """
    rank_a = 12 - rank_first
    rank_b = 12 - rank_second
    suit_a = suit_first
    suit_b = suit_second
    if rank_b < rank_a:  # lower index is the higher card; take it first
        rank_a, rank_b = rank_b, rank_a
        suit_a, suit_b = suit_b, suit_a

    label_a = labels[suit_a]
    if label_a < 0:
        label_a = next_label
        next_label += 1
    if suit_b == suit_a:
        label_b = label_a
    else:
        label_b = labels[suit_b]
        if label_b < 0:
            label_b = next_label

    first = rank_a * 4 + label_a
    second = rank_b * 4 + label_b
    if second < first:
        first, second = second, first
    return first * 52 + second


@jit(nopython=True, cache=True)
def postflop_bucket(
    hole_ranks, hole_suits, board_ranks, board_suits, board_ids, matrix, hand_to_col, sentinel
):
    """Bucket for a (hand, board) pair, or -1 if the combination is not legal.

    Returns -1 rather than raising where ``DenseBucketer`` raises ``KeyError``:
    a kernel has no exceptions, and the caller is better placed to say which of
    "board absent from the abstraction" or "hand impossible on this board" it
    is looking at.
    """
    labels = np.empty(4, dtype=np.int64)
    next_label = suit_labels(board_ranks, board_suits, board_ranks.shape[0], labels)

    target = board_id(board_ranks, board_suits, labels)
    row = np.searchsorted(board_ids, target)
    if row >= board_ids.shape[0] or board_ids[row] != target:
        return -1

    column = hand_to_col[hand_id(hole_ranks, hole_suits, labels, next_label)]
    if column < 0:
        return -1
    bucket = matrix[row, column]
    if bucket == sentinel:
        return -1
    return bucket
