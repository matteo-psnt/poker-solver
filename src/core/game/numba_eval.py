"""A seven-card hand evaluator numba can compile.

WHY THIS EXISTS. ``eval7`` is a Cython extension, so a ``nopython`` kernel
cannot call it — and a traversal that has to leave the kernel at every showdown
pays more in boundary crossings than compiling saved. This is the gate on
compiling the walk at all: if a native evaluator is not both exact and fast,
that project stops here.

WHAT "EXACT" MEANS, AND WHY IT IS WEAKER THAN IT SOUNDS. Nothing reads a hand's
absolute strength; ``get_payoff`` only ever asks which of two hands is better,
via ``compare_hands``. So this does not have to reproduce eval7's numbers — it
has to induce the SAME ORDER, ties included. That is what
``test_numba_eval.py`` checks: eval7 rank and this rank must agree on every
comparison, which is equivalent to the map between them being strictly
monotonic.

THE ENCODING. One integer per hand, comparable with ``<``:

    category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5

with ranks 0..12 (2..A) in order of significance, so the arithmetic compare is
the poker compare. Nine categories, high card 0 to straight flush 8.

WHY A FLUSH SHORT-CIRCUITS. With seven cards a flush cannot coexist with quads
or a full house: quads plus a five-flush needs at least eight cards (a rank
appears once per suit, so at most one quad card lies in the flush suit), and a
full house needs three cards outside a five-card flush suit when only two are
left. So once five cards share a suit, the answer is a flush or a straight
flush and nothing above them is reachable.
"""

from __future__ import annotations

import numpy as np
from numba import jit

# Category codes, ordered as poker orders them.
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

# A 13-bit rank mask with the five bits of each straight set, best first. The
# wheel is last and is the only one that is not five consecutive bits: the ace
# plays low, so it is A-5-4-3-2 = bit 12 plus bits 3..0.
_STRAIGHTS = np.array(
    [0b1111100000000 >> shift for shift in range(9)] + [0b1000000001111],
    dtype=np.int64,
)
# High card of each straight above, in the same order: A down to 6, then the
# wheel, whose high card is the five.
_STRAIGHT_HIGH = np.array([12 - shift for shift in range(9)] + [3], dtype=np.int64)


@jit(nopython=True, cache=True)
def _straight_high(mask, straights, highs):
    """Rank of the best straight in ``mask``, or -1 if there is none."""
    for i in range(straights.shape[0]):
        if mask & straights[i] == straights[i]:
            return highs[i]
    return -1


@jit(nopython=True, cache=True)
def _top_ranks(mask, count, skip):
    """The ``count`` highest ranks set in ``mask``, packed most-significant first.

    ``skip`` is a mask of ranks already spent on the category itself (the pair,
    the trips), which must not reappear as kickers.
    """
    packed = 0
    taken = 0
    rank = 12
    while rank >= 0 and taken < count:
        bit = 1 << rank
        if mask & bit and not skip & bit:
            packed = (packed << 4) | rank
            taken += 1
        rank -= 1
    # Left-align so hands with fewer kickers still compare against the same
    # field width (a flush packs five, quads packs one).
    return packed << (4 * (count - taken)) if taken < count else packed


@jit(nopython=True, cache=True)
def _top_rank(mask):
    """The highest rank set in ``mask``, or -1."""
    rank = 12
    while rank >= 0:
        if mask >> rank & 1:
            return rank
        rank -= 1
    return -1


@jit(nopython=True, cache=True)
def hand_rank(ranks, suits):
    """Order-equivalent rank of a 7-card hand. Higher is better.

    ``ranks`` are 0..12 for 2..A and ``suits`` 0..3, matching eval7's own
    encoding so callers pass its numbers straight through. Scalars only: this
    runs twice per showdown inside the walk, and an allocation here is the
    whole cost of the call.
    """
    # Multiplicity as four rank masks: a rank is in `twice` once seen twice, etc.
    mask = 0
    twice = 0
    thrice = 0
    four = 0
    # Per-suit rank masks and counts, unrolled because a kernel array is a heap alloc.
    suit0 = 0
    suit1 = 0
    suit2 = 0
    suit3 = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0

    for i in range(ranks.shape[0]):
        bit = 1 << ranks[i]
        if mask & bit == 0:
            mask |= bit
        elif twice & bit == 0:
            twice |= bit
        elif thrice & bit == 0:
            thrice |= bit
        else:
            four |= bit
        suit = suits[i]
        if suit == 0:
            suit0 |= bit
            count0 += 1
        elif suit == 1:
            suit1 |= bit
            count1 += 1
        elif suit == 2:
            suit2 |= bit
            count2 += 1
        else:
            suit3 |= bit
            count3 += 1

    flush_mask = 0
    if count0 >= 5:
        flush_mask = suit0
    elif count1 >= 5:
        flush_mask = suit1
    elif count2 >= 5:
        flush_mask = suit2
    elif count3 >= 5:
        flush_mask = suit3
    if flush_mask != 0:
        high = _straight_high(flush_mask, _STRAIGHTS, _STRAIGHT_HIGH)
        if high >= 0:
            return (STRAIGHT_FLUSH << 20) | (high << 16)
        return (FLUSH << 20) | _top_ranks(flush_mask, 5, 0)

    # Rank multiplicities, high to low within each class.
    quad = _top_rank(four)
    trips = thrice & ~four
    trip = _top_rank(trips)
    trip2 = _top_rank(trips & ~(1 << trip)) if trip >= 0 else -1
    pairs = twice & ~thrice
    pair = _top_rank(pairs)
    pair2 = _top_rank(pairs & ~(1 << pair)) if pair >= 0 else -1

    if quad >= 0:
        return (QUADS << 20) | (quad << 16) | (_top_ranks(mask, 1, 1 << quad) << 12)

    if trip >= 0 and (pair >= 0 or trip2 >= 0):
        # Two trips is a full house using the higher as the trips; the lower
        # trip's pair beats any separate pair, since it was found first.
        kicker = trip2 if trip2 > pair else pair
        return (FULL_HOUSE << 20) | (trip << 16) | (kicker << 12)

    high = _straight_high(mask, _STRAIGHTS, _STRAIGHT_HIGH)
    if high >= 0:
        return (STRAIGHT << 20) | (high << 16)

    if trip >= 0:
        return (TRIPS << 20) | (trip << 16) | (_top_ranks(mask, 2, 1 << trip) << 4)

    if pair >= 0 and pair2 >= 0:
        skip = (1 << pair) | (1 << pair2)
        return (TWO_PAIR << 20) | (pair << 16) | (pair2 << 12) | (_top_ranks(mask, 1, skip) << 8)

    if pair >= 0:
        return (PAIR << 20) | (pair << 16) | (_top_ranks(mask, 3, 1 << pair) << 4)

    return (HIGH_CARD << 20) | _top_ranks(mask, 5, 0)


@jit(nopython=True, cache=True)
def compare(ranks_a, suits_a, ranks_b, suits_b):
    """-1 if a wins, 1 if b wins, 0 for a tie — ``compare_hands``' convention."""
    left = hand_rank(ranks_a, suits_a)
    right = hand_rank(ranks_b, suits_b)
    if left > right:
        return -1
    if left < right:
        return 1
    return 0
