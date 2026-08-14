"""Does the native evaluator order hands exactly as eval7 does?

`numba_eval` exists because a `nopython` kernel cannot call eval7, and a
traversal that leaves the kernel at every showdown pays back more in boundary
crossings than compiling saved. That makes this the gate on compiling the walk:
if the ordering is not exact, every showdown payoff shifts and every published
number moves with it.

The bar is the ORDER, not the numbers. Nothing reads absolute hand strength —
`get_payoff` only asks which of two hands wins — so agreement means the map
between eval7's rank and this one is strictly monotonic AND injective both
ways. Two failures hide in that phrasing and both are checked: a tie split into
a win (one eval7 rank reaching two native ranks) and a win collapsed into a tie
(two eval7 ranks sharing one native rank).

`test_every_seven_card_hand` is the real assurance and runs all 133,784,560 of
them in ~6 minutes. It is affordable only because the bookkeeping is per RANK
CLASS, not per hand: there are 4,824 distinct values, so the dictionaries stay
tiny however many hands pass through.
"""

from __future__ import annotations

import itertools
import random

import eval7
import numpy as np
import pytest

from src.core.game.numba_eval import hand_rank

DECK = [eval7.Card(rank + suit) for rank in "23456789TJQKA" for suit in "cdhs"]
CARD_RANKS = np.array([card.rank for card in DECK], dtype=np.int64)
CARD_SUITS = np.array([card.suit for card in DECK], dtype=np.int64)

# Every category, plus the shapes that catch a careless evaluator: both
# straight-flush ends, two trips resolving to a boat, three pairs resolving to
# the best two, and the wheel, where the ace plays low.
NAMED_HANDS = {
    "royal flush": ("As", "Ks", "Qs", "Js", "Ts", "2h", "3d"),
    "straight flush": ("9s", "8s", "7s", "6s", "5s", "Ah", "Kd"),
    "steel wheel": ("As", "2s", "3s", "4s", "5s", "Kh", "Qd"),
    "quads": ("Ah", "Ad", "Ac", "As", "Kh", "Qd", "Js"),
    "full house": ("Ah", "Ad", "Ac", "Kh", "Kd", "Qs", "Jc"),
    "two trips": ("Ah", "Ad", "Ac", "Kh", "Kd", "Ks", "Qc"),
    "flush": ("Ks", "Js", "9s", "5s", "3s", "Ah", "Qd"),
    "broadway straight": ("Ah", "Kd", "Qc", "Js", "Th", "3d", "2c"),
    "wheel": ("Ah", "2d", "3c", "4s", "5h", "Kd", "Qc"),
    "trips": ("Ah", "Ad", "Ac", "Kh", "Qd", "Js", "9c"),
    "two pair": ("Ah", "Ad", "Kc", "Ks", "Qh", "Jd", "9c"),
    "three pairs": ("2h", "2d", "3c", "3s", "4h", "4d", "5c"),
    "pair": ("Ah", "Ad", "Kc", "Qs", "Jh", "9d", "7c"),
    "high card": ("Ah", "Kd", "Qc", "Js", "9h", "7d", "5c"),
}


def _native(cards) -> int:
    ranks = np.array([c.rank for c in cards], dtype=np.int64)
    suits = np.array([c.suit for c in cards], dtype=np.int64)
    return int(hand_rank(ranks, suits))


def _assert_same_order(pairs) -> int:
    """``pairs`` is an iterable of (eval7 rank, native rank). Returns class count."""
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for theirs, ours in pairs:
        known = forward.get(theirs)
        if known is None:
            assert ours not in reverse, (
                f"native rank {ours} shared by eval7 ranks {reverse[ours]} and "
                f"{theirs} — a win collapsed into a tie"
            )
            forward[theirs] = ours
            reverse[ours] = theirs
        else:
            assert known == ours, (
                f"eval7 rank {theirs} reached native ranks {known} and {ours} "
                "— a tie split into a win"
            )

    ordered = sorted(forward.items())
    natives = [ours for _, ours in ordered]
    assert natives == sorted(natives), "the two evaluators disagree on order"
    return len(forward)


class TestOrderingMatchesEval7:
    def test_named_categories(self):
        hands = [[eval7.Card(n) for n in names] for names in NAMED_HANDS.values()]
        _assert_same_order((eval7.evaluate(h), _native(h)) for h in hands)

    def test_categories_rank_against_each_other_as_poker_does(self):
        """A sanity check the ordering test cannot give: the categories' actual order."""
        order = [
            "high card",
            "pair",
            "two pair",
            "trips",
            "wheel",
            "flush",
            "full house",
            "quads",
            "steel wheel",
            "royal flush",
        ]
        ranked = [_native([eval7.Card(n) for n in NAMED_HANDS[name]]) for name in order]
        assert ranked == sorted(ranked), dict(zip(order, ranked, strict=True))

    def test_a_random_sweep(self):
        rng = random.Random(20260814)
        hands = [rng.sample(DECK, 7) for _ in range(30_000)]
        classes = _assert_same_order((eval7.evaluate(h), _native(h)) for h in hands)
        assert classes > 2_000, f"only {classes} rank classes reached; sweep is too thin"

    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_every_seven_card_hand(self):
        """All 133,784,560 of them. The gate, and it is affordable — see the module docstring."""
        ranks = np.empty(7, dtype=np.int64)
        suits = np.empty(7, dtype=np.int64)

        def pairs():
            for combo in itertools.combinations(range(52), 7):
                for slot in range(7):
                    card = combo[slot]
                    ranks[slot] = CARD_RANKS[card]
                    suits[slot] = CARD_SUITS[card]
                yield eval7.evaluate([DECK[i] for i in combo]), int(hand_rank(ranks, suits))

        assert _assert_same_order(pairs()) == 4_824
