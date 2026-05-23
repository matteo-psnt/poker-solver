"""The bridge from the precomputed artifact to vector-CFR board contexts.

What matters here is that a context says the same thing the solver would say:
the same bucket for the same (hand, board), a rank order that matches the
evaluator, and a blocking relation that is actually symmetric. A silent
disagreement on any of those would not fail — it would train, and every
measurement taken on top of it would be measuring a different abstraction than
the one that ships.
"""

from __future__ import annotations

from typing import ClassVar

import eval7
import numpy as np
import pytest

from src.core.game.state import FULL_DECK, Street
from src.pipeline.abstraction.vector_universe import (
    build_hand_context,
    build_universe,
    iter_universe,
    sample_boards,
)

BOARD = (0, 14, 27, 39, 51)


class StubAbstraction:
    """Deterministic stand-in with the artifact's interface and none of its size.

    Buckets key on the hand and the visible board so the per-street values
    differ, which is what the transition derivation needs to see.
    """

    counts: ClassVar[dict[Street, int]] = {
        Street.PREFLOP: 169,
        Street.FLOP: 4,
        Street.TURN: 5,
        Street.RIVER: 6,
    }

    def get_bucket(self, hole_cards, board, street):
        if street == Street.PREFLOP:
            return (repr(hole_cards[0])[0].encode()[0] * 3) % 169
        seed = sum(card.__hash__() for card in board) + hash(repr(hole_cards[0]))
        return abs(seed) % self.counts[street]

    def num_buckets(self, street):
        return self.counts[street]


@pytest.fixture(scope="module")
def context():
    return build_hand_context(BOARD, StubAbstraction())


class TestShape:
    def test_live_hands_exclude_the_board(self, context):
        """Forty-seven cards remain, so C(47,2) holdings are live."""
        assert context.num_hands == 47 * 46 // 2
        board = set(BOARD)
        assert not board & set(context.hand_cards.flatten().tolist())

    def test_every_street_gets_a_bucket_per_hand(self, context):
        assert context.bucket_of_hand.shape == (4, context.num_hands)
        for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
            buckets = context.buckets_for(street)
            assert (buckets >= 0).all()
            assert (buckets < StubAbstraction.counts[street]).all()

    def test_blocking_is_symmetric_and_self_blocking(self, context):
        assert (context.blocks == context.blocks.T).all()
        assert context.blocks.diagonal().all()


class TestAgreementWithTheAbstraction:
    def test_buckets_match_asking_the_abstraction_directly(self, context):
        """The context must not reorder or reindex what the abstraction said."""
        abstraction = StubAbstraction()
        cards = [FULL_DECK[i] for i in BOARD]
        for street, seen in ((Street.FLOP, 3), (Street.TURN, 4), (Street.RIVER, 5)):
            buckets = context.buckets_for(street)
            for hand_index in (0, 17, context.num_hands - 1):
                first, second = context.hand_cards[hand_index]
                expected = abstraction.get_bucket(
                    (FULL_DECK[first], FULL_DECK[second]), tuple(cards[:seen]), street
                )
                assert buckets[hand_index] == expected

    def test_showdown_ranks_order_a_known_pair_correctly(self, context):
        """A hand making a better five-card hand must rank above a worse one.

        Anchored to the evaluator rather than to a hard-coded number, so this
        survives an eval7 upgrade while still catching an inverted comparison.
        """
        board = [eval7.Card(repr(FULL_DECK[i])) for i in BOARD]
        pairs = [
            (
                index,
                eval7.evaluate(
                    [*board, eval7.Card(repr(FULL_DECK[a])), eval7.Card(repr(FULL_DECK[b]))]
                ),
            )
            for index, (a, b) in enumerate(context.hand_cards[:40].tolist())
        ]
        best = max(pairs, key=lambda item: item[1])[0]
        worst = min(pairs, key=lambda item: item[1])[0]
        assert context.showdown_rank[best] > context.showdown_rank[worst]


class TestUniverse:
    def test_sampled_boards_are_five_distinct_cards(self):
        for board in sample_boards(np.random.default_rng(3), 25):
            assert board.shape == (5,)
            assert len(set(board.tolist())) == 5

    def test_a_universe_is_one_context_per_board(self):
        universe = build_universe(StubAbstraction(), 3, rng=np.random.default_rng(1))
        assert len(universe) == 3
        assert all(c.num_hands == 47 * 46 // 2 for c in universe)

    def test_a_short_board_is_rejected(self):
        with pytest.raises(ValueError, match="five cards"):
            build_hand_context((1, 2, 3), StubAbstraction())


class TestStreaming:
    def test_iter_universe_yields_the_same_contexts_as_the_list(self):
        """Streaming must not change the universe, only when it exists in RAM."""
        listed = build_universe(StubAbstraction(), 3, rng=np.random.default_rng(11))
        streamed = list(iter_universe(StubAbstraction(), 3, rng=np.random.default_rng(11)))
        assert len(listed) == len(streamed)
        for a, b in zip(listed, streamed, strict=True):
            assert np.array_equal(a.hand_cards, b.hand_cards)
            assert np.array_equal(a.bucket_of_hand, b.bucket_of_hand)

    def test_iter_universe_does_not_retain_what_it_yields(self):
        """The point of streaming: one context resident, not the whole universe.

        A context carries an ``(H, H)`` blocking matrix — about 1.2 MB at a full
        board — so twenty thousand of them is 23 GB in a list. Deriving from
        enough boards to populate a 600-bucket river is only possible if the
        generator lets each one go.
        """
        import sys

        stream = iter_universe(StubAbstraction(), 5, rng=np.random.default_rng(2))
        first = next(stream)
        # Only this test frame and the argument tuple should hold it; a generator
        # that accumulated into a list would push the count higher.
        assert sys.getrefcount(first) <= 4
