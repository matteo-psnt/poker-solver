"""Properties of board canonicalization, over boards the example tests do not name.

Both pin MEASURED failures. Order-dependence reached training: `canonicalize_board`
sorted on rank alone, Python's sort is stable, and on a paired board the suit
labelling followed the caller's tuple order -- `chance.py` deals the flop as
`deck.cards[:3]` off a shuffle and nothing sorts it, so a hand's bucket became a
function of the deal order. It changed the hand id on 7.2% of flops and the bucket
on 0.175%, and cc77a61 fixed it by sorting on `(rank, suit)`. Suit-relabelling
invariance is the property the whole abstraction is built on: it is what makes one
canonical board stand for all 4! of its relabellings.

A residual gap is NOT covered here and is not a regression in these properties:
two hands related by a board-fixing suit permutation still get distinct canonical
ids (0.19% of equivalent hand-pairs land in different buckets). Fixing it means
canonicalizing over the board's stabilizer and re-precomputing.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.game.state import Card
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    canonical_board_id,
    canonical_hand_id,
    canonicalize_board,
    get_canonical_board_id,
)

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]


def _boards(size: int) -> st.SearchStrategy[tuple[str, ...]]:
    """Boards of `size` distinct cards, as rank+suit strings."""
    return st.lists(st.sampled_from(DECK), min_size=size, max_size=size, unique=True).map(tuple)


def _cards(board: tuple[str, ...]) -> tuple[Card, ...]:
    return tuple(Card.new(c) for c in board)


@st.composite
def _deals(draw, size: int) -> tuple[tuple[str, ...], tuple[str, str]]:
    """A board of `size` cards plus two hole cards, all distinct."""
    dealt = draw(st.lists(st.sampled_from(DECK), min_size=size + 2, max_size=size + 2, unique=True))
    return tuple(dealt[:size]), (dealt[size], dealt[size + 1])


@pytest.mark.timeout(60)
@given(st.integers(min_value=3, max_value=5).flatmap(_deals))
def test_the_bucket_key_is_a_function_of_the_board_as_a_set(deal):
    """Deal order must not reach the bucket. Every permutation, not a sample of
    them: the defect only showed on the orderings that tied.

    The pair `(board id, hand id)` is what a bucket is looked up by, and it is
    the HAND half the pre-fix bug moved -- the board id was already invariant,
    which is why an assertion on it alone would have watched this bug go past.
    """
    board, hole = deal
    cards = (Card.new(hole[0]), Card.new(hole[1]))
    keys = set()
    for order in itertools.permutations(board):
        board_id, labels = canonical_board_id(_cards(order))
        keys.add((board_id, canonical_hand_id(cards, labels)))

    assert len(keys) == 1, f"{board} + {hole} bucket differently by board order: {sorted(keys)}"


@pytest.mark.timeout(30)
@given(
    st.integers(min_value=3, max_value=5).flatmap(_boards),
    st.permutations(SUITS),
)
def test_canonical_board_id_is_invariant_under_relabelling_the_suits(board, relabelled):
    """[T♠ 9♥ 8♠] and [T♥ 9♠ 8♥] are the same board. If they were not, the
    abstraction would be storing every suit permutation of a board separately."""
    swap = dict(zip(SUITS, relabelled, strict=True))
    other = tuple(card[0] + swap[card[1]] for card in board)

    assert canonical_board_id(_cards(board))[0] == canonical_board_id(_cards(other))[0]


@pytest.mark.timeout(30)
@given(st.integers(min_value=3, max_value=5).flatmap(_boards))
def test_the_two_canonicalisation_paths_agree(board):
    """`canonical_board_id` is the allocation-free copy of `canonicalize_board`
    on the runtime lookup path. Two implementations of one mapping drift."""
    cards = _cards(board)
    canonical, mapping = canonicalize_board(cards)
    board_id, labels = canonical_board_id(cards)

    assert labels == mapping.mapping
    assert get_canonical_board_id(canonical) == board_id
