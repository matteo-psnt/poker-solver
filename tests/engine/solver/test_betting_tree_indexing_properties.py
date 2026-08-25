"""Properties of the `(node_id, bucket) -> row -> slots` indexing, over the whole
tree rather than the handful of nodes an example test can name.

This is the one arithmetic in the solver where being wrong is silent. Two
infosets sharing a slot range train each other's regrets and nothing raises;
`static_array.view` guards only the bucket bound, and its own comment says an
out-of-range bucket "would alias another node's infoset". It has happened at
this exact seam: the vector bridge mis-indexed 1,320 of 1,326 preflop rows,
contaminating every vector-vs-scalar and warm-start figure taken before the fix,
because a row order was rebuilt by a route that disagreed with `row()`.

So: `row_to_infoset` must invert `row`, `slots` must agree with the cached
`row_slot_starts`/`row_widths` the vector path reduces over, and no two infosets
may share a slot.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from tests.test_helpers import make_test_config

# Deliberately small, as in `test_betting_tree.py`: the indexing claims are
# abstraction-independent, and a big bucket count only makes enumeration slow.
BUCKETS = {Street.FLOP: 4, Street.TURN: 5, Street.RIVER: 6}


@pytest.fixture(scope="module")
def tree():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=40)
    return BettingTree(
        GameRules(small_blind=config.game.small_blind, big_blind=config.game.big_blind),
        ActionModel(config),
        starting_stack=config.game.starting_stack,
        buckets_per_street=BUCKETS,
    )


@st.composite
def _infosets(draw, tree):
    """A `(node_id, bucket)` that exists: bucket counts vary by street."""
    node_id = draw(st.integers(min_value=0, max_value=len(tree.nodes) - 1))
    bucket = draw(st.integers(min_value=0, max_value=int(tree.buckets_per_node[node_id]) - 1))
    return node_id, bucket


@pytest.mark.timeout(60)
@given(st.data())
def test_row_to_infoset_inverts_row(tree, data):
    node_id, bucket = data.draw(_infosets(tree))
    row = tree.row(node_id, bucket)

    assert 0 <= row < tree.num_rows
    assert tree.row_to_infoset(row) == (node_id, bucket)


@pytest.mark.timeout(60)
@given(st.data())
def test_slots_agree_with_the_cached_row_layout(tree, data):
    """`row_slot_starts`/`row_widths` are what the vector bridge reduces over.
    They are built from cumulative widths, `slots()` from per-node strides --
    two routes to one number, which is exactly how they drift apart."""
    node_id, bucket = data.draw(_infosets(tree))
    start, end = tree.slots(node_id, bucket)
    row = tree.row(node_id, bucket)

    assert 0 <= start < end <= tree.num_slots
    assert end - start == int(tree.num_actions[node_id])
    assert start == int(tree.row_slot_starts[row])
    assert end - start == int(tree.row_widths[row])


@pytest.mark.timeout(60)
@given(st.data())
def test_two_infosets_never_share_a_slot(tree, data):
    """The aliasing `static_array.view` fears, checked from the other side: two
    distinct infosets, drawn independently, must not overlap."""
    first = data.draw(_infosets(tree))
    second = data.draw(_infosets(tree))
    if first == second:
        return

    a_start, a_end = tree.slots(*first)
    b_start, b_end = tree.slots(*second)

    assert a_end <= b_start or b_end <= a_start, f"{first} and {second} share slots"


@pytest.mark.timeout(60)
def test_the_rows_tile_the_slot_array_exactly(tree):
    """Not a property test -- the closing statement the properties sample. Every
    slot belongs to exactly one row, so nothing is orphaned or double-owned."""
    starts = tree.row_slot_starts
    widths = tree.row_widths

    assert starts.shape == widths.shape == (tree.num_rows,)
    assert int(starts[0]) == 0
    assert np.array_equal(starts[1:], np.cumsum(widths)[:-1])
    assert int(np.sum(widths)) == tree.num_slots
