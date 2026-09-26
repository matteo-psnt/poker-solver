"""The fused reductions must reproduce numpy's BYTES, not just its values.

Every live run's trajectory rides on them, so a one-ulp drift is a lineage
break. Measured while building them: sequential association mismatched
reduceat on every run of 3+ hands, and numpy starts `sum(axis=-1)` at +0.0 but
a reduceat run's tail at -0.0, which only an all -0.0 row reveals.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.solver.vector.fused import (
    bucket_dot,
    gathered_action_sum,
    segment_blocks,
    segment_rows,
)
from src.engine.solver.vector.kernel import build_segments

HANDS = 1081


def _data(num_buckets: int, actions: int, seed: int):
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, num_buckets, HANDS)
    block = rng.random((5, num_buckets, actions), dtype=np.float32)
    children = (rng.standard_normal((5, HANDS, actions)) * 3).astype(np.float32)
    # Signed zeros are where numpy's two starting values show: whole rows of -0.0
    # for `sum(axis=-1)`, whole bucket runs of it for reduceat.
    children[:, ::5, :] = -0.0
    children[:, ::7, -1] = 0.0
    children[1] = -0.0
    return buckets, block, children


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("actions", [2, 3, 6])
def test_bucket_dot_matches_numpy(actions):
    buckets, block, children = _data(600, actions, seed=actions)
    expected = (block[:, buckets, :] * children).sum(axis=-1)
    assert _bytes_equal(bucket_dot(block, buckets, children), expected)


@pytest.mark.parametrize("actions", [2, 3, 6])
def test_gathered_action_sum_matches_gather_then_sum(actions):
    """The kernel's `gather_children(...).sum(axis=-1)`, reading node and terminal storage."""
    rng = np.random.default_rng(20 + actions)
    nodes = (rng.standard_normal((40, HANDS)) * 3).astype(np.float32)
    terminals = (rng.standard_normal((70, HANDS)) * 3).astype(np.float32)
    nodes[::3] = -0.0
    terminals[::2] = -0.0
    is_terminal = rng.random((25, actions)) < 0.5
    targets = np.where(
        is_terminal, rng.integers(0, 70, (25, actions)), rng.integers(0, 40, (25, actions))
    )
    gathered = np.empty((25, HANDS, actions), dtype=np.float32)
    for a in range(actions):
        term = is_terminal[:, a]
        gathered[term, :, a] = terminals[targets[term, a]]
        gathered[~term, :, a] = nodes[targets[~term, a]]
    expected = gathered.sum(axis=-1)
    got = gathered_action_sum(targets, is_terminal, nodes, terminals)
    assert _bytes_equal(got, expected)


# 600 buckets gives runs of 1-8 hands; 40 gives ~27, past numpy's 8-lane
# threshold; 3 gives ~360, past its 128-element split.
@pytest.mark.parametrize("num_buckets", [600, 40, 3])
def test_segment_sums_match_reduceat(num_buckets):
    buckets, block, children = _data(num_buckets, 4, seed=num_buckets)
    value = (block[:, buckets, :] * children).sum(axis=-1)
    segments = build_segments(buckets)
    expected_children = np.add.reduceat(
        children[:, segments.hand_order, :], segments.segment_start, axis=1
    )
    expected_value = np.add.reduceat(value[:, segments.hand_order], segments.segment_start, axis=1)
    got_children = segment_blocks(children, segments.hand_order, segments.segment_start)
    got_value = segment_rows(value, segments.hand_order, segments.segment_start)
    assert _bytes_equal(got_children, expected_children)
    assert _bytes_equal(got_value, expected_value)
