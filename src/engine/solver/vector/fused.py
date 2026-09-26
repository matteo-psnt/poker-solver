"""The backward pass's hand-space reductions as single loops, byte-identical to numpy.

Each replaces a numpy expression that materialised an ``(nodes, hands, actions)``
temporary per chunk -- the kernel's dominant cost, and memory traffic that every
worker on a node contends for. Association and the signed-zero identity follow
numpy's exactly, pinned against it in ``tests/engine/solver/vector/test_fused.py``.
"""

from __future__ import annotations

import numpy as np
from numba import jit

# Signed zero is where the two numpy paths differ, so each keeps its own start:
# a `sum(axis=-1)` begins at +0.0, a reduceat run's pairwise tail at -0.0. An all
# -0.0 row sums to +0.0 on the first and -0.0 on the second.
_AXIS_START = np.float32(0.0)
_RUN_START = np.float32(-0.0)
_LEAF = 128  # numpy's pairwise block: above it the run is split in two


@jit(nopython=True, cache=True)
def _leaf(values, n):
    """numpy's ``pairwise_sum`` at or below its block: a loop under 8, else 8 lanes."""
    if n < 8:
        res = _RUN_START
        for i in range(n):
            res += values[i]
        return res
    r0, r1, r2, r3 = values[0], values[1], values[2], values[3]
    r4, r5, r6, r7 = values[4], values[5], values[6], values[7]
    i = 8
    limit = n - (n % 8)
    while i < limit:
        r0 += values[i]
        r1 += values[i + 1]
        r2 += values[i + 2]
        r3 += values[i + 3]
        r4 += values[i + 4]
        r5 += values[i + 5]
        r6 += values[i + 6]
        r7 += values[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += values[i]
        i += 1
    return res


@jit(nopython=True, cache=True)
def _pairwise(values, n):
    """numpy's ``pairwise_sum`` for any length; the split is iterative because a
    cached numba function that recurses segfaults on reload."""
    if n <= _LEAF:
        return _leaf(values, n)
    starts = np.empty(64, dtype=np.int64)
    sizes = np.empty(64, dtype=np.int64)
    lefts = np.empty(64, dtype=np.float32)
    has_left = np.zeros(64, dtype=np.int64)
    starts[0] = 0
    sizes[0] = n
    depth = 1
    while True:
        top = depth - 1
        if sizes[top] > _LEAF:
            half = sizes[top] // 2
            half -= half % 8
            starts[depth] = starts[top]
            sizes[depth] = half
            has_left[depth] = 0
            depth += 1
            continue
        value = _leaf(values[starts[top] :], sizes[top])
        depth -= 1
        while True:
            if depth == 0:
                return value
            parent = depth - 1
            if has_left[parent] == 0:
                lefts[parent] = value
                has_left[parent] = 1
                half = sizes[parent] // 2
                half -= half % 8
                starts[depth] = starts[parent] + half
                sizes[depth] = sizes[parent] - half
                has_left[depth] = 0
                depth += 1
                break
            value = lefts[parent] + value
            depth -= 1


@jit(nopython=True, cache=True)
def bucket_dot(block, buckets, children):
    """``(block[:, buckets, :] * children).sum(axis=-1)``: the actor's node value."""
    n, hands, actions = children.shape
    out = np.empty((n, hands), dtype=np.float32)
    for i in range(n):
        for h in range(hands):
            b = buckets[h]
            acc = _AXIS_START
            for a in range(actions):
                acc += block[i, b, a] * children[i, h, a]
            out[i, h] = acc
    return out


@jit(nopython=True, cache=True)
def gathered_action_sum(targets, is_terminal, node_values, terminal_values):
    """``action_sum`` of the children ``gather_children`` would build, without building them."""
    n, actions = targets.shape
    hands = node_values.shape[1]
    out = np.empty((n, hands), dtype=np.float32)
    for i in range(n):
        for h in range(hands):
            acc = _AXIS_START
            for a in range(actions):
                t = targets[i, a]
                if is_terminal[i, a]:
                    acc += terminal_values[t, h]
                else:
                    acc += node_values[t, h]
            out[i, h] = acc
    return out


@jit(nopython=True, cache=True)
def card_mass(reach, hand_cards):
    """``(T, 52)`` range mass holding each card: a hand adds its reach to both cards.

    Replaces ``reach @ incidence``, whose bytes depend on how many rows BLAS is
    handed; the fold values are computed per chunk now, so they must not.
    """
    rows, hands = reach.shape
    out = np.zeros((rows, 52), dtype=np.float32)
    for t in range(rows):
        for h in range(hands):
            r = reach[t, h]
            out[t, hand_cards[h, 0]] += r
            out[t, hand_cards[h, 1]] += r
    return out


@jit(nopython=True, cache=True)
def regret_matched(source, slot_base, stride, num_buckets, num_actions, uniform):
    """``regret_match(source[slot_index(...)])`` read straight off storage, ``(n, B, A)``.

    ``np.maximum(x, 0.0)`` is +0.0 for either zero and NaN for NaN; the row
    total is a +0.0-started ``sum(axis=-1)``.
    """
    n = slot_base.shape[0]
    out = np.empty((n, num_buckets, num_actions), dtype=np.float32)
    positive = np.empty(num_actions, dtype=np.float32)
    for i in range(n):
        for b in range(num_buckets):
            base = slot_base[i] + b * stride
            total = _AXIS_START
            for a in range(num_actions):
                x = source[base + a]
                p = x if (x > 0.0 or x != x) else np.float32(0.0)
                positive[a] = p
                total += p
            if total > 0.0:
                for a in range(num_actions):
                    out[i, b, a] = positive[a] / total
            else:
                for a in range(num_actions):
                    out[i, b, a] = uniform
    return out


@jit(nopython=True, cache=True)
def add_into_slots(target, slot_base, stride, block):
    """``target[slot_index(...)] += block``: one row per node, ``(n, B, A)`` bucket-major."""
    n, num_buckets, num_actions = block.shape
    for i in range(n):
        for b in range(num_buckets):
            base = slot_base[i] + b * stride
            for a in range(num_actions):
                target[base + a] += block[i, b, a]


@jit(nopython=True, cache=True)
def add_weighted_into_slots(target, slot_base, stride, mass, strategy):
    """``target[slot_index(...)] += mass[:, :, None] * strategy`` without the product array."""
    n, num_buckets, num_actions = strategy.shape
    for i in range(n):
        for b in range(num_buckets):
            base = slot_base[i] + b * stride
            m = mass[i, b]
            for a in range(num_actions):
                target[base + a] += m * strategy[i, b, a]


@jit(nopython=True, cache=True)
def deposit_children(below, own, opposing, strategy, buckets, targets, is_terminal, actor, other):
    """Write each node child's ranges: the actor's reach times its strategy, the other's as is.

    ``own[:, :, None] * strategy[:, buckets, :]`` sliced per action, without the
    ``(n, hands, actions)`` product; terminal children are the caller's.
    """
    n, actions = targets.shape
    hands = own.shape[1]
    for i in range(n):
        for a in range(actions):
            if is_terminal[i, a]:
                continue
            t = targets[i, a]
            for h in range(hands):
                below[actor, t, h] = own[i, h] * strategy[i, buckets[h], a]
                below[other, t, h] = opposing[i, h]


@jit(nopython=True, cache=True)
def _run_sum(head, scratch, rest):
    """numpy reduces a run as its first element plus the pairwise sum of the rest."""
    if rest == 0:
        return head
    return head + _pairwise(scratch, rest)


@jit(nopython=True, cache=True)
def segment_rows(values, hand_order, segment_start):
    """``np.add.reduceat(values[:, hand_order], segment_start, axis=1)`` for ``(n, hands)``."""
    n, hands = values.shape
    segments = segment_start.shape[0]
    out = np.empty((n, segments), dtype=np.float32)
    scratch = np.empty(hands, dtype=np.float32)
    for i in range(n):
        for s in range(segments):
            lo = segment_start[s]
            hi = segment_start[s + 1] if s + 1 < segments else hands
            rest = hi - lo - 1
            for k in range(rest):
                scratch[k] = values[i, hand_order[lo + 1 + k]]
            out[i, s] = _run_sum(values[i, hand_order[lo]], scratch, rest)
    return out


@jit(nopython=True, cache=True)
def segment_blocks(values, hand_order, segment_start):
    """``np.add.reduceat(values[:, hand_order, :], segment_start, axis=1)`` for ``(n, hands, A)``."""
    n, hands, actions = values.shape
    segments = segment_start.shape[0]
    out = np.empty((n, segments, actions), dtype=np.float32)
    scratch = np.empty(hands, dtype=np.float32)
    for i in range(n):
        for s in range(segments):
            lo = segment_start[s]
            hi = segment_start[s + 1] if s + 1 < segments else hands
            rest = hi - lo - 1
            for a in range(actions):
                for k in range(rest):
                    scratch[k] = values[i, hand_order[lo + 1 + k], a]
                out[i, s, a] = _run_sum(values[i, hand_order[lo], a], scratch, rest)
    return out


__all__ = (
    "add_into_slots",
    "add_weighted_into_slots",
    "bucket_dot",
    "card_mass",
    "deposit_children",
    "gathered_action_sum",
    "regret_matched",
    "segment_blocks",
    "segment_rows",
)
