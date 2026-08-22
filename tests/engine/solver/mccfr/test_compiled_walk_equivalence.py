"""Does the compiled kernel compute what the tree walk computes?

THE invariant the production default rests on, and the one that makes the
fallback in `StaticTreeSolver.train_iteration` a throughput choice rather than
a silent correctness fork. The kernel cannot run against an abstraction that
only implements `get_bucket` -- every small-game and unit-test stand-in -- so a
solver quietly uses the tree walk there. That is invisible ONLY while the two
agree bit for bit. The moment they do not, the same config computes different
strategies depending on which abstraction happened to be loaded, and nothing
would say so. This test is what keeps that from being possible.

Bit-identity, not tolerance: every shared array byte-for-byte, every
per-iteration utility, and both random streams left in the same place -- the
last because identical arrays alone would not prove the draws happened in the
same order.

The fixture mirrors PRODUCTION's per-street bucket counts on purpose. Both
failures this change hit in the wild lived in the gap between a convenient
fixture and the real artifact: a sentinel shared across streets whose dtypes
differ, and an absent-board path unreachable under full coverage. A fixture
with one bucket count for every street makes the three sentinels coincide and
cannot see the first at all.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.compiled_walk import CompiledContext, run_iteration, run_iterations
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline.abstraction.postflop.bucketer import (
    N_HAND_COLUMNS,
    DenseBucketer,
    bucket_dtype,
    build_hand_column_index,
)
from src.pipeline.abstraction.postflop.suit_isomorphism import canonical_board_id
from tests.test_helpers import make_test_config

ARRAYS = ("regrets", "strategy_sum", "reach_counts", "cumulative_utility", "visited")
# Production's counts, because the DTYPES they imply are the thing that
# matters: 100 fits uint8 (sentinel 255) while 300 and 600 need uint16
# (65535). A fixture using one count for every street makes the three
# sentinels coincide, which is exactly why the gate could not see the
# one-sentinel-for-all-streets bug that killed the first real run.
BUCKETS = {Street.FLOP: 100, Street.TURN: 300, Street.RIVER: 600}


def artifact():
    """A DenseBucketer covering every board reachable from a small deck slice.

    Full coverage matters: a missing board makes the Python path raise and the
    kernel return -1, which is a difference in behaviour rather than in
    arithmetic and would mask the thing under test.
    """
    import itertools

    ids: dict[Street, set[int]] = {s: set() for s in (Street.FLOP, Street.TURN, Street.RIVER)}
    for size, street in ((3, Street.FLOP), (4, Street.TURN), (5, Street.RIVER)):
        for combo in itertools.combinations(FULL_DECK, size):
            ids[street].add(canonical_board_id(combo)[0])

    rng = np.random.default_rng(3)
    board_ids, matrices = {}, {}
    for street, values in ids.items():
        sorted_ids = np.array(sorted(values), dtype=np.int64)
        board_ids[street] = sorted_ids
        dtype = bucket_dtype(BUCKETS[street])
        matrices[street] = rng.integers(
            0, BUCKETS[street], size=(sorted_ids.size, N_HAND_COLUMNS), dtype=dtype
        )
    return DenseBucketer(dict(BUCKETS), board_ids, matrices, build_hand_column_index())


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_compiled_kernel_is_bit_identical_to_the_tree_walk():
    config = make_test_config(seed=42, starting_stack=60, iteration_weighting="dcfr")
    bucketer = artifact()
    action_model = ActionModel(config)
    tree = build_betting_tree(
        GameRules(config.game.small_blind, config.game.big_blind),
        action_model,
        bucketer,
        starting_stack=config.game.starting_stack,
    )

    def arm(compiled: bool, iterations: int):
        storage = StaticArrayStorage(tree)
        solver = StaticTreeSolver(action_model, bucketer, storage, config, tree=tree)
        context = CompiledContext(tree, bucketer, FULL_DECK) if compiled else None
        random.seed(1234)
        np.random.seed(1234)
        values = []
        for i in range(iterations):
            solver.iteration = i
            values.append(
                run_iteration(solver, context, i) if compiled else solver.train_iteration()
            )
        arrays = {name: getattr(storage, name).copy() for name in ARRAYS}
        return arrays, values, (random.random(), np.random.random())

    n = 250
    reference, ref_values, ref_streams = arm(False, n)
    got, got_values, got_streams = arm(True, n)

    for name in ARRAYS:
        assert np.array_equal(got[name], reference[name]), (
            f"{name} diverged in {int((got[name] != reference[name]).sum())} of "
            f"{reference[name].size} entries — the fallback is now a correctness fork"
        )
    assert got_values == ref_values
    # Same draws, in the same order: identical arrays alone would not show it.
    assert got_streams == ref_streams

    # And the comparison must not have passed on two empty tables.
    assert int(got["visited"].sum()) > 0
    assert np.count_nonzero(got["strategy_sum"]) > 0


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_one_crossing_per_range_is_the_same_stream_as_one_per_iteration():
    """``run_iterations`` hands the generators over once for the whole range.

    What that must not change: the draws, their order, or where both streams
    are left -- a range that advanced the kernel's copy but restored Python's
    from the wrong place would diverge on the NEXT call, not this one, so the
    check draws once more from each stream after the range.
    """
    config = make_test_config(seed=42, starting_stack=60, iteration_weighting="dcfr")
    bucketer = artifact()
    action_model = ActionModel(config)
    tree = build_betting_tree(
        GameRules(config.game.small_blind, config.game.big_blind),
        action_model,
        bucketer,
        starting_stack=config.game.starting_stack,
    )
    context = CompiledContext(tree, bucketer, FULL_DECK)

    def arm(ranges):
        storage = StaticArrayStorage(tree)
        solver = StaticTreeSolver(action_model, bucketer, storage, config, tree=tree)
        random.seed(1234)
        np.random.seed(1234)
        values = []
        for block in ranges:
            values.extend(float(v) for v in run_iterations(solver, context, block))
        arrays = {name: getattr(storage, name).copy() for name in ARRAYS}
        return arrays, values, (random.random(), np.random.random()), solver.applied_updates

    n = 250
    singles = arm([range(i, i + 1) for i in range(n)])
    batched = arm([range(100), range(100, n)])
    for name in ARRAYS:
        assert np.array_equal(batched[0][name], singles[0][name]), name
    assert batched[1:] == singles[1:]
    assert int(singles[0]["visited"].sum()) > 0

    # A worker's range is interleaved, so the kernel's loop must honour the step.
    order = [range(s, n, 3) for s in range(3)]
    strided_singles = arm([range(i, i + 1) for block in order for i in block])
    strided = arm(order)
    for name in ARRAYS:
        assert np.array_equal(strided[0][name], strided_singles[0][name]), name
    assert strided[1:] == strided_singles[1:]
