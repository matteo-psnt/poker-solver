"""Would the equivalence gate notice if the edge table were wrong?

``test_tree_traversal_equivalence`` asserts two traversals agree bit for bit.
That is only worth something if disagreement is reachable — a comparison that
cannot fail reads exactly like one that cannot break, and this repo has already
been bitten once by a guard that had been silently dead for weeks.

So each test here corrupts the fast path's data in one specific way and
requires the comparison to catch it. The mutations are chosen to be the four
distinct things the edge table can get wrong, not four flavours of the same
thing: what a terminal pays, where a row lives, how many cards the dealer owes,
and when the random stream is touched.

Mutating the TREE rather than the code is what makes this work: the
state-based traversal reads none of `edges`, `node_spec` or `TerminalOutcome`
— it rebuilds a ``GameState`` and asks the rules engine. So a corrupted table
moves one arm of the comparison and not the other, which is the same asymmetry
a real bug would have.
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np

from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from tests.engine.solver.mccfr.test_tree_traversal_equivalence import ARRAYS, build
from tests.test_helpers import make_test_config

ITERATIONS = 120


def _train(config, built, *, walk_tree: bool):
    action_model, abstraction, tree = built
    storage = StaticArrayStorage(tree)
    solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)
    solver._walk_tree = walk_tree
    random.seed(4242)
    np.random.seed(4242)
    for iteration in range(ITERATIONS):
        solver.iteration = iteration
        solver.train_iteration()
    return {name: getattr(storage, name).copy() for name in ARRAYS}


def _assert_gate_catches(mutate, *, expect_stream_shift: bool = False):
    """Corrupt the fast path's view of one tree; require the arrays to diverge."""
    config = make_test_config(seed=42, starting_stack=200, iteration_weighting="dcfr")

    clean = build(config)
    reference = _train(config, clean, walk_tree=False)
    healthy = _train(config, build(config), walk_tree=True)
    assert all(np.array_equal(healthy[name], reference[name]) for name in ARRAYS), (
        "the unmutated fast path already disagrees; this test cannot attribute anything"
    )

    broken_build = build(config)
    mutations = mutate(broken_build[2])
    assert mutations > 0, "the mutation matched nothing — it would prove nothing"
    broken = _train(config, broken_build, walk_tree=True)

    assert any(not np.array_equal(broken[name], reference[name]) for name in ARRAYS), (
        f"{mutations} mutations applied and every array still matched — "
        "the equivalence comparison does not bind"
    )
    if expect_stream_shift:
        assert not np.array_equal(broken["visited"], reference["visited"]), (
            "a changed draw order should move which rows get reached at all"
        )


def test_a_swapped_payoff_table_is_caught():
    """Winner and loser columns exchanged: every showdown pays backwards."""

    def mutate(tree) -> int:
        count = 0
        for node_id, edges in enumerate(tree.edges):
            rebuilt = []
            for edge in edges:
                terminal = edge.terminal
                if terminal is None or terminal.is_fold:
                    rebuilt.append(edge)
                    continue
                swapped = dataclasses.replace(terminal, win=terminal.lose, lose=terminal.win)
                rebuilt.append(dataclasses.replace(edge, terminal=swapped))
                count += 1
            tree.edges[node_id] = tuple(rebuilt)
        _resync(tree)
        return count

    _assert_gate_catches(mutate)


def test_a_fold_paying_the_wrong_seat_is_caught():
    """The one terminal whose winner is decided by the betting line, not cards."""

    def mutate(tree) -> int:
        count = 0
        for node_id, edges in enumerate(tree.edges):
            rebuilt = []
            for edge in edges:
                terminal = edge.terminal
                if terminal is None or not terminal.is_fold:
                    rebuilt.append(edge)
                    continue
                flipped = dataclasses.replace(terminal, fold=(terminal.fold[1], terminal.fold[0]))
                rebuilt.append(dataclasses.replace(edge, terminal=flipped))
                count += 1
            tree.edges[node_id] = tuple(rebuilt)
        _resync(tree)
        return count

    _assert_gate_catches(mutate)


def test_a_one_slot_layout_shift_is_caught():
    """The failure mode flat indexing reintroduced: reading a neighbour's row.

    An off-by-one in ``slot_base + bucket * num_actions`` does not crash and
    does not fall off the array. It leaks regret between unrelated infosets.
    """

    def mutate(tree) -> int:
        # Every node but the last: shifting the last one would run its final
        # row off the end of the array, which is a crash rather than the silent
        # aliasing this is meant to reproduce.
        count = 0
        for node_id in range(len(tree.nodes) - 1):
            spec = tree.node_spec[node_id]
            tree.node_spec[node_id] = (*spec[:5], spec[5] + 1, *spec[6:])
            count += 1
        return count

    _assert_gate_catches(mutate)


def test_a_wrong_chance_deal_count_is_caught():
    """Deal the wrong number of board cards and the runout stops lining up.

    This is the mutation that also moves the random stream, so it is the one
    that proves the comparison is sensitive to *when* cards are drawn and not
    only to what the arithmetic does with them afterwards.
    """

    def mutate(tree) -> int:
        count = 0
        for node_id, edges in enumerate(tree.edges):
            rebuilt = []
            for edge in edges:
                if edge.terminal is None and edge.deal == 3:
                    rebuilt.append(dataclasses.replace(edge, deal=2))
                    count += 1
                else:
                    rebuilt.append(edge)
            tree.edges[node_id] = tuple(rebuilt)
        _resync(tree)
        return count

    _assert_gate_catches(mutate, expect_stream_shift=True)


def _resync(tree) -> None:
    """Point ``node_spec`` at the mutated edge tuples.

    ``node_spec`` caches each node's edges so a visit costs one index; rewriting
    ``tree.edges`` alone would leave the traversal reading the originals and the
    mutation would silently do nothing.
    """
    tree.node_spec[:] = [
        (*spec[:7], tree.edges[node_id]) for node_id, spec in enumerate(tree.node_spec)
    ]
