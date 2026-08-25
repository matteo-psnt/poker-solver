"""Reassembling a checkpoint into a different strategy of the same arrays.

Both variants are exact identities, not approximations, and that is the whole
reason they are worth measuring from retained rungs instead of retrained runs:
`strategy_sum` weights an iterate at add time and never rediscounts it, so a
window sum IS a difference of two rungs, and `max(regret, 0)` under the average
normaliser IS regret matching. A test that only checked "the numbers moved"
would pass on a version of either that quietly measured something else.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.numba_ops import average_strategy, regret_matching
from src.engine.solver.storage.policy_assembly import WINDOW_SHRINKAGE, assemble_policy
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint, save_checkpoint
from tests.test_helpers import make_test_config

BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}


@pytest.fixture(scope="module")
def tree() -> BettingTree:
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    return BettingTree(
        GameRules(small_blind=1, big_blind=2),
        ActionModel(config),
        starting_stack=20,
        buckets_per_street=BUCKETS,
    )


def _ladder(tree: BettingTree, path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A run with two retained rungs. Returns (early sum, late sum, late regrets)."""
    rng = np.random.default_rng(7)
    storage = StaticArrayStorage(tree)
    early = rng.random(storage.strategy_sum.shape).astype(storage.strategy_sum.dtype)
    storage.strategy_sum[:] = early
    storage.visited[:] = 1
    save_checkpoint(storage, path, 100, retain_every=100)

    # A window's worth of further training: strategy mass only ever accumulates.
    added = rng.random(storage.strategy_sum.shape).astype(storage.strategy_sum.dtype)
    # Some rows are never touched again, which is the case the shrinkage covers.
    added[: added.shape[0] // 3] = 0.0
    late = early + added
    storage.strategy_sum[:] = late
    storage.regrets[:] = rng.normal(size=storage.regrets.shape).astype(storage.regrets.dtype)
    regrets = storage.regrets.copy()
    save_checkpoint(storage, path, 200, retain_every=100)
    return early, late, regrets


def test_current_iterate_is_regret_matching(tree, tmp_path):
    _, _, regrets = _ladder(tree, tmp_path)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    record = assemble_policy(storage, tmp_path, iterate="current")

    assert record == {"policy_iterate": "current"}
    for node_id in range(len(tree)):
        for bucket in range(int(tree.buckets_per_node[node_id])):
            start, stop = tree.slots(node_id, bucket)
            np.testing.assert_allclose(
                average_strategy(storage.strategy_sum[start:stop]),
                regret_matching(regrets[start:stop]),
                rtol=1e-6,
            )


def test_window_is_the_difference_of_two_rungs(tree, tmp_path):
    early, late, _ = _ladder(tree, tmp_path)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    record = assemble_policy(storage, tmp_path, window_from=100)

    assert record["avg_window_from"] == 100
    np.testing.assert_allclose(
        storage.strategy_sum,
        np.maximum(late - early, 0.0) + WINDOW_SHRINKAGE * early,
        rtol=1e-5,
    )
    # A third of the table saw nothing in the window, so it keeps the average.
    assert record["avg_window_empty_slot_fraction"] == pytest.approx(1 / 3, abs=0.02)


def test_window_leaves_an_untouched_row_on_the_full_average(tree, tmp_path):
    early, _, _ = _ladder(tree, tmp_path)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    assemble_policy(storage, tmp_path, window_from=100)
    start, stop = tree.slots(0, 0)
    np.testing.assert_allclose(
        average_strategy(storage.strategy_sum[start:stop]),
        average_strategy(early[start:stop]),
        rtol=1e-5,
    )


def test_a_window_of_the_current_iterate_is_refused(tree, tmp_path):
    _ladder(tree, tmp_path)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    with pytest.raises(ValueError, match="no averaging weight"):
        assemble_policy(storage, tmp_path, iterate="current", window_from=100)


def test_the_default_leaves_the_average_alone(tree, tmp_path):
    _, late, _ = _ladder(tree, tmp_path)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    assert assemble_policy(storage, tmp_path) == {}
    np.testing.assert_array_equal(storage.strategy_sum, late)


def _ladder_of(tree: BettingTree, path, rungs: tuple[int, ...]) -> list[np.ndarray]:
    """A run retaining every rung in ``rungs``. Returns each rung's strategy sum."""
    rng = np.random.default_rng(11)
    storage = StaticArrayStorage(tree)
    storage.visited[:] = 1
    sums = []
    for rung in rungs:
        storage.strategy_sum += rng.random(storage.strategy_sum.shape).astype(
            storage.strategy_sum.dtype
        )
        sums.append(storage.strategy_sum.copy())
        save_checkpoint(storage, path, rung, retain_every=1)
    return sums


def test_gamma_reweighting_is_the_band_weighted_combination(tree, tmp_path):
    rungs = (100, 200, 300)
    sums = _ladder_of(tree, tmp_path, rungs)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    record = assemble_policy(
        storage, tmp_path, avg_gamma=0.0, source_gamma=2.0, loaded_iteration=300
    )

    assert record == {"avg_gamma": 0.0, "avg_gamma_rungs": 3}
    edges = (0, *rungs)

    def mass(exponent, low, high):
        return (high ** (exponent + 1) - low ** (exponent + 1)) / (exponent + 1)

    coefficients = [
        mass(0.0, edges[k], edges[k + 1]) / mass(2.0, edges[k], edges[k + 1]) for k in range(3)
    ]
    scale = max(coefficients)
    windows = [sums[0], sums[1] - sums[0], sums[2] - sums[1]]
    expected = sum((c / scale) * window for c, window in zip(coefficients, windows, strict=True))
    np.testing.assert_allclose(storage.strategy_sum, expected, rtol=1e-4, atol=1e-6)


def test_gamma_matching_the_source_leaves_the_average_alone(tree, tmp_path):
    sums = _ladder_of(tree, tmp_path, (100, 200, 300))
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    assemble_policy(storage, tmp_path, avg_gamma=2.0, source_gamma=2.0, loaded_iteration=300)
    # Every band's coefficient is 1, so the ladder recombines into the rung.
    np.testing.assert_allclose(storage.strategy_sum, sums[-1], rtol=1e-4, atol=1e-6)


def test_gamma_needs_the_ladder(tree, tmp_path):
    _ladder_of(tree, tmp_path, (100,))
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    with pytest.raises(ValueError, match="retained ladder"):
        assemble_policy(storage, tmp_path, avg_gamma=0.0, source_gamma=2.0, loaded_iteration=100)
