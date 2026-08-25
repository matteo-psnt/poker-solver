"""Reassembling a checkpoint into a different strategy of the same arrays.

Both variants are exact identities, not approximations, and that is the whole
reason they are worth measuring from retained rungs instead of retrained runs:
`strategy_sum` weights an iterate at add time and never rediscounts it, so a
window sum IS a difference of two rungs, and `max(regret, 0)` under the average
normaliser IS regret matching. A test that only checked "the numbers moved"
would pass on a version of either that quietly measured something else.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.numba_ops import average_strategy, regret_matching
from src.engine.solver.storage.policy_assembly import (
    WINDOW_SHRINKAGE,
    _window_coefficients,
    assemble_policy,
    source_gamma_of,
)
from src.engine.solver.storage.static_array import _ARRAYS, StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import (
    AbstractionMismatchError,
    _legacy_index_maps,
    load_checkpoint,
    save_checkpoint,
)
from src.shared.config.schema import SolverConfig
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


def test_a_window_reads_a_node_major_rung_in_the_new_order(tree, tmp_path):
    """MEASURED: every published run predates the bucket-major layout, so the
    window base is a v1 snapshot. Reading it without the permutation combined
    two different orderings and the task died on the fingerprint instead."""
    early, late, _ = _ladder(tree, tmp_path)
    row_source, slot_source = _legacy_index_maps(tree)
    for rung in (100, 200):
        root = zarr.open(zarr.DirectoryStore(str(tmp_path / f"static-{rung}.zarr")), mode="r+")
        for name in _ARRAYS:
            gather = slot_source if name in ("regrets", "strategy_sum") else row_source
            current = root[name][:]
            scattered = np.empty_like(current)
            scattered[gather] = current
            root[name][:] = scattered
        root.attrs["fingerprint"] = tree.legacy_fingerprint()
    manifest = tmp_path / "STATIC_CHECKPOINT.json"
    raw = json.loads(manifest.read_text())
    raw["fingerprint"] = tree.legacy_fingerprint()
    manifest.write_text(json.dumps(raw))

    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    assemble_policy(storage, tmp_path, window_from=100)

    np.testing.assert_allclose(
        storage.strategy_sum,
        np.maximum(late - early, 0.0) + WINDOW_SHRINKAGE * early,
        rtol=1e-5,
    )


def test_gamma_reweighting_survives_production_magnitudes(tree, tmp_path):
    """The float32 table holds `sum_t t^2 * reach`, which at 600M iterations is
    ~1e20 per slot while the band coefficients span 1 .. 2.5e-3. A test whose
    values are all O(1) cannot see whether that combination still resolves, and
    the combination is what the whole reweighting rests on."""
    rungs = tuple(i * 50_000_000 for i in range(1, 13))
    rng = np.random.default_rng(3)
    storage = StaticArrayStorage(tree)
    storage.visited[:] = 1
    reference = np.zeros(storage.strategy_sum.shape, dtype=np.float64)
    bands = []
    for index, rung in enumerate(rungs):
        low = rungs[index - 1] if index else 0
        # A band's real contribution: its iterations' t^2 weight times a reach.
        band = ((rung**3 - low**3) / 3.0) * rng.random(storage.strategy_sum.shape)
        bands.append(band)
        reference += band
        storage.strategy_sum[:] = reference.astype(np.float32)
        save_checkpoint(storage, tmp_path, rung, retain_every=50_000_000)

    fresh = StaticArrayStorage(tree)
    load_checkpoint(fresh, tmp_path)
    assemble_policy(fresh, tmp_path, avg_gamma=0.0, source_gamma=2.0, loaded_iteration=rungs[-1])

    coefficients = _window_coefficients(list(rungs), 0.0, 2.0)
    expected = sum(c * band for c, band in zip(coefficients, bands, strict=True))
    # Row-normalised at read time, so only the ratio within a row matters.
    for node_id in range(len(tree)):
        for bucket in range(int(tree.buckets_per_node[node_id])):
            start, stop = tree.slots(node_id, bucket)
            np.testing.assert_allclose(
                average_strategy(fresh.strategy_sum[start:stop]),
                expected[start:stop] / expected[start:stop].sum(),
                rtol=1e-4,
            )


@pytest.mark.parametrize(
    ("weighting", "gamma", "expected"),
    [("dcfr", 2.0, 2.0), ("dcfr", 1.0, 1.0), ("linear", 2.0, 1.0), ("none", 2.0, 0.0)],
)
def test_the_source_exponent_follows_the_weighting_not_the_gamma_field(weighting, gamma, expected):
    """A linear run adds `t^1` and never reads `dcfr_gamma`; reweighting it as
    if it were gamma=2 would correct an exponent it never used. The PCS
    flagship is a linear run, so this is the reachable case, not a hypothetical."""
    solver = SolverConfig(iteration_weighting=weighting, dcfr_gamma=gamma)
    assert source_gamma_of(solver) == expected


def test_gamma_over_a_window_is_the_windows_bands_only(tree, tmp_path):
    """A fixed-WIDTH uniform window is the one comparison that holds averaging
    noise constant while the endpoint moves, so it separates "the solver is
    still learning" from "averaging more iterates shrinks the average's
    variance". Both make the all-history gamma=0 curve fall."""
    rungs = (100, 200, 300, 400)
    sums = _ladder_of(tree, tmp_path, rungs)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    record = assemble_policy(
        storage, tmp_path, avg_gamma=0.0, source_gamma=2.0, loaded_iteration=400, window_from=200
    )

    assert record["avg_gamma_rungs"] == 2
    assert record["avg_window_from"] == 200
    coefficients = _window_coefficients([300, 400], 0.0, 2.0, 200)
    bands = [sums[2] - sums[1], sums[3] - sums[2]]
    expected = sum(c * b for c, b in zip(coefficients, bands, strict=True))
    expected = np.maximum(expected, 0.0) + WINDOW_SHRINKAGE * sums[1]
    np.testing.assert_allclose(storage.strategy_sum, expected, rtol=1e-4, atol=1e-6)


def test_a_window_at_the_source_gamma_matches_the_plain_window(tree, tmp_path):
    """Target == source over a window must reduce to the plain subtraction, or
    the two window arms are not on the same footing."""
    sums = _ladder_of(tree, tmp_path, (100, 200, 300))
    a = StaticArrayStorage(tree)
    load_checkpoint(a, tmp_path)
    assemble_policy(
        a, tmp_path, avg_gamma=2.0, source_gamma=2.0, loaded_iteration=300, window_from=100
    )
    b = StaticArrayStorage(tree)
    load_checkpoint(b, tmp_path)
    assemble_policy(b, tmp_path, window_from=100)
    np.testing.assert_allclose(a.strategy_sum, b.strategy_sum, rtol=1e-4, atol=1e-6)
    assert sums[-1] is not None


def _run_with(tree: BettingTree, path, seed: int, abstraction: str | None = "abs-1"):
    """A one-rung run whose strategy_sum is distinctive to ``seed``."""
    rng = np.random.default_rng(seed)
    storage = StaticArrayStorage(tree)
    storage.visited[:] = 1
    storage.strategy_sum[:] = rng.random(storage.strategy_sum.shape).astype(
        storage.strategy_sum.dtype
    )
    save_checkpoint(storage, path, 800, abstraction_id=abstraction)
    return storage.strategy_sum.copy()


def test_mixing_a_run_with_itself_is_the_exact_identity(tree, tmp_path):
    """THE guard. Rows are normalised per row at read time, so a mixture that
    got the scaling wrong would still look like a plausible strategy -- this is
    the only cheap check that says the operation is the one intended."""
    _run_with(tree, tmp_path, seed=5)
    for weight in (0.0, 0.25, 0.5, 1.0):
        storage = StaticArrayStorage(tree)
        load_checkpoint(storage, tmp_path)
        before = storage.strategy_sum.copy()
        assemble_policy(storage, tmp_path, mix_run=tmp_path, mix_at=800, mix_weight=weight)
        for node_id in range(len(tree)):
            for bucket in range(int(tree.buckets_per_node[node_id])):
                start, stop = tree.slots(node_id, bucket)
                np.testing.assert_allclose(
                    average_strategy(storage.strategy_sum[start:stop]),
                    average_strategy(before[start:stop]),
                    rtol=1e-5,
                    err_msg=f"weight={weight}",
                )


def test_mixing_blends_the_two_runs_by_total_mass(tree, tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a = _run_with(tree, a_dir, seed=5)
    b = _run_with(tree, b_dir, seed=9)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, a_dir)
    record = assemble_policy(storage, a_dir, mix_run=b_dir, mix_at=800, mix_weight=0.5)

    assert record == {"mix_run": "b", "mix_at": 800, "mix_weight": 0.5}
    expected = 0.5 * a / a.sum(dtype=np.float64) + 0.5 * b / b.sum(dtype=np.float64)
    np.testing.assert_allclose(storage.strategy_sum, expected, rtol=1e-4, atol=1e-12)


def test_mixing_across_a_different_bucket_assignment_is_refused(tree, tmp_path):
    """Two runs can share a tree, a layout and bucket COUNTS while row `i` holds
    a different hand in each, and nothing about the arrays would reveal it."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _run_with(tree, a_dir, seed=5, abstraction="abs-1")
    _run_with(tree, b_dir, seed=9, abstraction="abs-2")
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, a_dir)
    with pytest.raises(AbstractionMismatchError, match="different hands"):
        assemble_policy(storage, a_dir, mix_run=b_dir, mix_at=800)


def test_a_mixture_is_scored_plain(tree, tmp_path):
    _run_with(tree, tmp_path, seed=5)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, tmp_path)
    with pytest.raises(ValueError, match="scored plain"):
        assemble_policy(storage, tmp_path, mix_run=tmp_path, mix_at=800, avg_gamma=0.0)


def test_mixing_runs_that_never_recorded_an_abstraction_is_allowed(tree, tmp_path):
    """MEASURED on the pool: the first real mixture was refused because NO
    ordinary training run records `abstraction_id` in its checkpoint manifest
    (`static_parallel` never passes it), so a manifest-only guard rejects every
    real pair. The caller verifies the abstraction off the run metadata."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a = _run_with(tree, a_dir, seed=5, abstraction=None)
    b = _run_with(tree, b_dir, seed=9, abstraction=None)
    storage = StaticArrayStorage(tree)
    load_checkpoint(storage, a_dir)
    record = assemble_policy(storage, a_dir, mix_run=b_dir, mix_at=800, mix_weight=0.5)

    assert record["mix_run"] == "b"
    expected = 0.5 * a / a.sum(dtype=np.float64) + 0.5 * b / b.sum(dtype=np.float64)
    np.testing.assert_allclose(storage.strategy_sum, expected, rtol=1e-4, atol=1e-12)
