"""Tests for per-(worker, batch) seed derivation."""

from src.pipeline.training.parallel_worker import _compute_seed

# Grid size of a realistic run: 8 workers x 125 batches, i.e. 1M iterations at the
# default iterations_per_worker=1000. Distinctness must hold over the whole grid a
# real run visits, not just a handful of pairs.
_WORKERS = 8
_BATCHES = 125


def test_seed_is_deterministic():
    assert _compute_seed(42, 3, 7) == _compute_seed(42, 3, 7)


def test_seed_grid_has_no_collisions():
    """Every (worker, batch) pair must get its own stream.

    Two pairs sharing a seed draw correlated MCCFR samples, which no downstream
    assertion would catch — the training result just gets quietly worse.
    """
    grid = {(w, b): _compute_seed(42, w, b) for w in range(_WORKERS) for b in range(_BATCHES)}
    assert len(set(grid.values())) == len(grid)


def test_seed_grid_distinct_across_base_seeds():
    """A different base seed must move the whole grid, not overlap it."""
    grid_a = {_compute_seed(42, w, b) for w in range(_WORKERS) for b in range(_BATCHES)}
    grid_b = {_compute_seed(43, w, b) for w in range(_WORKERS) for b in range(_BATCHES)}
    assert not (grid_a & grid_b)


def test_seed_fits_numpy_legacy_seeding_range():
    """``_compute_seed`` feeds ``np.random.seed``, which rejects >= 2**32."""
    for w in range(_WORKERS):
        for b in (0, 1, _BATCHES - 1):
            seed = _compute_seed(2**31 - 1, w, b)
            assert 0 <= seed < 2**32
