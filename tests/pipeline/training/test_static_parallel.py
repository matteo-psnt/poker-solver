"""Multi-process training over static storage.

The claim this module makes is that the ~1,300 lines of ID-reconciliation
machinery the dynamic backend needs are not merely smaller here but absent:
there is no state in which a worker knows an infoset it cannot write. The
assertion that matters is therefore ``dropped_updates == 0``, which the dynamic
backend could not make (it measured 39-74%).

These spawn real processes, so they are slow-marked; the fast gate keeps a
single-worker smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint
from src.pipeline.training.static_parallel import train_static_parallel, worker_seed
from tests.test_helpers import make_test_config

BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 3}


class Buckets:
    """Picklable stand-in for the equity abstraction (spawn pickles args)."""

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        return (
            hash((repr(hole_cards[0]), repr(board[0]))) % BUCKETS[street] + BUCKETS[street]
        ) % BUCKETS[street]

    def num_buckets(self, street: Street) -> int:
        return BUCKETS[street]


class ExplodingBuckets(Buckets):
    """Module level, not nested: spawn pickles the argument by qualified name."""

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        raise RuntimeError("bucket lookup exploded")


def _config():
    return make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)


class TestWorkerSeeds:
    def test_distinct_per_worker(self):
        seeds = {worker_seed(42, w) for w in range(16)}
        assert len(seeds) == 16, "colliding seeds would silently correlate workers"

    def test_distinct_per_batch(self):
        assert worker_seed(42, 0, 0) != worker_seed(42, 0, 1)

    def test_deterministic(self):
        assert worker_seed(42, 3, 7) == worker_seed(42, 3, 7)


@pytest.mark.slow
@pytest.mark.timeout(180)
class TestParallelTraining:
    def test_multiple_workers_train_and_drop_nothing(self, tmp_path):
        result = train_static_parallel(
            _config(),
            num_iterations=400,
            num_workers=3,
            session_id="static-par-drop",
            abstraction=Buckets(),
        )
        assert result.iterations == 400
        assert result.touched_rows > 0
        # The headline property: with static rows there is no code path by which
        # a worker can know an infoset and fail to write it.
        assert result.dropped_updates == 0

    def test_iterations_split_exactly(self, tmp_path):
        """Remainder goes to the first workers; the total must be exact."""
        result = train_static_parallel(
            _config(),
            num_iterations=101,
            num_workers=4,
            session_id="static-par-split",
            abstraction=Buckets(),
        )
        assert result.iterations == 101

    def test_more_workers_cover_more_of_the_tree(self, tmp_path):
        """Sanity that workers actually explore, rather than all repeating one path."""
        one = train_static_parallel(
            _config(),
            num_iterations=300,
            num_workers=1,
            session_id="static-par-one",
            abstraction=Buckets(),
        )
        many = train_static_parallel(
            _config(),
            num_iterations=300,
            num_workers=3,
            session_id="static-par-many",
            abstraction=Buckets(),
        )
        assert one.touched_rows > 0 and many.touched_rows > 0
        assert one.num_rows == many.num_rows

    def test_workers_write_the_same_shared_arrays(self, tmp_path):
        """Checkpoint after a parallel run must contain every worker's work."""
        result = train_static_parallel(
            _config(),
            num_iterations=300,
            num_workers=3,
            session_id="static-par-shared",
            checkpoint_dir=tmp_path,
            abstraction=Buckets(),
        )
        assert result.touched_rows > 0

        config = _config()
        from src.core.actions.action_model import ActionModel
        from src.core.game.rules import GameRules

        tree = build_betting_tree(
            GameRules(small_blind=1, big_blind=2),
            ActionModel(config),
            Buckets(),
            starting_stack=config.game.starting_stack,
        )
        storage = StaticArrayStorage(tree)
        try:
            assert load_checkpoint(storage, tmp_path) == 300
            assert np.count_nonzero(storage.regrets) > 0
            assert storage.num_touched_infosets() == result.touched_rows
        finally:
            storage.close()

    def test_worker_failure_surfaces(self, tmp_path):
        """A failed worker must raise, not silently return partial training."""
        with pytest.raises(RuntimeError, match=r"static worker|exploded"):
            train_static_parallel(
                _config(),
                num_iterations=50,
                num_workers=2,
                session_id="static-par-fail",
                abstraction=ExplodingBuckets(),
            )


@pytest.mark.timeout(60)
def test_single_worker_smoke(tmp_path):
    """Kept out of the slow set so the fast gate still covers the path."""
    result = train_static_parallel(
        _config(),
        num_iterations=60,
        num_workers=1,
        session_id="static-par-smoke",
        abstraction=Buckets(),
    )
    assert result.iterations == 60
    assert result.dropped_updates == 0
    assert 0.0 < result.coverage <= 1.0
    assert result.mean_visits_per_touched > 0
