"""Public chance sampling through the multi-process coordinator.

What the coordinator has to deliver for this kernel is the same as for the
scalar one -- an absolute target, a ladder, a resume that keeps counting --
plus one thing the scalar path never needed: ``visited`` derived from the
table, since nothing here touches rows one at a time.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import StaticCheckpointManifest, load_checkpoint
from src.engine.solver.vector import compile_tree
from src.pipeline.training import pcs_parallel
from src.pipeline.training.static_parallel import train_static_parallel
from tests.pipeline.training.test_static_parallel import Buckets, _config, session
from tests.test_helpers import make_test_config


def _train(tmp_path, *, target: int, workers: int, name: str, **overrides):
    config = _config().merge({"pcs": overrides}) if overrides else _config()
    return train_static_parallel(
        config,
        num_iterations=target,
        num_workers=workers,
        session_id=session(name),
        checkpoint_dir=tmp_path / "run",
        abstraction=Buckets(),
        checkpoint_every=4,
        checkpoint_retain_every=4,
        resume=True,
        worker=pcs_parallel.pcs_worker,
        before_checkpoint=pcs_parallel.mark_visited_from_strategy,
    )


class TestSampling:
    def test_runouts_share_the_flop_and_differ_below_it(self):
        rng = np.random.default_rng(3)
        boards = pcs_parallel.sample_boards(rng, 3)
        assert len(boards) == 3
        assert all(len(set(board.tolist())) == 5 for board in boards)
        assert all(np.array_equal(board[:3], boards[0][:3]) for board in boards)
        assert len({tuple(sorted(board[3:].tolist())) for board in boards}) > 1

    def test_every_full_board_leaves_the_same_live_hand_count(self):
        assert pcs_parallel.LIVE_HANDS == 47 * 46 // 2


class TestVisitedDerivation:
    def test_marks_exactly_the_rows_holding_mass(self):
        """Against the tree's own accessors, not a rebuilt row order.

        This function once rebuilt the row boundaries with ``np.repeat`` --
        node-major, the retired layout -- so it summed mass over the wrong
        slot ranges and flagged the wrong rows, while its only test asserted
        ``touched_rows > 0``, which the scramble does not break.
        """
        config = make_test_config(seed=1, starting_stack=20)
        tree = build_betting_tree(
            GameRules(config.game.small_blind, config.game.big_blind),
            ActionModel(config),
            Buckets(),
            starting_stack=config.game.starting_stack,
        )
        storage = StaticArrayStorage(tree)
        try:
            marked = [(5, 1), (len(tree.nodes) // 2, 0)]
            for node_id, bucket in marked:
                lo, _hi = tree.slots(node_id, bucket)
                storage.strategy_sum[lo] = 1.0
            pcs_parallel.mark_visited_from_strategy(storage)
            expected = sorted(tree.row(n, b) for n, b in marked)
            assert sorted(np.flatnonzero(storage.visited).tolist()) == expected
        finally:
            storage.close()


class TestWorkerSizing:
    def test_the_count_is_clamped_by_memory_not_cores(self):
        config = make_test_config(seed=1, starting_stack=20)
        tree = build_betting_tree(
            GameRules(1, 2), ActionModel(config), Buckets(), starting_stack=20
        )
        terminals = compile_tree(tree, GameRules(1, 2)).num_terminals
        one = pcs_parallel.worker_bytes(tree, terminals)
        room = pcs_parallel.NODE_HEADROOM_BYTES + 3 * one + one // 2
        assert pcs_parallel.ram_safe_workers(tree, terminals, shared_bytes=0, memory=room) == 3
        assert pcs_parallel.ram_safe_workers(tree, terminals, shared_bytes=0, memory=0) == 1


class TestCoordinator:
    @pytest.mark.timeout(120)
    def test_one_worker_writes_a_ladder_with_visited_rows(self, tmp_path):
        result = _train(tmp_path, target=8, workers=1, name="pcs-one")
        assert result.iterations == 8
        assert result.dropped_updates == 0
        manifest = StaticCheckpointManifest.read(tmp_path / "run")
        assert manifest is not None
        assert manifest.iteration == 8
        assert manifest.ladder() == [4, 8]
        assert result.touched_rows > 0, "visited must be derived, or evaluation reads an empty run"

    @pytest.mark.timeout(120)
    def test_a_resume_keeps_counting_from_the_rung(self, tmp_path):
        _train(tmp_path, target=4, workers=1, name="pcs-resume")
        tree = build_betting_tree(
            GameRules(1, 2), ActionModel(_config()), Buckets(), starting_stack=20
        )
        storage = StaticArrayStorage(tree)
        assert load_checkpoint(storage, tmp_path / "run") == 4
        before = storage.strategy_sum.copy()

        result = _train(tmp_path, target=8, workers=1, name="pcs-resume")
        assert result.iterations == 8
        assert load_checkpoint(storage, tmp_path / "run") == 8
        assert not np.array_equal(storage.strategy_sum, before)
        assert np.all(storage.strategy_sum[before > 0] >= before[before > 0])

    @pytest.mark.slow
    @pytest.mark.timeout(240)
    def test_two_workers_tile_the_target_exactly(self, tmp_path):
        result = _train(tmp_path, target=8, workers=2, name="pcs-two", runouts_per_flop=2)
        assert result.iterations == 8
        manifest = StaticCheckpointManifest.read(tmp_path / "run")
        assert manifest is not None
        assert manifest.iteration == 8


@pytest.mark.skipif(os.cpu_count() is None, reason="no core count to size against")
def test_node_memory_is_readable():
    assert pcs_parallel.node_memory_bytes() > 0
