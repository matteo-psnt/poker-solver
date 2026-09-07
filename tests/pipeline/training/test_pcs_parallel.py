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
from src.core.game.state import Street
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

    def test_no_two_runouts_of_one_iteration_are_the_same_board(self):
        """A turn and river are a SET, so drawing them independently repeats one.

        Measured: at K=4 a repeat lands about every 200 iterations, which killed
        a 300-iteration CFR-BR run 13 iterations in. Two copies of one board are
        one observation counted twice — tolerable for plain PCS, and impossible
        for a best response, which must choose ONE action across boards it
        cannot tell apart and so cannot be run on them separately.
        """
        rng = np.random.default_rng(0)
        for _ in range(2000):
            boards = pcs_parallel.sample_boards(rng, 4)
            keys = {frozenset(int(card) for card in board[3:]) for board in boards}
            assert len(keys) == 4

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


class TestSharedTurnSampling:
    """What a LEGAL turn best response needs from the sampler.

    With runouts that share only the flop, every turn is alone in its partition
    and a per-hand argmax there chooses using a river that has not been dealt.
    Sharing the turn puts them in one partition, so the maximisation is a real
    sample of the turn's continuation.
    """

    def test_every_runout_shares_the_flop_and_the_turn(self):
        rng = np.random.default_rng(5)
        for _ in range(200):
            boards = pcs_parallel.sample_boards_shared_turn(rng, 4)
            assert len(boards) == 4
            assert all(len(set(b.tolist())) == 5 for b in boards)
            for b in boards:
                assert np.array_equal(b[:4], boards[0][:4])

    def test_the_rivers_are_distinct(self):
        """Two identical boards are one observation counted twice."""
        rng = np.random.default_rng(6)
        for _ in range(200):
            boards = pcs_parallel.sample_boards_shared_turn(rng, 6)
            assert len({int(b[4]) for b in boards}) == 6

    def test_the_turn_partition_is_one_group_so_the_argmax_is_joint(self):
        """The property the whole sampler exists for, checked through the driver."""
        rng = np.random.default_rng(7)
        boards = pcs_parallel.sample_boards_shared_turn(rng, 4)
        visible = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
        for street, seen in visible.items():
            groups = {tuple(sorted(int(c) for c in b[:seen])) for b in boards}
            expected = 1 if street in (Street.FLOP, Street.TURN) else len(boards)
            assert len(groups) == expected, street

    def test_the_flop_sampler_does_not_share_the_turn(self):
        """The contrast that makes the turn BR illegal under the old sampler."""
        rng = np.random.default_rng(8)
        turns = set()
        for _ in range(50):
            boards = pcs_parallel.sample_boards(rng, 4)
            turns.add(len({tuple(sorted(int(c) for c in b[:4])) for b in boards}))
        assert max(turns) > 1


class TestMultiKernelSizing:
    """A legal turn BR holds one kernel per runout, and the sizer has to know.

    The joint maximisation needs the runouts' values at the same time, so the
    ~3.7 GB of hand-space scratch multiplies. Sizing a node as if one kernel
    were live would over-subscribe it and the run dies on the OOM killer rather
    than refusing at submit.
    """

    def _tree(self):
        config = make_test_config(seed=1, starting_stack=20)
        return build_betting_tree(
            GameRules(config.game.small_blind, config.game.big_blind),
            ActionModel(config),
            Buckets(),
            starting_stack=20,
        )

    def test_scratch_scales_linearly_with_the_kernel_count(self):
        """Exact, so it does not depend on the fixture tree's size.

        Only the per-kernel terms move; the interpreter overhead and the one
        chunk of block temporaries are fixed however many kernels are live.
        """
        tree = self._tree()
        sizes = {
            k: pcs_parallel.worker_bytes(tree, 100, br_streets="river", kernels=k)
            for k in (1, 2, 4)
        }
        step = sizes[2] - sizes[1]
        assert step > 0
        assert sizes[4] - sizes[1] == 3 * step

    def test_more_kernels_means_fewer_workers(self):
        tree = self._tree()
        one = pcs_parallel.ram_safe_workers(
            tree, 100, shared_bytes=0, memory=64 * 1024**3, br_streets="river", kernels=1
        )
        four = pcs_parallel.ram_safe_workers(
            tree, 100, shared_bytes=0, memory=64 * 1024**3, br_streets="river", kernels=4
        )
        assert four < one
        assert four >= 1
