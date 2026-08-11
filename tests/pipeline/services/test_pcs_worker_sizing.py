"""How many workers a PCS node holds, when a legal turn BR multiplies the scratch.

The clamp and the log line used to describe different workers: the clamp was
sized for one kernel while ``runout_mode='turn'`` holds one per runout.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.vector import compile_tree
from src.engine.solver.vector.kernel import DTYPE
from src.pipeline.services import pcs_training
from src.pipeline.services.pcs_training import worker_footprint
from src.pipeline.training.pcs_parallel import (
    LIVE_HANDS,
    NODE_HEADROOM_FRACTION,
    ram_safe_workers,
    worker_bytes,
)
from tests.pipeline.training.test_static_parallel import Buckets
from tests.test_helpers import make_test_config

TURN = {"cfr_br": "turn_river", "runout_mode": "turn", "runouts_per_flop": 4}
FLOP = {"cfr_br": "river", "runout_mode": "flop", "runouts_per_flop": 4}


def _config(**pcs):
    return make_test_config(starting_stack=20).merge({"pcs": pcs})


def test_a_turn_best_response_holds_one_kernel_per_runout():
    assert worker_footprint(_config(**TURN))["kernels"] == 4


def test_flop_mode_rebinds_a_single_kernel():
    assert worker_footprint(_config(**FLOP))["kernels"] == 1


def test_the_turn_footprint_clamps_the_worker_count_harder():
    """The defect this pins: R=4 sized as one kernel let ~50 workers onto a
    64-core box that holds 8 — an OOM minutes into a 23-hour run."""
    rules = GameRules(1, 2)
    tree = build_betting_tree(rules, ActionModel(_config()), Buckets(), starting_stack=20)
    terminals = compile_tree(tree, rules).num_terminals
    box = 64 * 1024**3
    turn = ram_safe_workers(
        tree, terminals, shared_bytes=0, memory=box, **worker_footprint(_config(**TURN))
    )
    flop = ram_safe_workers(
        tree, terminals, shared_bytes=0, memory=box, **worker_footprint(_config(**FLOP))
    )
    assert turn < flop
    assert turn >= 1


def test_each_kernel_costs_the_same_scratch_again():
    """`turn < flop` alone would still pass if `kernels` regressed to 2, and the
    clamp would still oversubscribe. Pin the scaling itself: `kernels` holds K
    copies of the per-kernel scratch and moves nothing else, so the increments
    are equal. No magic byte counts -- the identity is what has to hold."""
    rules = GameRules(1, 2)
    tree = build_betting_tree(rules, ActionModel(_config()), Buckets(), starting_stack=20)
    terminals = compile_tree(tree, rules).num_terminals
    one, two, four = (
        worker_bytes(tree, terminals, br_streets="turn_river", runouts=4, kernels=k)
        for k in (1, 2, 4)
    )
    per_kernel = two - one
    assert per_kernel > 0
    assert four - one == 3 * per_kernel


def test_train_pcs_hands_the_kernel_count_to_the_clamp(monkeypatch, tmp_path):
    """WHERE THE DEFECT LIVED: not in the footprint, but in the call that ignored
    it. Every other test here builds `worker_footprint` itself, so all of them
    would still pass if `train_pcs` went back to letting `kernels` default to 1 --
    which is precisely the shadow-fix shape this project keeps paying for."""

    class StopError(Exception):
        pass

    seen: dict[str, object] = {}

    def record(tree, num_terminals, **kwargs):
        seen.update(kwargs)
        raise StopError

    monkeypatch.setattr(pcs_training.blueprint, "build_card_abstraction", lambda _c: Buckets())
    monkeypatch.setattr(
        pcs_training.blueprint, "resolve_card_abstraction_hash", lambda _c: "stub-abstraction"
    )
    monkeypatch.setattr(pcs_training.pcs_parallel, "ram_safe_workers", record)

    with pytest.raises(StopError):
        pcs_training.train_pcs(
            "quick_test",
            iterations=1,
            runs_dir=tmp_path,
            config_overrides={
                "pcs__cfr_br": "turn_river",
                "pcs__runout_mode": "turn",
                "pcs__runouts_per_flop": 4,
            },
        )

    assert seen.get("kernels") == 4, (
        f"the clamp was sized for {seen.get('kernels', 1)} kernel(s) while the run holds 4"
    )
    assert seen["runouts"] == 4
    assert seen["br_streets"] == "turn_river"


def test_evaluate_terminals_transients_are_counted():
    """The defect this pins: the clamp read 10.85 GB against a worker's measured
    12.17 GB at 200 bb, sized 11 onto a box that held 10, and the OOM killer
    took one — the coordinator then blocked on a chunk that had already run.

    `evaluate_terminals` peaks above the arrays it reads: `walk` holds the copy
    of one seat's showdown reaches while the rank walk builds its output. Half
    of any tree's terminals are showdowns, so that is `num_terminals * hands`.
    """
    rules = GameRules(1, 2)
    tree = build_betting_tree(rules, ActionModel(_config()), Buckets(), starting_stack=20)
    terminals = compile_tree(tree, rules).num_terminals

    walk = worker_bytes(tree, terminals, br_streets="river", showdown="walk")
    counted = 2 * (terminals // 2) * LIVE_HANDS * np.dtype(DTYPE).itemsize
    assert walk - worker_bytes(tree, 0, br_streets="river", showdown="walk") > counted

    # `matmul` stacks BOTH seats before the product, so each array is twice as
    # wide and the clamp must not size a matmul run off the walk figure.
    assert worker_bytes(tree, terminals, br_streets="river", showdown="matmul") > walk


def test_headroom_is_a_fraction_so_a_big_node_keeps_real_margin():
    """A flat 3 GB was 2.2% of a 128 GiB box and 0.5% of a 512 GB one: the
    bigger the node, the thinner the margin, which is backwards.
    """
    rules = GameRules(1, 2)
    tree = build_betting_tree(rules, ActionModel(_config()), Buckets(), starting_stack=20)
    terminals = compile_tree(tree, rules).num_terminals
    fp = worker_footprint(_config(**FLOP))

    small = ram_safe_workers(tree, terminals, shared_bytes=0, memory=128 * 1024**3, **fp)
    big = ram_safe_workers(tree, terminals, shared_bytes=0, memory=512 * 1024**3, **fp)
    per = worker_bytes(tree, terminals, **fp)

    # Both leave at least the fraction free, so neither runs at 99%.
    for workers, box in ((small, 128 * 1024**3), (big, 512 * 1024**3)):
        assert box - workers * per >= NODE_HEADROOM_FRACTION * box
