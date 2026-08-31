"""How many workers a PCS node holds, when a legal turn BR multiplies the scratch.

The clamp and the log line used to describe different workers: the clamp was
sized for one kernel while ``runout_mode='turn'`` holds one per runout.
"""

from __future__ import annotations

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.vector import compile_tree
from src.pipeline.services.pcs_training import worker_footprint
from src.pipeline.training.pcs_parallel import ram_safe_workers
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
