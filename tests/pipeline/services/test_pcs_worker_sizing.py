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
from src.pipeline.abstraction.vector_universe import build_hand_context
from src.pipeline.services import pcs_training
from src.pipeline.services.pcs_training import worker_footprint
from src.pipeline.training.pcs_parallel import (
    LIVE_HANDS,
    NODE_HEADROOM_FRACTION,
    ram_safe_workers,
    worker_bytes,
)
from tests.memory_record import MemoryRecord
from tests.pipeline.training.test_static_parallel import Buckets
from tests.test_helpers import make_test_config

TURN = {"cfr_br": "turn_river", "runout_mode": "turn", "runouts_per_flop": 4}
FLOP = {"cfr_br": "river", "runout_mode": "flop", "runouts_per_flop": 4}


def _config(**pcs):
    return make_test_config(starting_stack=20).merge({"pcs": pcs})


def test_a_turn_best_response_holds_one_kernel_per_runout():
    assert worker_footprint(_config(**TURN))["kernels"] == 4


def test_plain_pcs_on_shared_turns_rebinds_a_single_kernel():
    """MEASURED 09-25 on production (cfr_br off, turn runouts): the clamp sized a
    worker at 16.99 GB for 4 kernels, workers logged peak RSS 7.59 GB, and the
    D64 ran 7 workers on 64 vCPUs. Only a best response's argmax is joint."""
    assert worker_footprint(_config(**{**TURN, "cfr_br": "off"}))["kernels"] == 1


def test_flop_mode_rebinds_a_single_kernel():
    assert worker_footprint(_config(**FLOP))["kernels"] == 1


def test_the_turn_footprint_clamps_the_worker_count_harder():
    """The defect this pins: R=4 sized as one kernel let ~50 workers onto a
    64-core box that holds 8 — an OOM minutes into a 23-hour run."""
    rules = GameRules(1, 2)
    tree = build_betting_tree(rules, ActionModel(_config()), Buckets(), starting_stack=20)
    compiled = compile_tree(tree, rules)
    box = 64 * 1024**3
    turn = ram_safe_workers(
        compiled, shared_bytes=0, memory=box, **worker_footprint(_config(**TURN))
    )
    flop = ram_safe_workers(
        compiled, shared_bytes=0, memory=box, **worker_footprint(_config(**FLOP))
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
    compiled = compile_tree(tree, rules)
    one, two, four = (
        worker_bytes(compiled, br_streets="turn_river", runouts=4, kernels=k) for k in (1, 2, 4)
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

    def capture(compiled, **kwargs):
        seen.update(kwargs)
        raise StopError

    monkeypatch.setattr(
        pcs_training.blueprint, "build_card_abstraction", lambda _c, **_kw: Buckets()
    )
    monkeypatch.setattr(
        pcs_training.blueprint, "resolve_card_abstraction_hash", lambda _c: "stub-abstraction"
    )
    monkeypatch.setattr(pcs_training.pcs_parallel, "ram_safe_workers", capture)

    with pytest.raises(StopError):
        pcs_training.train_pcs(
            "quick_test",
            iterations=1,
            runs_dir=tmp_path,
            # A tracker with no sink records nowhere and refuses; the clamp is
            # reached well after the run is opened.
            sink=MemoryRecord(),
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


def test_headroom_is_a_fraction_so_a_big_node_keeps_real_margin():
    """A flat 3 GB was 2.2% of a 128 GiB box and 0.5% of a 512 GB one: the
    bigger the node, the thinner the margin, which is backwards.
    """
    rules = GameRules(1, 2)
    tree = build_betting_tree(rules, ActionModel(_config()), Buckets(), starting_stack=20)
    compiled = compile_tree(tree, rules)
    fp = worker_footprint(_config(**FLOP))

    small = ram_safe_workers(compiled, shared_bytes=0, memory=128 * 1024**3, **fp)
    big = ram_safe_workers(compiled, shared_bytes=0, memory=512 * 1024**3, **fp)
    per = worker_bytes(compiled, **fp)

    # Both leave at least the fraction free, so neither runs at 99%.
    for workers, box in ((small, 128 * 1024**3), (big, 512 * 1024**3)):
        assert box - workers * per >= NODE_HEADROOM_FRACTION * box


def test_the_clamp_sizes_the_ring_the_kernel_actually_allocates():
    """The clamp's node term must equal `VectorCFR`'s real reach+value bytes.

    These are two different modules reading one tree, and a divergence is
    silent: the clamp hands out more workers than the node holds and the OOM
    killer takes one, which is exactly what cost a 200 bb probe its whole run.
    Measured against the arrays themselves rather than against a second copy of
    the formula, so re-deriving the shape in either place fails here.
    """
    from src.engine.solver.vector.kernel import VectorCFR

    rules = GameRules(1, 2)
    config = _config()
    tree = build_betting_tree(rules, ActionModel(config), Buckets(), starting_stack=20)
    compiled = compile_tree(tree, rules)
    context = build_hand_context(np.arange(5), Buckets())
    kernel = VectorCFR(compiled, context)

    allocated = kernel.reach.nbytes + kernel.value.nbytes
    per_hand = 4 * context.num_hands * np.dtype(DTYPE).itemsize
    assert allocated == 2 * compiled.widest_level * per_hand


def test_the_clamp_sizes_the_terminals_the_kernel_actually_allocates():
    """Terminals hold one value per seat and hand, nothing else, and the clamp says so.

    Holding a whole-tree RANGE per terminal as well cost 1.24 GB a worker at
    production, plus a 0.62 GB spike to value them in one batch. Measured
    against the kernel's own arrays, so a range array coming back fails here.
    """
    from src.engine.solver.vector.kernel import VectorCFR

    rules = GameRules(1, 2)
    tree = build_betting_tree(rules, ActionModel(_config()), Buckets(), starting_stack=20)
    compiled = compile_tree(tree, rules)
    kernel = VectorCFR(compiled, build_hand_context(np.arange(5), Buckets()))

    terminal_arrays = sum(
        array.nbytes
        for name, array in vars(kernel).items()
        if name.startswith("terminal_") and isinstance(array, np.ndarray)
    )
    assert terminal_arrays == 2 * compiled.num_terminals * LIVE_HANDS * np.dtype(DTYPE).itemsize
