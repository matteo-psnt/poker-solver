"""Hogwild public chance sampling over the shared static table.

The coordinator is :func:`static_parallel.train_static_parallel` unchanged --
chunks, absolute iteration indices, per-worker counters, a checkpoint per
chunk. What differs is the worker: one :class:`PublicChanceSamplingCFR` per
process, each drawing its own IID boards and writing every live hand's update
for that board straight into the shared ``regrets``/``strategy_sum``.

Per-worker memory is the kernel's hand-space scratch, ~2.5-3.5 GB at the
production tree, and it is PRIVATE -- only the table is shared -- so the
worker count is a RAM question before it is a CPU one. :func:`ram_safe_workers`
sizes it from the tree and the node.
"""

from __future__ import annotations

import logging
import os
import resource
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.game.rules import GameRules
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.vector import compile_tree
from src.engine.solver.vector.cfr_br import BR_REGIONS, CFRBestResponse, TrunkLayout
from src.engine.solver.vector.hand_context import enumerate_live_hands
from src.engine.solver.vector.kernel import DTYPE, MAX_BLOCK_ELEMENTS
from src.engine.solver.vector.pcs import PublicChanceSamplingCFR
from src.pipeline.abstraction.vector_universe import build_hand_context
from src.pipeline.training.static_parallel import _build_local, worker_seed
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.engine.solver.betting_tree import BettingTree
    from src.engine.solver.protocols import BucketingStrategy
    from src.engine.solver.vector.hand_context import HandContext
    from src.shared.config import Config

logger = logging.getLogger(__name__)

NUM_CARDS = 52
BOARD_CARDS = 5
# Live hands on any full board: C(47, 2). Fixes every scratch shape.
LIVE_HANDS = int(enumerate_live_hands(np.arange(BOARD_CARDS)).shape[0])
# What a worker holds beyond its hand-space arrays: the tree and its compiled
# form, the bucketer's caches, numba, the interpreter and one pass's
# temporaries. Measured 08-23 on a D16als_v6 at the production tree: peak RSS
# 5.74 GB per worker, of which ~0.8 GB is the mmapped abstraction's shared
# pages; six workers ran on the 32 GiB node. Revise from the task log's
# "peak RSS" line, not from here.
WORKER_OVERHEAD_BYTES = 1_600_000_000
# The coordinator, the shared table's page cache and the OS.
NODE_HEADROOM_BYTES = 3_000_000_000


def sample_boards(rng: np.random.Generator, runouts: int) -> list[np.ndarray]:
    """``runouts`` DISTINCT full boards for one iteration, all beneath one flop.

    One runout is a uniform five-card draw -- the real deal distribution, with
    no canonical-class table to get wrong. Several share the first draw's flop
    and re-draw turn and river, which is what makes their increments one
    iteration's joint sample rather than K independent ones.

    Distinct because a turn and river are a SET: drawing them independently
    repeats a runout about once per 200 iterations at K=4, and two copies of one
    board are one observation counted twice. Harmless double-weighting for plain
    PCS, fatal for CFR-BR, whose best response must choose one action across
    boards it cannot tell apart and so cannot be run per board.
    """
    first = rng.choice(NUM_CARDS, BOARD_CARDS, replace=False)
    if runouts == 1:
        return [first]
    flop = first[:3]
    rest = np.setdiff1d(np.arange(NUM_CARDS), flop)
    boards = [first]
    seen = {frozenset(int(card) for card in first[3:])}
    while len(boards) < runouts:
        pair = rng.choice(rest, 2, replace=False)
        key = frozenset(int(card) for card in pair)
        if key in seen:
            continue
        seen.add(key)
        boards.append(np.concatenate([flop, pair]))
    return boards


def iteration_contexts(
    rng: np.random.Generator, abstraction: BucketingStrategy, runouts: int
) -> tuple[list[HandContext], list[np.ndarray]]:
    """Contexts and the boards behind them; CFR-BR needs the cards themselves."""
    boards = sample_boards(rng, runouts)
    return [build_hand_context(board, abstraction) for board in boards], boards


TRUNK_ARRAY = "trunk_regrets"


def trunk_arrays(config: Config, tree: BettingTree) -> dict[str, int]:
    """The extra shared arrays a run needs -- CFR-BR's opponent table, or none."""
    if config.pcs.cfr_br == "off":
        return {}
    return {TRUNK_ARRAY: TrunkLayout(tree, BR_REGIONS[config.pcs.cfr_br]).num_slots}


def worker_bytes(
    tree: BettingTree, num_terminals: int, *, br_streets: str = "off", runouts: int = 1
) -> int:
    """Private bytes one worker's kernel allocates, from the tree's shape.

    ONE kernel whatever ``runouts`` is: CFR-BR rebinds a single kernel per
    runout rather than holding one each, which is the only reason
    ``runouts_per_flop`` above 1 fits a 32 GiB node at all.
    """
    per_hand = 4 * LIVE_HANDS * np.dtype(DTYPE).itemsize
    scratch = (len(tree) + num_terminals) * per_hand  # reach, value, both players
    cache = tree.num_slots * np.dtype(DTYPE).itemsize  # bucket-space strategy cache
    temporaries = 6 * MAX_BLOCK_ELEMENTS * np.dtype(DTYPE).itemsize  # one chunk's blocks
    picks = 0
    if br_streets != "off":
        # One int8 action per (best-response node, live hand), held from the
        # backward pass that chose it to the forward pass that plays it.
        streets = frozenset(BR_REGIONS[br_streets])
        nodes = sum(1 for node in tree.nodes if node.street in streets)
        picks = nodes * LIVE_HANDS * runouts
    return scratch + cache + temporaries + picks + WORKER_OVERHEAD_BYTES


def node_memory_bytes() -> int:
    return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


def ram_safe_workers(
    tree: BettingTree,
    num_terminals: int,
    *,
    shared_bytes: int,
    memory: int | None = None,
    br_streets: str = "off",
    runouts: int = 1,
) -> int:
    """How many workers this node can hold, from the arithmetic above."""
    total = node_memory_bytes() if memory is None else memory
    available = total - shared_bytes - NODE_HEADROOM_BYTES
    per_worker = worker_bytes(tree, num_terminals, br_streets=br_streets, runouts=runouts)
    return max(1, int(available // per_worker))


def mark_visited_from_strategy(storage: StaticArrayStorage) -> None:
    """``visited`` is what evaluation reads as 'has an answer'; derive it from mass.

    A row with no accumulated strategy plays uniform either way, so this is
    numerically a no-op -- but ``TreePolicySource`` reports an unvisited row as
    untrained, and a checkpoint that never set the flag scores like an empty one.
    """
    tree = storage.tree
    mass = np.add.reduceat(np.asarray(storage.strategy_sum, dtype=np.float64), tree.row_slot_starts)
    storage.visited[:] = mass > 0.0


def pcs_worker(
    config: Config,
    worker_id: int,
    session_id: str,
    indices: Sequence[int],
    base_seed: int,
    result_queue: Any,
    abstraction: BucketingStrategy | None = None,
    chunk_start: int = 0,
    counters: Any = None,
    _merge_lock: Any = None,  # the scalar worker's tally merge; PCS writes stay Hogwild
) -> None:
    """Sample a board per index in ``indices`` and write its update to the shared table.

    The same shape as the scalar worker: attach, seed, train, report. Every
    knob comes from ``config`` (``solver`` for the regret bookkeeping, ``pcs``
    for the sampler), so nothing else crosses the process boundary.
    """
    configure_logging(config.system.log_level)
    storage = None
    try:
        _, abstraction, tree = _build_local(config, abstraction)
        extra = trunk_arrays(config, tree)
        storage = StaticArrayStorage(tree, session_id=session_id, attach=True, extra=extra)
        compiled = compile_tree(tree, GameRules(config.game.small_blind, config.game.big_blind))
        solver, pcs = config.solver, config.pcs
        kernel: PublicChanceSamplingCFR | CFRBestResponse
        if pcs.cfr_br == "off":
            kernel = PublicChanceSamplingCFR(
                compiled,
                storage.regrets,
                storage.strategy_sum,
                weighting=solver.iteration_weighting,
                dcfr_alpha=solver.dcfr_alpha,
                dcfr_beta=solver.dcfr_beta,
                dcfr_gamma=solver.dcfr_gamma,
                cfr_plus=solver.cfr_plus,
                alternating=pcs.alternating,
                showdown=pcs.showdown,
            )
        else:
            kernel = CFRBestResponse(
                compiled,
                storage.regrets,
                storage.strategy_sum,
                storage.extra_array(TRUNK_ARRAY),
                br_streets=BR_REGIONS[pcs.cfr_br],
                weighting=solver.iteration_weighting,
                dcfr_alpha=solver.dcfr_alpha,
                dcfr_beta=solver.dcfr_beta,
                dcfr_gamma=solver.dcfr_gamma,
                cfr_plus=solver.cfr_plus,
                showdown=pcs.showdown,
                num_boards=pcs.runouts_per_flop,
                sequential=True,
            )
            logger.info(
                "[cfr-br worker %d] hybrid opponent best-responds on %s; "
                "trunk table %d slots (%.0f MB), %d shared blueprint slots",
                worker_id,
                pcs.cfr_br,
                storage.extra_array(TRUNK_ARRAY).shape[0],
                storage.extra_array(TRUNK_ARRAY).nbytes / 1e6,
                tree.num_slots,
            )
        rng = np.random.default_rng(worker_seed(base_seed, worker_id, chunk_start))

        started = time.time()
        banked = int(counters[worker_id]) if counters is not None else 0
        count = 0
        for iteration in indices:
            contexts, boards = iteration_contexts(rng, abstraction, pcs.runouts_per_flop)
            if isinstance(kernel, CFRBestResponse):
                kernel.iterate(contexts, int(iteration), boards=boards)
            else:
                kernel.iterate(contexts, int(iteration))
            count += 1
            if counters is not None:
                counters[worker_id] = banked + count
        elapsed = time.time() - started
        if isinstance(kernel, CFRBestResponse):
            logger.info(
                "[cfr-br worker %d] %d best responses recorded, %d trunk rows nonzero",
                worker_id,
                kernel.best_responses,
                int(np.count_nonzero(storage.extra_array(TRUNK_ARRAY))),
            )
        # Measured, because the worker count is sized from an estimate of this
        # number and the task log is the only place the estimate can be checked.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        logger.info(
            "[pcs worker %d] %d boards in %.0fs (%.3f boards/s), peak RSS %.2f GB",
            worker_id,
            kernel.boards,
            elapsed,
            kernel.boards / elapsed if elapsed > 0 else 0.0,
            peak_bytes / 1e9,
        )
        result_queue.put(
            {
                "worker_id": worker_id,
                "iterations": count,
                "elapsed_s": elapsed,
                "dropped": 0,
                "error": None,
            }
        )
    except Exception as exc:  # surfaced by the coordinator, never swallowed
        logger.exception(f"[pcs worker {worker_id}] failed")
        result_queue.put({"worker_id": worker_id, "error": repr(exc)})
    finally:
        if storage is not None:
            storage.close()


__all__ = (
    "LIVE_HANDS",
    "TRUNK_ARRAY",
    "iteration_contexts",
    "mark_visited_from_strategy",
    "pcs_worker",
    "ram_safe_workers",
    "sample_boards",
    "trunk_arrays",
    "worker_bytes",
)
