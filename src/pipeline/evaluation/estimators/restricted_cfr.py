"""Train on a FINITE board pool and score it exactly: where does CFR stop improving?

The full game's exploitability can only be estimated on a board sample, and its
floor mixes three suspects -- the abstraction, public chance sampling, and the
weighting. A fixed pool of ``flops x turns x rivers`` boards is a finite game
whose in-abstraction best response :class:`BoardMixtureCFR` computes EXACTLY,
so the same production tree, abstraction and DCFR knobs can be trained two ways
on it and both scored on one ruler:

    full      every iteration passes ALL boards -- full-width, deterministic.
    sampled   every iteration passes the rivers under ONE drawn (flop, turn),
              exactly production's ``runout_mode=turn`` sampler on the pool.

If ``full`` keeps falling while ``sampled`` flattens, sampling is the floor; if
both flatten at one level, the abstraction is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engine.solver.vector.kernel import DTYPE
from src.engine.solver.vector.mixture import BoardMixtureCFR
from src.engine.solver.vector.pcs import PublicChanceSamplingCFR
from src.pipeline.abstraction.vector_universe import build_hand_context

if TYPE_CHECKING:
    from src.engine.solver.protocols import BucketingStrategy
    from src.engine.solver.vector.compiled_tree import CompiledTree

MODES = ("full", "sampled")


@dataclass(frozen=True)
class RestrictedConfig:
    """The pool's shape, the arm, and how long it trains."""

    flops: int = 2
    turns: int = 2
    rivers: int = 4
    seed: int = 7
    mode: str = "full"
    board_passes: int = 8_000  # matched across arms: full spends K per iteration
    evaluations: int = 14  # log-spaced checkpoints of the curve


def board_pool(
    rng: np.random.Generator, flops: int, turns: int, rivers: int
) -> list[list[np.ndarray]]:
    """Boards grouped by (flop, turn): each group is the rivers one sampled iteration passes."""
    groups: list[list[np.ndarray]] = []
    seen_flops: set[frozenset[int]] = set()
    while len(seen_flops) < flops:
        flop = rng.choice(52, 3, replace=False)
        key = frozenset(int(c) for c in flop)
        if key in seen_flops:
            continue
        seen_flops.add(key)
        rest = np.setdiff1d(np.arange(52), flop)
        for turn in rng.choice(rest, turns, replace=False):
            left = np.setdiff1d(rest, [turn])
            groups.append(
                [
                    np.concatenate([flop, [turn, river]])
                    for river in rng.choice(left, rivers, replace=False)
                ]
            )
    return groups


def train_restricted(
    compiled: CompiledTree,
    abstraction: BucketingStrategy,
    config: RestrictedConfig,
    *,
    weighting: str,
    dcfr_alpha: float,
    dcfr_beta: float,
    dcfr_gamma: float,
    cfr_plus: bool,
    big_blind: float,
) -> dict[str, Any]:
    """The arm's exploitability curve on its own pool, in mbb per hand."""
    if config.mode not in MODES:
        raise ValueError(f"Unknown mode {config.mode!r}; expected one of {MODES}.")
    rng = np.random.default_rng(config.seed)
    groups = board_pool(rng, config.flops, config.turns, config.rivers)
    boards = [board for group in groups for board in group]
    contexts = [build_hand_context(board, abstraction) for board in boards]
    grouped = [contexts[i * config.rivers : (i + 1) * config.rivers] for i in range(len(groups))]

    regrets = np.zeros(compiled.tree.num_slots, dtype=DTYPE)
    strategy_sum = np.zeros(compiled.tree.num_slots, dtype=DTYPE)
    trainer = PublicChanceSamplingCFR(
        compiled,
        regrets,
        strategy_sum,
        weighting=weighting,
        dcfr_alpha=dcfr_alpha,
        dcfr_beta=dcfr_beta,
        dcfr_gamma=dcfr_gamma,
        cfr_plus=cfr_plus,
    )
    scorer = BoardMixtureCFR(compiled, contexts, boards=boards)
    for board in scorer.boards:
        board.strategy_sum = strategy_sum
    scorer.strategy_sum = strategy_sum
    initial = np.ones(contexts[0].num_hands, dtype=DTYPE)
    pairs = float(np.mean([(~c.blocks).sum() for c in contexts]))

    per_iteration = len(contexts) if config.mode == "full" else config.rivers
    iterations = max(1, config.board_passes // per_iteration)
    marks = sorted({round(float(m)) for m in np.geomspace(1, iterations, config.evaluations)})
    curve: list[dict[str, float]] = []
    draw = np.random.default_rng(config.seed + 1)
    started, scoring = time.monotonic(), 0.0
    for iteration in range(iterations):
        batch = contexts if config.mode == "full" else grouped[int(draw.integers(len(grouped)))]
        trainer.iterate(batch, iteration)
        if iteration + 1 in marks:
            tick = time.monotonic()
            chips = scorer.exploitability(initial, pairs)
            scoring += time.monotonic() - tick
            curve.append(
                {
                    "iteration": iteration + 1,
                    "board_passes": (iteration + 1) * per_iteration,
                    "seconds": time.monotonic() - started - scoring,
                    "mbb": 1000.0 * chips / big_blind,
                }
            )
    return {
        "mode": config.mode,
        "pool": [config.flops, config.turns, config.rivers],
        "boards": len(contexts),
        "pool_seed": config.seed,
        "curve": curve,
        "final_mbb": curve[-1]["mbb"],
    }


__all__ = ("MODES", "RestrictedConfig", "board_pool", "train_restricted")
