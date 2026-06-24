"""Public chance sampling: one sampled board per iteration, every hand at once.

A :class:`VectorCFR` pass values one board exactly -- card removal, showdown
ranks, all 1,081 live hands -- and writes into the same ``(node, bucket,
action)`` table the scalar trainer fills one deal at a time. Drawing a fresh
board every iteration makes that pass an unbiased sample of the full game's
chance layer (Johanson et al. 2012), which is what turns the subgame solver
into a blueprint trainer. Boards are IID, never a cycled schedule: reuse is
biased outside balanced regret matching, and this table is DCFR over
imperfect-recall buckets.

Regret and average-strategy semantics are the production scalar kernel's
(``numba_ops``): DCFR's t^a/(t^a+1) on positive and t^b/(t^b+1) on negative
stored regrets of every row this board occupies, the strategy sum weighted by
t^gamma, ``iteration`` the 0-based absolute index every worker reads as t.
Only the opponent's strategy enters the passes through one seam --
``VectorCFR._strategy_block`` -- so a best-responding opponent (CFR-BR) is a
substitution there, not a second kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Street
from src.engine.solver.numba_ops import compute_dcfr_strategy_weight
from src.engine.solver.vector.kernel import DTYPE, NodeGroup, VectorCFR, build_groups

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.engine.solver.vector.compiled_tree import CompiledTree
    from src.engine.solver.vector.hand_context import HandContext

STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)


def dcfr_discount(iteration: int, alpha: float, beta: float) -> tuple[float, float] | None:
    """``(positive, negative)`` multipliers for stored regrets at ``iteration``.

    Exactly ``numba_ops.apply_regret_updates``: nothing before t=2, and an
    exponent of 0 is x0.5, not a no-op.
    """
    if iteration <= 1:
        return None

    def factor(exponent: float) -> float:
        if exponent == 0.0:
            return 0.5
        power = float(iteration) ** exponent
        return power / (power + 1.0)

    return factor(alpha), factor(beta)


class PublicChanceSamplingCFR:
    """Drives one :class:`VectorCFR` over sampled boards into a shared table.

    ``regrets``/``strategy_sum`` are the caller's -- shared memory under
    Hogwild -- and the kernel's own allocations are replaced by them. The
    kernel is built on the first board, because its scratch is shaped by the
    live-hand count and every full board has the same one.
    """

    def __init__(
        self,
        compiled: CompiledTree,
        regrets: np.ndarray,
        strategy_sum: np.ndarray,
        *,
        weighting: str = "dcfr",
        dcfr_alpha: float = 1.5,
        dcfr_beta: float = 0.0,
        dcfr_gamma: float = 2.0,
        cfr_plus: bool = False,
        alternating: bool = False,
        showdown: str = "walk",
    ):
        if weighting not in ("none", "linear", "dcfr"):
            raise ValueError(f"Unknown iteration weighting {weighting!r}.")
        self.compiled = compiled
        self.regrets = regrets
        self.strategy_sum = strategy_sum
        self.weighting = weighting
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.dcfr_gamma = dcfr_gamma
        self.cfr_plus = cfr_plus
        self.alternating = alternating
        self.showdown = showdown
        self.groups: list[NodeGroup] = build_groups(compiled)
        self.kernel: VectorCFR | None = None
        self._delta: np.ndarray | None = None
        self.boards = 0  # board passes completed, for throughput accounting

    def _bound(self, context: HandContext) -> VectorCFR:
        if self.kernel is None:
            kernel = VectorCFR(
                self.compiled,
                context,
                cfr_plus=self.cfr_plus,
                showdown=self.showdown,
                groups=self.groups,
            )
            kernel.regrets = self.regrets
            kernel.strategy_sum = self.strategy_sum
            self.kernel = kernel
        else:
            self.kernel.bind(context)
        return self.kernel

    def _schedule(self, kernel: VectorCFR, iteration: int, runouts: int) -> None:
        """Set the pass controls for absolute iteration ``iteration``."""
        kernel.update_players = (iteration % 2,) if self.alternating else (0, 1)
        if self.weighting == "dcfr":
            weight = float(compute_dcfr_strategy_weight(iteration, self.dcfr_gamma))
            scale = 1.0
            kernel.regret_discount = dcfr_discount(iteration, self.dcfr_alpha, self.dcfr_beta)
        elif self.weighting == "linear":
            weight = scale = float(max(iteration, 1))
            kernel.regret_discount = None
        else:
            weight = scale = 1.0
            kernel.regret_discount = None
        # Averaged over the runouts of one iteration, so K passes carry the
        # weight of one board: the update is an estimate of the chance
        # expectation, not K times it.
        kernel.strategy_weight = weight / runouts
        kernel.delta_scale = scale / runouts

    def iterate(self, contexts: Sequence[HandContext], iteration: int) -> None:
        """One iteration: the runouts of one sampled board, at absolute ``iteration``.

        Several contexts are runouts beneath ONE flop, valued against the same
        strategy and summed before the discount -- the mixture's joint rule,
        since flooring or discounting per runout is a different algorithm.
        """
        if not contexts:
            raise ValueError("An iteration needs at least one board.")
        kernel = self._bound(contexts[0])
        self._schedule(kernel, iteration, len(contexts))
        initial = np.ones(kernel.num_hands, dtype=DTYPE)

        if len(contexts) == 1:
            kernel.iterate(initial)
            self.boards += 1
            return

        if self._delta is None:
            self._delta = np.zeros(self.compiled.tree.num_slots, dtype=DTYPE)
        self._delta[:] = 0.0
        occupied: dict[Street, set[int]] = {street: set() for street in STREETS}
        kernel.regret_target = self._delta
        try:
            for context in contexts:
                kernel.bind(context)
                for street in STREETS:
                    occupied[street].update(kernel.segments[street].segment_bucket.tolist())
                kernel.iterate(initial)
                self.boards += 1
        finally:
            kernel.regret_target = None

        union = {
            street: np.array(sorted(rows), dtype=np.int64) for street, rows in occupied.items()
        }
        for group in self.groups:
            if group.actor not in kernel.update_players:
                continue
            buckets = self.compiled.tree.num_buckets(group.street)
            for chunk in kernel.chunks(group):
                index = kernel._slot_index(group, chunk)  # noqa: SLF001 -- the kernel's own layout
                block = self._delta[index].reshape(-1, buckets, group.num_actions)
                kernel.apply_regret_block(index, block, union[group.street])


__all__ = ("PublicChanceSamplingCFR", "dcfr_discount")
