"""A ladder of blueprints, each cut for a different stack depth.

Every bot-vs-bot match on Chipzen is elimination with escalating blinds, so one
match sweeps through many depths: a measured 40-hand match ran 39-100 bb and put
29 of 85 decisions below the trained depth. One blueprint cannot answer that.

MEASURED on 40,000 duplicate deals per row, a native rung against the 100 bb
blueprint AT that rung's depth: **-465.9 +/- 6.3 mbb/hand at 6 bb, -315.9 +/- 8.1
at 10 bb, -217.5 +/- 10.0 at 15 bb** -- the rung winning every time, and by more
the shallower it gets. The incumbent duelled against ITSELF at the same depths
reads ~0, which is what rules out a harness artefact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.engine.solver.policy.source import ScorableBlueprint

logger = logging.getLogger(__name__)

# How far above the table a rung may be cut and still be preferred to a shallower
# one: measured to hold across the fielded ladder's widest gap, 50 -> 100 bb.
SNAP_UP_SPAN = 2.0


@dataclass(frozen=True)
class Rung:
    """One trained blueprint and the depth it was cut for, in big blinds."""

    depth: float
    blueprint: ScorableBlueprint

    @classmethod
    def of(cls, blueprint: ScorableBlueprint) -> Rung:
        game = blueprint.config.game
        return cls(depth=game.starting_stack / game.big_blind, blueprint=blueprint)


class DepthLadder:
    """Picks the blueprint to answer a hand of a given effective depth.

    SNAP UP: the shallowest rung cut at least this deep, when it is within
    ``SNAP_UP_SPAN`` of the table -- MEASURED 10-05 in every adjacent pair of the
    fielded ladder, the deeper rung wins at every table between them (up to ~90
    mbb/hand), and over whole elimination matches up beats log-nearest 50.6% and
    50.9% (3.7 and 6.2 sigma). Across a WIDER gap the deeper rung loses (100 vs 25
    bb at a 30 bb table: -99.5), so there it falls back to nearest in log space.
    """

    def __init__(self, blueprints: Sequence[ScorableBlueprint]) -> None:
        if not blueprints:
            raise ValueError("A ladder needs at least one blueprint.")
        self.rungs = sorted((Rung.of(b) for b in blueprints), key=lambda r: r.depth)
        depths = [r.depth for r in self.rungs]
        if len(set(depths)) != len(depths):
            raise ValueError(f"Two rungs share a depth: {depths}. One would never be selected.")
        logger.info("Ladder: %s bb", ", ".join(f"{d:g}" for d in depths))

    @property
    def deepest(self) -> Rung:
        return self.rungs[-1]

    def select(self, depth: float) -> Rung:
        """The rung to answer a hand ``depth`` big blinds deep.

        A depth at or below zero cannot be logged and cannot be played either;
        the shallowest rung is the honest answer to it.
        """
        if depth <= 0:
            return self.rungs[0]
        above = [rung for rung in self.rungs if rung.depth >= depth * (1 - 1e-9)]
        if above and above[0].depth <= SNAP_UP_SPAN * depth:
            return above[0]
        target = math.log(depth)
        return min(self.rungs, key=lambda rung: abs(math.log(rung.depth) - target))
