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

    NEAREST IN LOG SPACE, because depth error is proportional: a 25 bb blueprint
    is as wrong at 50 bb as a 50 bb one is at 100. MEASURED across the 25-100 bb
    gap, the 100 bb rung against the 25 bb rung at each table:

        30 bb  -99.5 +/- 14.8   40 bb  -82.9 +/- 18.7   (the 25 bb rung wins)
        60 bb  +42.6 +/- 28.2   80 bb +140.5 +/- 37.4   (the 100 bb rung wins)

    The crossover is near sqrt(25 x 100) = 50, and log-nearest calls all four
    correctly. Snapping DOWN would have answered an 80 bb table with the 25 bb
    rung and given up 140 mbb/hand; between ADJACENT rungs 1.67x apart the choice
    measures as nothing, so this only matters where the ladder has a gap -- which
    is exactly when it is easy to get wrong.
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
        target = math.log(depth)
        return min(self.rungs, key=lambda rung: abs(math.log(rung.depth) - target))
