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

    SNAPS DOWN: the deepest rung at or below the table. A blueprint that thinks
    it has less money than it does cannot plan a bet it has no chips to
    complete, while one that thinks it has more strands itself half way through.
    Below the shallowest rung there is nothing to snap to, so that rung answers.
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
        """The rung to answer a hand ``depth`` big blinds deep."""
        chosen = self.rungs[0]
        for rung in self.rungs:
            if rung.depth <= depth:
                chosen = rung
            else:
                break
        return chosen
