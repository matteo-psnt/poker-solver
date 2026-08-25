"""The agent that answers a frame with a trained blueprint.

Kept out of ``agents.py`` because that module is what the baseline probe
imports, and this one drags in the engine, the resolver and numba -- 1.2s of
import the check-call run has no use for.

Never raises. A decision that does not arrive would abandon the hand, and an
abandoned hand is not scored at all; a spot we cannot name is answered with the
cheapest legal action instead and counted, so the tally says how often it
happened rather than the score quietly absorbing it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.game.actions import ActionType
from src.interfaces.gtowizard.adapter import (
    Spot,
    TableScale,
    reconstruct,
    table_scale,
    wire_amount,
)
from src.interfaces.gtowizard.agents import Move
from src.interfaces.gtowizard.protocol import BET, CALL, CHECK, FOLD

if TYPE_CHECKING:
    from src.engine.solver.policy.source import ScorableBlueprint
    from src.interfaces.gtowizard.protocol import Frame, Turn

logger = logging.getLogger(__name__)


class BlueprintPlayer:
    """Plays one blueprint against GTO Wizard AI.

    ``allow_depth_mismatch`` exists to be left alone. Their game is 200 bb, and
    a 100 bb strategy fielded there is off our tree from the first action, so a
    score from it would separate nothing.
    """

    name = "blueprint"

    def __init__(
        self,
        blueprint: ScorableBlueprint,
        *,
        use_resolver: bool | None = None,
        budget_ms: int | None = None,
        allow_depth_mismatch: bool = False,
    ) -> None:
        self.blueprint = blueprint
        self.use_resolver = use_resolver
        self.budget_ms = budget_ms
        self.allow_depth_mismatch = allow_depth_mismatch
        self._scale: TableScale | None = None

    def scale_for(self, frame: Frame) -> TableScale:
        """The chip mapping for this game, checked once and then reused."""
        if self._scale is None:
            scale = table_scale(frame.game, self.blueprint)
            if not scale.depth_matches and not self.allow_depth_mismatch:
                raise ValueError(
                    f"Their table is {scale.their_depth:g} bb deep and this blueprint is "
                    f"cut for {scale.our_depth:g}. Train a {scale.their_depth:g} bb arm "
                    "(`--set game__starting_stack="
                    f"{int(scale.their_depth * self.blueprint.config.game.big_blind)}`), or "
                    "pass --allow-depth-mismatch to field it anyway and read the result "
                    "as a translation probe rather than a score."
                )
            self._scale = scale
        return self._scale

    def decide(self, frame: Frame) -> Move:
        turn = frame.turn
        try:
            scale = self.scale_for(frame)
        except ValueError:
            raise
        try:
            spot = reconstruct(self.blueprint, frame, scale)
        except Exception:
            logger.exception("Could not replay hand %s; passing.", frame.hand_id)
            return _passive(turn)
        if spot.truncated:
            logger.warning("Replay of hand %s did not land on our seat; passing.", frame.hand_id)
            return _passive(turn, off_tree=spot.off_tree, truncated=True)
        try:
            return self._answer(frame, spot)
        except Exception:
            logger.exception("No usable action for hand %s; passing.", frame.hand_id)
            return _passive(turn, off_tree=spot.off_tree)

    def _answer(self, frame: Frame, spot: Spot) -> Move:
        # Imported here, not at module scope: the engine and its numba kernels
        # are the expensive half of this process's start-up, and `--help` and
        # the baseline probe must not pay for them.
        from src.engine.search.agent import BlueprintAgent  # noqa: PLC0415

        agent = BlueprintAgent(self.blueprint, use_resolver=self.use_resolver)
        chosen = agent.act(spot.state, time_budget_ms=self.budget_ms)
        turn = frame.turn
        if chosen.type is ActionType.FOLD:
            return Move(FOLD, off_tree=spot.off_tree)
        if chosen.type is ActionType.CHECK:
            return Move(CHECK, off_tree=spot.off_tree)
        if chosen.type is ActionType.CALL:
            return Move(CALL, off_tree=spot.off_tree)
        if not turn.allows(BET):
            # Facing a shove their frame can still carry a positive raise_max
            # while offering only fold and call; sending a bet there is a 4xx,
            # and a 4xx mid-hand abandons a hand that would have been scored.
            return _passive(turn, off_tree=spot.off_tree)
        return Move(BET, wire_amount(chosen, turn, frame.game, spot), off_tree=spot.off_tree)


def _passive(turn: Turn, *, off_tree: int = 0, truncated: bool = False) -> Move:
    """The strongest non-aggressive action still on offer.

    Where the blueprint wanted to bet, calling is the nearer intent than
    folding -- it keeps the hand alive at the price the server will take.
    """
    if turn.allows(CHECK):
        return Move(CHECK, off_tree=off_tree, truncated=truncated)
    if turn.allows(CALL):
        return Move(CALL, off_tree=off_tree, truncated=truncated)
    return Move(FOLD, off_tree=off_tree, truncated=truncated)
