"""Playing a blueprint cut for one stack depth into a hand dealt at another.

Exploitability is denominated in the blinds of the game it was measured in, so a
25 bb and a 100 bb blueprint cannot be ranked by scoring each natively -- less
money is in play. Two strategies ranked at ONE table depth can be, and that
requires playing a blueprint somewhere it was not cut for.

It is the fiction the deployed seat already plays, made explicit: rebuild the
hand on a game whose stacks are the blueprint's OWN, ask it, and snap the answer
back onto the real table. The fiction is what is under measurement, so it is
reproduced rather than improved -- a fix belongs in the seat and here together.

ONE INVARIANT: the table and the blueprint share BLINDS. The replay carries each
action's amount across unchanged, so it converts STACKS and never currency; a
1/2 blueprint at a 5/10 table reads every raise in the wrong units and still
returns a plausible hand. `CrossDepthPlayer` refuses that rather than play it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.actions import Action, ActionType
from src.core.game.state import FULL_DECK
from src.engine.solver.policy.lookup import blueprint_action_distribution
from src.pipeline.blueprint.paths import advance_chance

if TYPE_CHECKING:
    from src.core.game.state import Card, GameState
    from src.engine.solver.policy.source import ScorableBlueprint


@dataclass
class CrossDepthTally:
    """What the fiction cost, per decision. ``truncated`` is the honest failure."""

    decisions: int = 0
    truncated: int = 0
    snapped: int = 0
    untrained: int = 0

    @property
    def truncation_rate(self) -> float:
        return self.truncated / self.decisions if self.decisions else 0.0

    def as_rates(self, prefix: str) -> dict[str, float | int]:
        """The tally as result fields, each rate beside its denominator."""
        per = self.decisions or 1
        return {
            f"{prefix}_decisions": self.decisions,
            f"{prefix}_truncation_rate": self.truncated / per,
            f"{prefix}_snap_rate": self.snapped / per,
            f"{prefix}_untrained_rate": self.untrained / per,
        }


def _placeholder_opponent(dead: tuple[Card, ...]) -> tuple[Card, Card]:
    """Two cards outside ``dead``. The opponent's holding never enters our key."""
    masks = {card.mask for card in dead}
    free = [card for card in FULL_DECK if card.mask not in masks]
    return (free[0], free[1])


def nearest_legal(wanted: Action, legal: tuple[Action, ...]) -> Action:
    """The legal action closest to ``wanted``: same type if offered, then size.

    Deterministic on purpose: a sampled mapping moves the spot under you on the
    next decision of the same hand -- measured on the live seat.
    """
    same = [action for action in legal if action.type is wanted.type]
    if same:
        return min(same, key=lambda action: abs(action.amount - wanted.amount))
    aggressive = [action for action in legal if action.is_aggressive()]
    if wanted.is_aggressive() and aggressive:
        return min(aggressive, key=lambda action: abs(action.amount - wanted.amount))
    for kind in (ActionType.CALL, ActionType.CHECK, ActionType.FOLD):
        for action in legal:
            if action.type is kind:
                return action
    return legal[0]


def safe_default(legal: tuple[Action, ...]) -> Action:
    """Check if it is free, otherwise fold -- the cheapest way to stay legal."""
    for kind in (ActionType.CHECK, ActionType.FOLD, ActionType.CALL):
        for action in legal:
            if action.type is kind:
                return action
    return legal[0]


class CrossDepthPlayer:
    """A blueprint answering states from a game with a different starting stack.

    Matched depths are the identity case and must stay so: the replay rebuilds the
    state it was given and never truncates. `test_cross_depth.py` pins that.
    """

    def __init__(self, blueprint: ScorableBlueprint, *, seed: int = 0) -> None:
        self.blueprint = blueprint
        self.rules = blueprint.rules
        self.action_model = blueprint.action_model
        self.stack = blueprint.config.game.starting_stack
        # Its OWN stream, not global `np.random`: a match reassigns it per game so
        # both sides draw from one sequence (see `depth_duel`).
        self.rng = np.random.default_rng(seed)
        self.tally = CrossDepthTally()

    @property
    def depth_in_blinds(self) -> float:
        return self.stack / self.blueprint.config.game.big_blind

    def act(self, state: GameState) -> Action:
        """An action legal at ``state``, chosen through this blueprint's own tree."""
        self.tally.decisions += 1
        self._check_units(state)
        legal = self.rules.get_legal_actions(state, action_model=self.action_model)
        mine = self._replay(state)
        if mine is None:
            self.tally.truncated += 1
            return safe_default(legal)
        wanted = self._blueprint_action(mine)
        if wanted is None:
            self.tally.untrained += 1
            return safe_default(legal)
        chosen = nearest_legal(wanted, legal)
        if chosen != wanted:
            self.tally.snapped += 1
        return chosen

    def _blueprint_action(self, state: GameState) -> Action | None:
        """A draw from the average strategy at ``state``, or None where untrained.

        Kept apart from a truncated replay in the tally: there the spot could not
        be named, here it was named and the training never learned it.
        """
        legal = self.rules.get_legal_actions(state, action_model=self.action_model)
        if not legal:
            return None
        source = self.blueprint.policy_source
        infoset = source.infoset_at(state, source.bucket_for(state, state.current_player))
        distribution = blueprint_action_distribution(
            infoset, state, self.rules, legal, use_average=True
        )
        if distribution is None:
            return None
        actions = list(distribution)
        weights = np.fromiter(distribution.values(), dtype=np.float64, count=len(actions))
        return actions[int(self.rng.choice(len(actions), p=weights / weights.sum()))]

    def _check_units(self, state: GameState) -> None:
        """Refuse a table whose blinds are not this blueprint's.

        `blind_to_call` (big minus small) is the only blind a `GameState` carries.
        """
        mine = self.rules.big_blind - self.rules.small_blind
        if state.blind_to_call and state.blind_to_call != mine:
            raise ValueError(
                f"Table blinds differ from this blueprint's: the state's "
                f"blind_to_call is {state.blind_to_call}, ours is {mine}. The "
                "replay converts stacks, never currency, so every raise amount "
                "would be read in the wrong units. Build the table from the "
                "blueprint's own blinds."
            )

    def _replay(self, state: GameState) -> GameState | None:
        """``state``'s history rebuilt on OUR stacks, or None if it did not land.

        None means the games diverged -- our shorter stack ended the betting where
        theirs did not, or the line reaches a seat that is not ours. Answering
        from the wrong seat is worse than answering with a default.
        """
        seat = state.current_player
        ours = state.hole_cards[seat]
        theirs = _placeholder_opponent((*state.board, *ours))
        hole = (ours, theirs) if seat == 0 else (theirs, ours)
        try:
            mine = self.rules.create_initial_state(
                starting_stack=self.stack, hole_cards=hole, button=state.button_position
            )
            consumed = 0
            for action in state.betting_history:
                mine, consumed = advance_chance(mine, state.board, consumed)
                if mine.is_terminal:
                    return None
                legal = self.rules.get_legal_actions(mine, action_model=self.action_model)
                if not legal:
                    return None
                mine = mine.apply_action(nearest_legal(action, legal), self.rules)
            mine, consumed = advance_chance(mine, state.board, consumed)
        except (ValueError, IndexError, KeyError):
            # `NeedsBoardError` (a board too short for the line our stack reaches)
            # is a ValueError; the rest are what a state built on the wrong stack
            # surfaces from indexing. All mean the same thing here.
            return None
        if mine.is_terminal or mine.current_player != seat:
            return None
        return mine
