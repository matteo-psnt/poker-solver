"""What every seat at someone else's table needs: their chips, our tree.

Shared by the chipzen and GTO Wizard adapters, which differ only in how their
wire spells a hand. The two kept separate copies of this once, and the copy
nobody was watching kept a bug the other had been fixed for.

Two rules keep the two chip currencies from contaminating each other: their
numbers price the bet, and our numbers name the infoset. An opponent bet we had
to snap on-tree costs us a slightly wrong *label*, never a mis-sized wager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.core.game.actions import Action, ActionType
from src.core.game.state import FULL_DECK, Card, GameState
from src.engine.search.action_translation import translate_action_distribution

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.rules import GameRules
    from src.engine.solver.policy.source import ScorableBlueprint


class AdapterError(ValueError):
    """A table cannot be mapped onto this blueprint's game.

    Distinct from a protocol error: the payload was well formed, but it
    describes a table this blueprint was not trained for. The caller's only sane
    response is a safe default, so the reason has to survive as a sentence.
    """


class Blinds(Protocol):
    """Their game's denomination, as either wire reports it."""

    @property
    def small_blind(self) -> int: ...
    @property
    def big_blind(self) -> int: ...
    @property
    def depth_in_blinds(self) -> float: ...


@dataclass(frozen=True)
class TableScale:
    """Chips at their table per one chip of ours, fixed for a match.

    Our tree is cut for a stack depth in big blinds, not for a chip
    denomination, so a table is playable exactly when its blinds are a whole
    multiple of ours and its stack is then our stack. ``factor`` is that
    multiple -- 50 for their 50/100 against our 1/2.
    """

    factor: int
    their_depth: float
    our_depth: float

    @property
    def depth_matches(self) -> bool:
        """True when their starting stack is our trained depth, to the chip."""
        return abs(self.their_depth - self.our_depth) < 1e-9

    def to_ours(self, chips: int) -> int:
        """Their chips into ours, rounded to nearest -- a label, never a wager."""
        return (chips + self.factor // 2) // self.factor

    def to_theirs(self, chips: int) -> int:
        """Our chips into theirs. Exact; ``factor`` is an integer by construction."""
        return chips * self.factor


def table_scale(table: Blinds, blueprint: ScorableBlueprint) -> TableScale:
    """How their chips map onto the blueprint's, or why they do not.

    Refuses rather than approximates. A blind ratio that is not a whole number
    would put every bet size between two of our tree's rungs, and a depth we did
    not train is a different game -- both are worth a forfeited match far less
    than they are worth a log line saying which one happened.
    """
    ours = blueprint.config.game
    if table.big_blind % ours.big_blind or table.small_blind % ours.small_blind:
        raise AdapterError(
            f"Table blinds {table.small_blind}/{table.big_blind} are not a whole "
            f"multiple of the blueprint's {ours.small_blind}/{ours.big_blind}."
        )
    factor = table.big_blind // ours.big_blind
    if table.small_blind // ours.small_blind != factor:
        raise AdapterError(
            f"Table blinds {table.small_blind}/{table.big_blind} scale unevenly "
            f"against the blueprint's {ours.small_blind}/{ours.big_blind}."
        )
    return TableScale(
        factor=factor,
        their_depth=table.depth_in_blinds,
        our_depth=ours.starting_stack / ours.big_blind,
    )


def opponent_hole_cards(dead: tuple[Card, ...]) -> tuple[Card, Card]:
    """Two cards outside ``dead``, standing in for a holding we cannot see.

    The opponent's cards do not enter our own bucket or our infoset key, so any
    two will do -- but a collision with the board or our hand would build a
    state that cannot exist, and bucketing would then be asked about it.
    """
    masks = {card.mask for card in dead}
    free = [card for card in FULL_DECK if card.mask not in masks]
    if len(free) < 2:
        raise AdapterError("Too few cards left to stand in for the opponent's hand.")
    return (free[0], free[1])


def their_raise(
    total: int,
    high: int,
    state: GameState,
    scale: TableScale,
    legal: tuple[Action, ...],
) -> tuple[Action, bool]:
    """Their raise to ``total`` over a round high of ``high`` (their chips) as one
    of our actions, plus whether the size had to be approximated."""
    increment = scale.to_ours(total) - scale.to_ours(high)
    if increment <= 0:
        # Their raise rounds to no increment in our coarser chips: it is a call
        # as far as our tree can tell.
        return Action(ActionType.CALL), True
    # OUR state names the shape, not their wire: a big blind raising over a limp
    # owes nothing, so our tree offers it BETs. Typing it a RAISE off the posted
    # blind found no RAISE, fell back to all-in, and ended our replay at the next
    # call -- 1,634 of 1,700 truncations (96%) in the 09-29..10-08 journal.
    kind = ActionType.RAISE if state.to_call > 0 else ActionType.BET
    # What the actor must actually part with: a RAISE's amount sits ON TOP of the
    # call, so comparing the bare amount to the stack would miss a shove by
    # exactly `to_call` and hand the translator an action nobody can afford.
    stack = state.stacks[state.current_player]
    committed = state.to_call + increment if kind is ActionType.RAISE else increment
    if committed >= stack:
        return Action(ActionType.ALL_IN, stack), True
    if not any(action.type is kind for action in legal):
        # The tree does not offer this shape here (a raise cap, typically). Fall
        # back to the most aggressive thing it does offer.
        return _most_aggressive(legal), True
    return Action(kind, increment), False


#: A preflop raise past this multiple of our largest raise increment is read as
#: all-in. Every tree since 2025-12 sizes 3-bets off `to_call`, so its biggest one
#: over a 2 bb open is to 5.5 bb, and a 10-12.5 bb 3-bet snapped there plays a big
#: pot as a small one. 1.6 fires on 66-70% of fermat6's and NashForge's 3-bets in
#: the 09-20..10-28 journal and on none from the field's on-tree bots (max 1.43).
OVERSIZED_RAISE = 1.6


def oversized(observed: Action, state: GameState, legal: tuple[Action, ...]) -> bool:
    """Whether ``observed`` is a preflop raise our tree can only understate."""
    if observed.type is not ActionType.RAISE or not state.street.is_preflop():
        return False
    raises = [action.amount for action in legal if action.type is ActionType.RAISE]
    jams = any(action.type is ActionType.ALL_IN for action in legal)
    return bool(raises) and jams and observed.amount > OVERSIZED_RAISE * max(raises)


def on_tree(
    observed: Action,
    state: GameState,
    action_model: ActionModel,
    rules: GameRules,
    legal: tuple[Action, ...],
) -> Action:
    """``observed`` if the tree offers it, else the likeliest legal stand-in.

    Snapped rather than sampled: a stable wrong mapping replays identically on
    the next turn of the same hand, and a sampled one would move the spot.
    """
    if observed in legal:
        return observed
    distribution = translate_action_distribution(state, observed, action_model, rules)
    if not distribution:
        return _most_aggressive(legal)
    return max(distribution, key=lambda pair: pair[1])[0]


def _most_aggressive(legal: tuple[Action, ...]) -> Action:
    """The biggest commitment on offer -- the last resort when a size is unreachable."""
    aggressive = [action for action in legal if action.is_aggressive()]
    if aggressive:
        return max(aggressive, key=lambda action: action.amount)
    for kind in (ActionType.CALL, ActionType.CHECK, ActionType.FOLD):
        for action in legal:
            if action.type is kind:
                return action
    return legal[0]
