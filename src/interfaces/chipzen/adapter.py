"""Turning a Chipzen turn into an infoset, and a blueprint's answer into a wire action.

Everything here is a pure function of one ``turn_request``. Nothing is carried
between turns on purpose: their frame ships the whole ``action_history``, so a
client that reconnects mid-hand rebuilds the same spot from the same bytes, and
there is no session state to get stale. It also makes the interesting half
testable without a socket -- which is the only way it is testable at all, since
we cannot hold an account and a match open in CI.

Why this package reaches into the engine when ``interfaces.cloud`` may not:
dispatching work and *playing a hand* are different jobs. The cloud package
queues tasks and must not know how a hand is solved; this one is the deployed
player, and naming an infoset is the entire thing it does.

Two rules keep the two chip currencies from contaminating each other:

- Their numbers price the bet. ``TurnState`` is authoritative for ``pot``,
  ``to_call``, ``min_raise`` and ``max_raise``, and every amount we send back is
  derived from those.
- Our numbers name the infoset. The reconstructed :class:`GameState` exists to
  be looked up in the tree and for nothing else, so an opponent bet we had to
  snap on-tree costs us a slightly wrong *label*, never a mis-sized wager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.game.actions import Action, ActionType
from src.core.game.state import FULL_DECK, Card, GameState
from src.engine.search.action_translation import translate_action_distribution
from src.interfaces.chipzen.protocol import GameConfig, ProtocolError, TurnState
from src.pipeline.blueprint.paths import advance_chance

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.rules import GameRules
    from src.engine.solver.policy.source import ScorableBlueprint

# Their vocabulary. There is no all-in on the wire: a shove is a raise whose
# amount equals max_raise, and a short call is just a call the server caps.
FOLD = "fold"
CHECK = "check"
CALL = "call"
RAISE = "raise"


class AdapterError(ValueError):
    """A turn cannot be mapped onto this blueprint's game.

    Distinct from :class:`~src.interfaces.chipzen.protocol.ProtocolError`: the
    payload was well-formed, but it describes a table this blueprint was not
    trained for. The caller's only sane response is a safe default, so the reason
    has to survive as a sentence rather than as a None.
    """


@dataclass(frozen=True)
class TableScale:
    """Chips at their table per one chip of ours, fixed for a match.

    Our tree is cut for a stack depth in big blinds, not for a chip denomination,
    so a table is playable exactly when its blinds are a whole multiple of ours
    and its stack is then our stack. ``factor`` is that multiple.
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


def table_scale(config: GameConfig, blueprint: ScorableBlueprint) -> TableScale:
    """How this table's chips map onto the blueprint's, or why they do not.

    Refuses rather than approximates. A blind ratio that is not a whole number
    would put every bet size between two of our tree's rungs, and a depth we did
    not train is a different game -- both are worth a forfeited match far less
    than they are worth a log line saying which one happened.
    """
    game = blueprint.config.game
    if config.big_blind % game.big_blind or config.small_blind % game.small_blind:
        raise AdapterError(
            f"Table blinds {config.small_blind}/{config.big_blind} are not a whole "
            f"multiple of the blueprint's {game.small_blind}/{game.big_blind}."
        )
    factor = config.big_blind // game.big_blind
    if config.small_blind // game.small_blind != factor:
        raise AdapterError(
            f"Table blinds {config.small_blind}/{config.big_blind} scale unevenly "
            f"against the blueprint's {game.small_blind}/{game.big_blind}."
        )
    return TableScale(
        factor=factor,
        their_depth=config.depth_in_blinds,
        our_depth=game.starting_stack / game.big_blind,
    )


def opponent_hole_cards(dead: tuple[Card, ...]) -> tuple[Card, Card]:
    """Two cards outside ``dead``, standing in for a holding we cannot see.

    The opponent's cards do not enter our own bucket or our infoset key, so any
    two will do -- but a collision with the board or our hand would build a state
    that cannot exist, and bucketing would then be asked about it.
    """
    masks = {card.mask for card in dead}
    free = [card for card in FULL_DECK if card.mask not in masks]
    if len(free) < 2:
        raise AdapterError("Too few cards left to stand in for the opponent's hand.")
    return (free[0], free[1])


@dataclass(frozen=True)
class Spot:
    """A Chipzen turn, resolved onto our tree.

    ``off_tree`` counts the opponent actions that had to be snapped to a legal
    size to get here. It is the honest measure of how far the label has drifted
    from the table, and the number to watch when a blueprint plays worse in the
    arena than it scores at home.
    """

    state: GameState
    scale: TableScale
    seat: int
    off_tree: int
    truncated: bool


def reconstruct(
    blueprint: ScorableBlueprint,
    turn: TurnState,
    seat: int,
    scale: TableScale,
) -> Spot:
    """Replay their history into one of our states, in our chips.

    Off-tree opponent sizes are snapped to the likeliest legal action rather than
    sampled: a stable wrong mapping replays identically on the next turn of the
    same hand, and a sampled one would move the spot under us mid-hand.

    ``truncated`` marks a history our tree ran out of room for -- their betting
    can outlive our action model's raise cap, and when it does the state we
    return is the deepest one we could reach rather than the real one.
    """
    rules: GameRules = blueprint.rules
    action_model: ActionModel = blueprint.action_model
    dead = (*turn.board, *turn.hole_cards)
    theirs = opponent_hole_cards(dead)
    hole = (turn.hole_cards, theirs) if seat == 0 else (theirs, turn.hole_cards)

    state = rules.create_initial_state(
        starting_stack=blueprint.config.game.starting_stack,
        hole_cards=hole,
        button=turn.button_seat(),
    )

    board = turn.board
    consumed = 0
    off_tree = 0
    truncated = False
    # Their per-round wagers, in their chips, so a raise-to can be read as the
    # increment our Action wants without trusting our own arithmetic for it.
    wagers = _opening_wagers(turn)
    phase = "preflop"

    for entry in turn.action_history:
        if entry.is_synthetic:
            continue
        if entry.phase != phase:
            phase = entry.phase
            wagers = [0, 0]
        state, consumed = advance_chance(state, board, consumed)
        if state.is_terminal:
            truncated = True
            break
        legal = rules.get_legal_actions(state, action_model=action_model)
        if not legal:
            truncated = True
            break
        observed, snapped = _observed_action(entry, wagers, state, scale, legal)
        chosen = _on_tree(observed, state, action_model, rules, legal)
        off_tree += int(snapped or chosen != observed)
        state = state.apply_action(chosen, rules)
        wagers[entry.seat] = max(wagers[entry.seat], entry.amount)

    state, consumed = advance_chance(state, board, consumed)
    if not truncated and (state.is_terminal or state.current_player != seat):
        # Their frame says it is our turn; if ours disagrees the replay drifted,
        # which is worth knowing about rather than answering from the wrong seat.
        truncated = True
    return Spot(state=state, scale=scale, seat=seat, off_tree=off_tree, truncated=truncated)


def _opening_wagers(turn: TurnState) -> list[int]:
    """Blinds and antes already posted, by seat, in their chips."""
    wagers = [0, 0]
    for entry in turn.action_history:
        if entry.is_synthetic:
            wagers[entry.seat] += entry.amount
    return wagers


def _observed_action(
    entry,
    wagers: list[int],
    state: GameState,
    scale: TableScale,
    legal: tuple[Action, ...],
) -> tuple[Action, bool]:
    """Their action as one of ours, plus whether the size had to be approximated.

    A raise is read as a total wager, converted to our chips, and expressed as
    the increment above the current high bet -- which is what
    :attr:`Action.amount` means for a RAISE and what a BET's total collapses to
    when nothing is yet in.
    """
    if entry.action == FOLD:
        return Action(ActionType.FOLD), False
    if entry.action == CHECK:
        return Action(ActionType.CHECK), False
    if entry.action == CALL:
        return Action(ActionType.CALL), False
    if entry.action != RAISE:
        raise ProtocolError(f"'{entry.action}' is not an action a seat can take.")

    high = max(wagers)
    increment = scale.to_ours(entry.amount) - scale.to_ours(high)
    if increment <= 0:
        # Their raise rounds to no increment in our coarser chips: it is a call
        # as far as our tree can tell.
        return Action(ActionType.CALL), True
    kind = ActionType.BET if high == 0 else ActionType.RAISE
    amount = increment if kind is ActionType.RAISE else scale.to_ours(entry.amount)
    # What the actor must actually part with: a RAISE's amount sits ON TOP of the
    # call, so comparing the bare amount to the stack would miss a shove by
    # exactly `to_call` and hand the translator an action nobody can afford.
    stack = state.stacks[state.current_player]
    committed = state.to_call + amount if kind is ActionType.RAISE else amount
    if committed >= stack:
        return Action(ActionType.ALL_IN, stack), True
    if not any(action.type is kind for action in legal):
        # The tree does not offer this shape here (a raise cap, typically). Fall
        # back to the most aggressive thing it does offer.
        return _most_aggressive(legal), True
    return Action(kind, amount), False


def _on_tree(
    observed: Action,
    state: GameState,
    action_model: ActionModel,
    rules: GameRules,
    legal: tuple[Action, ...],
) -> Action:
    """``observed`` if the tree offers it, else the likeliest legal stand-in."""
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


def wire_action(chosen: Action, turn: TurnState, spot: Spot) -> dict[str, Any]:
    """Our action as a ``turn_action`` payload, priced in their chips.

    The size is rebuilt from their ``to_call`` and their existing wager rather
    than from our reconstructed pot, then clamped into ``[min_raise, max_raise]``
    -- so a replay that drifted costs us a slightly odd size and never a rejected
    frame. An all-in is their ``max_raise``, since their wire has no all-in.
    """
    if chosen.type is ActionType.FOLD:
        return {"action": FOLD, "params": {}}
    if chosen.type is ActionType.CHECK:
        return {"action": CHECK, "params": {}}
    if chosen.type is ActionType.CALL:
        return {"action": CALL, "params": {}}
    if chosen.type is ActionType.ALL_IN:
        return {"action": RAISE, "params": {"amount": turn.max_raise}}

    mine = turn.round_wager(spot.seat)
    increment = spot.scale.to_theirs(chosen.amount)
    total = mine + turn.to_call + increment if chosen.type is ActionType.RAISE else increment
    return {"action": RAISE, "params": {"amount": _clamp_raise(total, turn)}}


def _clamp_raise(total: int, turn: TurnState) -> int:
    """A raise total the server will accept, or the nearest one it will."""
    if turn.max_raise <= 0:
        raise AdapterError("The blueprint wants to raise where raising is not offered.")
    return max(turn.min_raise, min(total, turn.max_raise))
