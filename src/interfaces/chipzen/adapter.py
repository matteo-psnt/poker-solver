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
from src.interfaces.chipzen.protocol import ProtocolError, TurnState
from src.interfaces.table import (
    AdapterError,
    TableScale,
    on_tree,
    opponent_hole_cards,
    oversized,
    their_raise,
)
from src.pipeline.blueprint.paths import advance_chance

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.rules import GameRules
    from src.core.game.state import GameState
    from src.engine.solver.policy.source import ScorableBlueprint

# Their vocabulary. There is no all-in on the wire: a shove is a raise whose
# amount equals max_raise, and a short call is just a call the server caps.
FOLD = "fold"
CHECK = "check"
CALL = "call"
RAISE = "raise"


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
    # Which of the three ways a replay can fail to land, or "" when it landed.
    # Named rather than inferred: the last attempt to identify the cause from the
    # journal read a `grep -B1` filter as a correlation and was wrong.
    truncated_reason: str = ""
    # An oversized preflop raise of theirs was replayed as all-in, so our
    # "call" (an ALL_IN in our tree) goes out as a shove (see `table.oversized`).
    read_as_jam: bool = False


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
    reason = ""
    # Their per-round wagers, in their chips, so a raise-to can be read as the
    # increment our Action wants without trusting our own arithmetic for it.
    wagers = _opening_wagers(turn)
    phase = "preflop"

    replayed = 0  # of their actions we got through before stopping
    read_as_jam = False
    for entry in turn.action_history:
        if entry.is_synthetic:
            continue
        if entry.phase != phase:
            phase = entry.phase
            wagers = [0, 0]
        state, consumed = advance_chance(state, board, consumed)
        if state.is_terminal:
            truncated = True
            reason = f"our tree ended the hand {replayed} of their actions in"
            break
        legal = rules.get_legal_actions(state, action_model=action_model)
        if not legal:
            truncated = True
            reason = f"our tree offered no legal action {replayed} of their actions in"
            break
        observed, snapped = _observed_action(entry, wagers, state, scale, legal)
        read_as_jam = oversized(observed, state, legal)
        if read_as_jam:
            observed = next(action for action in legal if action.type is ActionType.ALL_IN)
        chosen = on_tree(observed, state, action_model, rules, legal)
        off_tree += int(snapped or read_as_jam or chosen != observed)
        state = state.apply_action(chosen, rules)
        # A call's `amount` is the INCREMENT on the live wire (SB limp: 50 at 50/100),
        # not the total its docstring promises; a call always matches the high bet.
        wagers[entry.seat] = (
            max(wagers) if entry.action == CALL else max(wagers[entry.seat], entry.amount)
        )
        replayed += 1

    state, consumed = advance_chance(state, board, consumed)
    if not truncated and (state.is_terminal or state.current_player != seat):
        # Their frame says it is our turn; if ours disagrees the replay drifted,
        # which is worth knowing about rather than answering from the wrong seat.
        truncated = True
        reason = (
            "our tree ended the hand"
            if state.is_terminal
            else f"replay landed on seat {state.current_player}, not {seat}"
        )
    return Spot(
        state=state,
        scale=scale,
        seat=seat,
        off_tree=off_tree,
        truncated=truncated,
        truncated_reason=reason,
        read_as_jam=read_as_jam,
    )


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

    return their_raise(entry.amount, max(wagers), state, scale, legal)


def wire_action(chosen: Action, turn: TurnState, spot: Spot) -> dict[str, Any]:
    """Our action as a ``turn_action`` payload, priced in their chips.

    The size is rebuilt from their ``to_call`` and their existing wager rather
    than from our reconstructed pot, then clamped into ``[min_raise, max_raise]``
    -- so a replay that drifted costs us a slightly odd size and never a rejected
    frame. An all-in is their ``max_raise``, since their wire has no all-in.

    Aggression is filtered through ``turn.valid_actions`` last. Facing an all-in
    their frame still carries a positive ``max_raise`` while offering only fold
    and call, and sending a raise there is an ``action_rejected`` -- after which
    THEY pick a safe default and the blueprint's choice is thrown away. Measured
    on the first live hand against PluriBot.
    """
    if chosen.type is ActionType.FOLD:
        return {"action": FOLD, "params": {}}
    if chosen.type is ActionType.CHECK:
        return {"action": CHECK, "params": {}}
    if chosen.type is ActionType.CALL:
        return {"action": CALL, "params": {}}

    if not turn.allows(RAISE):
        return _passive(turn)

    if chosen.type is ActionType.ALL_IN:
        return {"action": RAISE, "params": {"amount": turn.max_raise}}

    mine = turn.round_wager(spot.seat)
    increment = spot.scale.to_theirs(chosen.amount)
    total = mine + turn.to_call + increment if chosen.type is ActionType.RAISE else increment
    return {"action": RAISE, "params": {"amount": _clamp_raise(total, turn)}}


def _passive(turn: TurnState) -> dict[str, Any]:
    """The strongest non-aggressive action still on offer.

    Where the blueprint wanted to raise, calling is the nearer intent than
    folding -- it keeps the hand alive at the price the server will take.
    """
    if turn.allows(CALL) and turn.to_call > 0:
        return {"action": CALL, "params": {}}
    if turn.allows(CHECK) and turn.to_call <= 0:
        return {"action": CHECK, "params": {}}
    return {"action": FOLD, "params": {}}


def _clamp_raise(total: int, turn: TurnState) -> int:
    """A raise total the server will accept, or the nearest one it will."""
    if turn.max_raise <= 0:
        raise AdapterError("The blueprint wants to raise where raising is not offered.")
    return max(turn.min_raise, min(total, turn.max_raise))
