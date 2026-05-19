"""Naming a spot in the game, and replaying it back into a :class:`GameState`.

Why a path rather than a node id
--------------------------------
:class:`~src.engine.solver.betting_tree.BettingTree` addresses a node by an
integer index, and offers ``node_id(state)`` but no ``children(node_id)`` and no
``state_for(node_id)`` -- enumeration happens by walking states. So a caller
cannot navigate by id even if it wanted to.

That constraint points the right way. A node id is a position in a layout that a
retrain or an abstraction change reshuffles, so a stored id silently becomes a
*different* spot; the failure has no symptom. A path is stated in the game's own
terms -- fold, call, bet 150 -- so it either replays to the same spot or refuses
because the action is no longer offered. Bookmarks, links and saved analyses all
depend on that difference.

The board is given, never dealt
-------------------------------
Replay takes the board explicitly and consumes it at each chance node instead of
sampling. Two reasons: an analysis that dealt its own runout would answer a
different question on every request, and bucketing is a function of the board, so
"the strategy at this node" is not defined until the board is.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Card, GameState
from src.engine.solver.mccfr.chance import is_chance_node
from src.engine.solver.policy.source import ScorableBlueprint

_TOKEN_BY_TYPE = {
    ActionType.FOLD: "f",
    ActionType.CHECK: "x",
    ActionType.CALL: "c",
    ActionType.BET: "b",
    ActionType.RAISE: "r",
    ActionType.ALL_IN: "A",
}
_TYPE_BY_TOKEN = {token: action_type for action_type, token in _TOKEN_BY_TYPE.items()}

_SIZED = (ActionType.BET, ActionType.RAISE)

SEPARATOR = "/"


class PathError(ValueError):
    """A path does not describe a reachable spot in this blueprint's game.

    Raised rather than returned as a null so a caller cannot mistake "that line
    does not exist" for "the strategy there is empty". The message names the
    offending token and what was actually on offer, because the usual cause is a
    path saved under a different action model -- and a bare "invalid path" would
    leave no way to tell that from a typo.
    """


@dataclass(frozen=True)
class ReplayedNode:
    """Where a path led, and what is true there.

    ``actor`` is the seat to act. It is ``None`` at a terminal, which is a
    legitimate destination -- a caller may well want to ask what a line ends in.
    """

    state: GameState
    actor: int | None
    legal_actions: tuple[Action, ...]
    board_consumed: int


def encode_action(action: Action) -> str:
    """Wire token for ``action`` -- ``"c"``, ``"b150"``, ``"A"``.

    Sized actions carry the amount they raise *to*, not by, matching
    :class:`Action`; a "by" encoding would need the pot to be interpreted and so
    would not survive being read on its own.
    """
    token = _TOKEN_BY_TYPE[action.type]
    return f"{token}{action.amount}" if action.type in _SIZED else token


def encode_path(actions: tuple[Action, ...]) -> str:
    """Render a line as the string a caller can store or put in a URL."""
    return SEPARATOR.join(encode_action(action) for action in actions)


def parse_path(path: str) -> tuple[str, ...]:
    """Split a path into tokens, rejecting anything that is not one.

    Validated here rather than during replay so a malformed path fails before any
    game state is built -- the error then points at the text the caller wrote
    instead of at a state they never asked for.
    """
    if not path or path == SEPARATOR:
        return ()
    tokens = tuple(token for token in path.split(SEPARATOR) if token)
    for token in tokens:
        kind = token[0]
        if kind not in _TYPE_BY_TOKEN:
            raise PathError(
                f"'{token}' does not start with an action: expected one of "
                f"{sorted(_TYPE_BY_TOKEN)}."
            )
        size = token[1:]
        if _TYPE_BY_TOKEN[kind] in _SIZED:
            if not size.isdigit():
                raise PathError(f"'{token}' is a sized action and needs an amount, e.g. 'b150'.")
        elif size:
            raise PathError(f"'{token}' takes no amount but carries '{size}'.")
    return tokens


def match_action(token: str, legal: tuple[Action, ...]) -> Action:
    """The legal action ``token`` names, or a refusal listing what was on offer.

    Public because play needs it too: a hand and the tree browser must agree on
    what a token means, and two copies of this matching would drift the moment a
    new action size appeared.
    """
    wanted_type = _TYPE_BY_TOKEN[token[0]]
    wanted_amount = int(token[1:]) if token[1:] else None
    for action in legal:
        if action.type is not wanted_type:
            continue
        if wanted_amount is None or action.amount == wanted_amount:
            return action
    offered = ", ".join(encode_action(action) for action in legal) or "nothing (terminal)"
    raise PathError(f"'{token}' is not available here. On offer: {offered}.")


def placeholder_hole_cards(board: tuple[Card, ...]) -> tuple[tuple[Card, Card], tuple[Card, Card]]:
    """Four cards not on ``board``, to stand in until a real combo is substituted.

    A node's identity does not depend on anyone's hole cards -- but a state
    cannot be built without them, and cards that collided with the board would
    make an impossible state that bucketing would then be asked about.
    """
    dead = {card.mask for card in board}
    free = [card for card in FULL_DECK if card.mask not in dead]
    if len(free) < 4:
        raise PathError(f"A board of {len(board)} cards leaves too few cards to deal hole cards.")
    return ((free[0], free[1]), (free[2], free[3]))


def replay(
    blueprint: ScorableBlueprint,
    path: str,
    board: tuple[Card, ...] = (),
    *,
    button: int = 0,
) -> ReplayedNode:
    """Walk ``path`` from the start of a hand and report where it lands.

    ``board`` is consumed in order as chance nodes are reached, so a path that
    crosses to the flop needs at least three cards. Supplying more than the line
    reaches is allowed and the surplus is ignored -- that is what lets a caller
    hold one runout fixed while walking back and forth along a line.

    Raises :class:`PathError` for a token that is not on offer, and for a board
    too short to reach the street the path asks for. Both are the caller's to
    fix, and both are silent corruption if answered with a default instead.
    """
    rules: GameRules = blueprint.rules
    state = rules.create_initial_state(
        starting_stack=blueprint.config.game.starting_stack,
        hole_cards=placeholder_hole_cards(board),
        button=button,
    )

    consumed = 0
    for token in parse_path(path):
        state, consumed = _advance_chance(state, board, consumed)
        if state.is_terminal:
            raise PathError(f"'{token}' comes after the hand has already ended.")
        legal = rules.get_legal_actions(state, action_model=blueprint.action_model)
        state = state.apply_action(match_action(token, legal), rules)

    state, consumed = _advance_chance(state, board, consumed)
    terminal = state.is_terminal
    return ReplayedNode(
        state=state,
        actor=None if terminal else state.current_player,
        legal_actions=(
            () if terminal else rules.get_legal_actions(state, action_model=blueprint.action_model)
        ),
        board_consumed=consumed,
    )


def _advance_chance(
    state: GameState, board: tuple[Card, ...], consumed: int
) -> tuple[GameState, int]:
    """Deal from ``board`` for as long as the state is waiting on cards."""
    while not state.is_terminal and is_chance_node(state):
        needed = state.street.board_card_count - len(state.board)
        if consumed + needed > len(board):
            raise PathError(
                f"This line reaches {state.street} and needs "
                f"{consumed + needed} board cards, but {len(board)} were given."
            )
        dealt = board[consumed : consumed + needed]
        consumed += needed
        state = state.replace(
            board=(*state.board, *dealt),
            current_player=1 - state.button_position,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
        )
    return state, consumed
