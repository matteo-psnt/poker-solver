"""Turning a GTO Wizard frame into an infoset, and a blueprint's answer into a wire action.

Everything here is a pure function of one :class:`Frame`. Nothing is carried
between turns on purpose: their frame ships the whole ``action_history``, so a
client that reconnects mid-hand rebuilds the same spot from the same bytes, and
there is no session state to get stale. It also makes the interesting half
testable without an API key -- which is the only way it is testable at all,
since the key is granted by hand and the endpoints that play are all gated.

Why this package reaches into the engine when ``interfaces.cloud`` may not:
dispatching work and *playing a hand* are different jobs. The cloud package
queues tasks and must not know how a hand is solved; this one is the deployed
player, and naming an infoset is the entire thing it does.

Two rules keep the two chip currencies from contaminating each other:

- Their numbers price the bet. :class:`~src.interfaces.gtowizard.protocol.Turn`
  is authoritative for the pot, the stacks and ``raise_range``, and every amount
  we send back is derived from those.
- Our numbers name the infoset. The reconstructed ``GameState`` exists to be
  looked up in the tree and for nothing else, so an opponent bet we had to snap
  on-tree costs us a slightly wrong *label*, never a mis-sized wager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.game.actions import Action, ActionType
from src.core.game.state import FULL_DECK, Card, GameState
from src.engine.search.action_translation import translate_action_distribution
from src.interfaces.gtowizard.protocol import BET, CALL, CHECK, FOLD, ProtocolError, Turn
from src.pipeline.blueprint.paths import advance_chance

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.rules import GameRules
    from src.engine.solver.policy.source import ScorableBlueprint
    from src.interfaces.gtowizard.protocol import Frame, Game


class AdapterError(ValueError):
    """A frame cannot be mapped onto this blueprint's game.

    Distinct from :class:`~src.interfaces.gtowizard.protocol.ProtocolError`: the
    payload was well formed, but it describes a table this blueprint was not
    trained for. The caller's only sane response is a safe default, so the
    reason has to survive as a sentence rather than as a None.
    """


@dataclass(frozen=True)
class TableScale:
    """Chips at their table per one chip of ours, fixed for a game.

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


def table_scale(game: Game, blueprint: ScorableBlueprint) -> TableScale:
    """How their chips map onto the blueprint's, or why they do not.

    Refuses rather than approximates. A blind ratio that is not a whole number
    would put every bet size between two of our tree's rungs, and a depth we did
    not train is a different game -- both are worth a forfeited run far less
    than they are worth a line saying which one happened. **Their game is 200 bb
    and every blueprint we have is cut for 100**, so this is the guard that
    fires until a 200 bb arm exists.
    """
    ours = blueprint.config.game
    if game.big_blind % ours.big_blind or game.small_blind % ours.small_blind:
        raise AdapterError(
            f"Table blinds {game.small_blind}/{game.big_blind} are not a whole "
            f"multiple of the blueprint's {ours.small_blind}/{ours.big_blind}."
        )
    factor = game.big_blind // ours.big_blind
    if game.small_blind // ours.small_blind != factor:
        raise AdapterError(
            f"Table blinds {game.small_blind}/{game.big_blind} scale unevenly "
            f"against the blueprint's {ours.small_blind}/{ours.big_blind}."
        )
    return TableScale(
        factor=factor,
        their_depth=game.depth_in_blinds,
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


@dataclass(frozen=True)
class Spot:
    """One of their frames, resolved onto our tree."""

    state: GameState
    scale: TableScale
    seat: int
    off_tree: int
    truncated: bool


def reconstruct(blueprint: ScorableBlueprint, frame: Frame, scale: TableScale) -> Spot:
    """Replay their history into one of our states, in our chips.

    Their history carries no seats -- it is a flat list of ``f``/``c``/``k``/
    ``bX`` with ``_`` between streets -- so the actor is derived the same way
    :meth:`Turn.button_seat` derives the button: heads-up seats strictly
    alternate within a round, starting with the button preflop and with the big
    blind on every later street.

    Off-tree opponent sizes are snapped to the likeliest legal action rather
    than sampled: a stable wrong mapping replays identically on the next turn of
    the same hand, and a sampled one would move the spot under us mid-hand.

    ``truncated`` marks a history our tree ran out of room for -- their betting
    can outlive our action model's raise cap, and when it does the state we
    return is the deepest one we could reach rather than the real one.
    """
    turn = frame.turn
    rules: GameRules = blueprint.rules
    action_model: ActionModel = blueprint.action_model
    seat = turn.hero_seat
    button = turn.button_seat()

    hero_cards = turn.players[seat].hole_cards
    if len(hero_cards) != 2:
        raise ProtocolError(f"Expected 2 hole cards, got {len(hero_cards)}.")
    theirs = opponent_hole_cards((*turn.board, *hero_cards))
    hole = (
        ((hero_cards[0], hero_cards[1]), theirs)
        if seat == 0
        else (theirs, (hero_cards[0], hero_cards[1]))
    )

    state = rules.create_initial_state(
        starting_stack=blueprint.config.game.starting_stack,
        hole_cards=hole,
        button=button,
    )

    board = turn.board
    consumed = 0
    off_tree = 0
    truncated = False
    turn.street_index()  # raises if the board and the history disagree

    for street, tokens in enumerate(turn.rounds):
        # Preflop the blinds are already in, and a raise is priced against the
        # big blind rather than against nothing.
        wagers = [0, 0] if street else _blind_wagers(frame.game, button)
        actor = button if street == 0 else 1 - button
        for token in tokens:
            state, consumed = advance_chance(state, board, consumed)
            if state.is_terminal:
                truncated = True
                break
            legal = rules.get_legal_actions(state, action_model=action_model)
            if not legal:
                truncated = True
                break
            observed, snapped = _observed_action(token, wagers, state, scale, legal)
            chosen = _on_tree(observed, state, action_model, rules, legal)
            off_tree += int(snapped or chosen != observed)
            state = state.apply_action(chosen, rules)
            wagers[actor] = _wager_after(token, wagers, actor)
            actor = 1 - actor
        if truncated:
            break

    if not truncated:
        # Only past a COMPLETE replay: a `break` above left betting unfinished,
        # and advancing here would deal the turn and river into a state whose
        # flop never closed.
        state, consumed = advance_chance(state, board, consumed)
    if not truncated and (state.is_terminal or state.current_player != seat):
        # Their frame says it is our turn; if ours disagrees the replay drifted,
        # which is worth knowing about rather than answering from the wrong seat.
        truncated = True
    return Spot(state=state, scale=scale, seat=seat, off_tree=off_tree, truncated=truncated)


def _blind_wagers(game: Game, button: int) -> list[int]:
    """Blinds already posted, by seat, in their chips. Heads-up the button is the SB."""
    wagers = [0, 0]
    wagers[button] = game.small_blind
    wagers[1 - button] = game.big_blind
    return wagers


def _wager_after(token: str, wagers: list[int], actor: int) -> int:
    """The actor's cumulative round wager once ``token`` has been applied."""
    if token.startswith(BET):
        return int(token[1:])
    if token == CALL:
        return max(wagers)
    return wagers[actor]


def _observed_action(
    token: str,
    wagers: list[int],
    state: GameState,
    scale: TableScale,
    legal: tuple[Action, ...],
) -> tuple[Action, bool]:
    """Their action as one of ours, plus whether the size had to be approximated.

    A ``bX`` is read as a total round wager, converted to our chips, and
    expressed as the increment above the current high bet -- which is what
    :attr:`Action.amount` means for a RAISE and what a BET's total collapses to
    when nothing is yet in.
    """
    if token == FOLD:
        return Action(ActionType.FOLD), False
    if token == CHECK:
        return Action(ActionType.CHECK), False
    if token == CALL:
        return Action(ActionType.CALL), False
    if not token.startswith(BET) or not token[1:].isdigit():
        raise ProtocolError(f"'{token}' is not an action a seat can take.")

    high = max(wagers)
    increment = scale.to_ours(int(token[1:])) - scale.to_ours(high)
    if increment <= 0:
        # Their raise rounds to no increment in our coarser chips: it is a call
        # as far as our tree can tell.
        return Action(ActionType.CALL), True
    kind = ActionType.BET if high == 0 else ActionType.RAISE
    amount = increment if kind is ActionType.RAISE else scale.to_ours(int(token[1:]))
    # What the actor must actually part with: a RAISE's amount sits ON TOP of
    # the call, so comparing the bare amount to the stack would miss a shove by
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


def round_wagers(turn: Turn, game: Game) -> list[int]:
    """Each seat's CUMULATIVE wager for the current round, in their chips.

    Replayed from the round's own tokens rather than read off a field, because
    their frame carries no per-seat wager: it reports ``common_pot`` and
    ``total_pot`` and leaves the split implicit. The blinds open the preflop
    round; every later round opens at nothing.
    """
    button = turn.button_seat()
    wagers = _blind_wagers(game, button) if turn.street == "preflop" else [0, 0]
    actor = button if turn.street == "preflop" else 1 - button
    for token in turn.rounds[-1]:
        wagers[actor] = _wager_after(token, wagers, actor)
        actor = 1 - actor
    return wagers


def wire_amount(chosen: Action, turn: Turn, game: Game, spot: Spot) -> int:
    """Our aggressive action as a CUMULATIVE round wager in their chips.

    A raise-to total is the round's current high wager plus what we are adding,
    so it is built from THEIR chips throughout and only the increment crosses
    the scale. Clamped into ``raise_range`` last, so a replay that drifted costs
    us a slightly odd size and never a rejected request. An all-in is their
    ``raise_max``, since their wire has no all-in.
    """
    if turn.raise_max <= 0:
        raise AdapterError("The blueprint wants to bet where betting is not offered.")
    if chosen.type is ActionType.ALL_IN:
        return turn.raise_max
    high = max(round_wagers(turn, game))
    total = high + spot.scale.to_theirs(chosen.amount)
    return max(turn.raise_min, min(total, turn.raise_max))
