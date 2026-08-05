"""Chance-node and board-dealing helpers for MCCFR solver.

Only :func:`deal_initial_state` needs the solver (stack, rules, button
alternation); the rest are pure functions of the state.

Where the randomness comes from
-------------------------------
Every dealing function takes an optional ``rng``. ``None`` -- the default, and
what training passes -- draws from the ``random`` module's process-global
instance, exactly as before: same functions, same call order, so the training
stream is unchanged and no golden number moves.

Passing an explicit :class:`random.Random` matters for anything that deals more
than one hand at a time. A server holding several play sessions would otherwise
have them all draw from one global, which makes no session reproducible from its
own seed and interleaves their draws by request timing. The parameter is
threaded rather than the global reseeded because reseeding is not composable:
two sessions doing it race, and the loser silently inherits the winner's stream.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.core.game.state import FULL_DECK, Card, GameState, Street

if TYPE_CHECKING:
    from .solver import MCCFRSolver


def deal_initial_state(self: MCCFRSolver, rng: random.Random | None = None) -> GameState:
    """Deal random cards and create the initial game state."""
    cards = random.sample(FULL_DECK, 4) if rng is None else rng.sample(FULL_DECK, 4)
    hole_cards = ((cards[0], cards[1]), (cards[2], cards[3]))

    return self.rules.create_initial_state(
        starting_stack=self.config.game.starting_stack,
        hole_cards=hole_cards,
        button=(self.iteration // 2) % 2,
    )


def is_chance_node(state: GameState) -> bool:
    """Check if state still needs chance-card dealing."""
    return len(state.board) < state.street.board_card_count


def draw_cards(state: GameState, count: int, rng: random.Random | None = None) -> list[Card]:
    """Draw ``count`` uniform random cards not present in the state.

    Rejection sampling against a bitmask of known cards: with at most 9 of 52
    cards known, acceptance is ~83%+, which beats rebuilding and shuffling the
    remaining deck by an order of magnitude.
    """
    known = 0
    for player_cards in state.hole_cards:
        for card in player_cards:
            known |= card.mask
    for card in state.board:
        known |= card.mask

    drawn: list[Card] = []
    randrange = random.randrange if rng is None else rng.randrange
    while len(drawn) < count:
        card = FULL_DECK[randrange(52)]
        if card.mask & known:
            continue
        known |= card.mask
        drawn.append(card)
    return drawn


def sample_chance_outcome(state: GameState, rng: random.Random | None = None) -> GameState:
    """Sample the next chance outcome (flop/turn/river card deals)."""
    board_size = len(state.board)

    if state.street == Street.FLOP and board_size == 0:
        count = 3
    elif (state.street == Street.TURN and board_size == 3) or (
        state.street == Street.RIVER and board_size == 4
    ):
        count = 1
    else:
        # is_chance_node reported this state needs dealing, but (street, board_size)
        # is not a legal deal point. Returning ``state`` unchanged would spin the
        # caller's "while chance node" loop forever; fail loudly on the malformed state.
        raise ValueError(f"Unexpected chance state: street={state.street}, board_size={board_size}")

    first_to_act = 1 - state.button_position

    return state.replace(
        board=(*state.board, *draw_cards(state, count, rng)),
        current_player=first_to_act,
        is_terminal=False,
        to_call=0,
        last_aggressor=None,
    )


def deal_remaining_cards(state: GameState, rng: random.Random | None = None) -> GameState:
    """Deal all remaining board cards for terminal all-in showdowns.

    A complete board returns ``state`` unchanged, so callers can run every
    terminal through here instead of repeating the board-length test. No caller
    ever passed a complete board before this became the contract.

    Folds are returned unchanged for the same reason: ``get_payoff`` resolves a
    fold from ``ended_by_fold`` and the folder's identity, never from the board,
    so completing the runout cannot change the payoff by a single chip. It is
    pure work — and measured at 34% of ALL terminal evaluations, because a fold
    is by far the most common way a hand ends.
    """
    if state.ended_by_fold:
        return state

    cards_needed = 5 - len(state.board)
    if cards_needed <= 0:
        return state

    return state.replace(
        street=Street.RIVER,
        board=(*state.board, *draw_cards(state, cards_needed, rng)),
        is_terminal=True,
        to_call=0,
    )
