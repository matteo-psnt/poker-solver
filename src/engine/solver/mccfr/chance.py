"""Chance-node and board-dealing helpers for MCCFR solver."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.core.game.state import FULL_DECK, Card, GameState, Street

if TYPE_CHECKING:
    from .solver import MCCFRSolver


def deal_initial_state(self: MCCFRSolver) -> GameState:
    """Deal random cards and create the initial game state."""
    cards = random.sample(FULL_DECK, 4)
    hole_cards = ((cards[0], cards[1]), (cards[2], cards[3]))

    return self.rules.create_initial_state(
        starting_stack=self.config.game.starting_stack,
        hole_cards=hole_cards,
        button=(self.iteration // 2) % 2,
    )


def is_chance_node(self: MCCFRSolver, state: GameState) -> bool:
    """Check if state still needs chance-card dealing."""
    return len(state.board) < state.street.board_card_count


def draw_cards(state: GameState, count: int) -> list[Card]:
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
    randrange = random.randrange
    while len(drawn) < count:
        card = FULL_DECK[randrange(52)]
        if card.mask & known:
            continue
        known |= card.mask
        drawn.append(card)
    return drawn


def sample_chance_outcome(self: MCCFRSolver, state: GameState) -> GameState:
    """Sample the next chance outcome (flop/turn/river card deals)."""
    board_size = len(state.board)

    if state.street == Street.FLOP and board_size == 0:
        count = 3
    elif state.street == Street.TURN and board_size == 3:
        count = 1
    elif state.street == Street.RIVER and board_size == 4:
        count = 1
    else:
        # is_chance_node reported this state needs dealing, but (street, board_size)
        # is not a legal deal point. Returning ``state`` unchanged would spin the
        # caller's "while chance node" loop forever; fail loudly on the malformed state.
        raise ValueError(f"Unexpected chance state: street={state.street}, board_size={board_size}")

    first_to_act = 1 - state.button_position

    return state.replace(
        board=(*state.board, *draw_cards(state, count)),
        current_player=first_to_act,
        is_terminal=False,
        to_call=0,
        last_aggressor=None,
    )


def deal_remaining_cards(self: MCCFRSolver, state: GameState) -> GameState:
    """Deal all remaining board cards for terminal all-in showdowns."""
    cards_needed = 5 - len(state.board)

    return state.replace(
        street=Street.RIVER,
        board=(*state.board, *draw_cards(state, cards_needed)),
        is_terminal=True,
        to_call=0,
    )
