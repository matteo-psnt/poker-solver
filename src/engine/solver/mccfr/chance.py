"""Chance-node and board-dealing helpers for MCCFR solver."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.core.game.state import Card, GameState, Street

if TYPE_CHECKING:
    from .solver import MCCFRSolver


def deal_initial_state(self: MCCFRSolver) -> GameState:
    """Deal random cards and create the initial game state."""
    self._deck.cards = list(self._deck_cards)
    random.shuffle(self._deck.cards)

    hole_cards = (
        (Card(self._deck.cards[0]), Card(self._deck.cards[1])),
        (Card(self._deck.cards[2]), Card(self._deck.cards[3])),
    )

    return self.rules.create_initial_state(
        starting_stack=self.config.game.starting_stack,
        hole_cards=hole_cards,
        button=(self.iteration // 2) % 2,
    )


def is_chance_node(self: MCCFRSolver, state: GameState) -> bool:
    """Check if state still needs chance-card dealing."""
    return len(state.board) < state.street.board_card_count


def prepare_shuffled_deck(self: MCCFRSolver, state: GameState) -> None:
    """Prepare a shuffled deck without known cards."""
    self._deck.cards = list(self._deck_cards)

    known_masks: set[int] = set()
    for player_cards in state.hole_cards:
        for card in player_cards:
            known_masks.add(card.mask)
    for card in state.board:
        known_masks.add(card.mask)

    self._deck.cards = [card for card in self._deck.cards if card.mask not in known_masks]
    random.shuffle(self._deck.cards)


def sample_chance_outcome(self: MCCFRSolver, state: GameState) -> GameState:
    """Sample the next chance outcome (flop/turn/river card deals)."""
    prepare_shuffled_deck(self, state)

    new_board = list(state.board)
    board_size = len(state.board)

    if state.street == Street.FLOP and board_size == 0:
        new_board.extend([Card(card) for card in self._deck.cards[:3]])
    elif state.street == Street.TURN and board_size == 3:
        new_board.append(Card(self._deck.cards[0]))
    elif state.street == Street.RIVER and board_size == 4:
        new_board.append(Card(self._deck.cards[0]))
    else:
        return state

    first_to_act = 1 - state.button_position

    return state.replace(
        board=tuple(new_board),
        current_player=first_to_act,
        is_terminal=False,
        to_call=0,
        last_aggressor=None,
    )


def deal_remaining_cards(self: MCCFRSolver, state: GameState) -> GameState:
    """Deal all remaining board cards for terminal all-in showdowns."""
    prepare_shuffled_deck(self, state)

    new_board = list(state.board)
    cards_needed = 5 - len(state.board)
    new_board.extend([Card(card) for card in self._deck.cards[:cards_needed]])

    return state.replace(
        street=Street.RIVER,
        board=tuple(new_board),
        is_terminal=True,
        to_call=0,
    )
