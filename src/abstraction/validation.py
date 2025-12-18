"""
Input validation utilities for card abstraction.

Provides validation functions for hole cards, board cards, and street compatibility.
"""

from typing import Tuple

from src.game.state import Card, Street


class ValidationError(ValueError):
    """Raised when validation fails."""

    pass


def validate_hole_cards(hole_cards: Tuple[Card, ...]) -> None:
    """
    Validate hole cards.

    Args:
        hole_cards: Tuple of hole cards

    Raises:
        ValidationError: If hole cards are invalid
    """
    if not isinstance(hole_cards, tuple):
        raise ValidationError(f"Hole cards must be a tuple, got {type(hole_cards)}")

    if len(hole_cards) != 2:
        raise ValidationError(f"Expected 2 hole cards, got {len(hole_cards)}")

    if hole_cards[0] == hole_cards[1]:
        raise ValidationError(f"Duplicate hole cards: {hole_cards[0]} appears twice")


def validate_board(board: Tuple[Card, ...], street: Street) -> None:
    """
    Validate board cards for a given street.

    Args:
        board: Tuple of board cards
        street: Current street

    Raises:
        ValidationError: If board is invalid for the street
    """
    if not isinstance(board, tuple):
        raise ValidationError(f"Board must be a tuple, got {type(board)}")

    expected_length = {
        Street.PREFLOP: 0,
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
    }

    expected = expected_length[street]
    actual = len(board)

    if actual != expected:
        raise ValidationError(f"Expected {expected} board cards for {street.name}, got {actual}")

    # Check for duplicates
    seen = set()
    for card in board:
        if card in seen:
            raise ValidationError(f"Duplicate board card: {card}")
        seen.add(card)


def validate_cards_compatible(hole_cards: Tuple[Card, ...], board: Tuple[Card, ...]) -> None:
    """
    Validate that hole cards don't conflict with board cards.

    Args:
        hole_cards: Tuple of hole cards
        board: Tuple of board cards

    Raises:
        ValidationError: If there are duplicate cards between hole cards and board
    """
    all_cards = list(hole_cards) + list(board)

    seen = set()
    for card in all_cards:
        if card in seen:
            raise ValidationError(f"Card {card} appears in both hole cards and board")
        seen.add(card)


def validate_equity_inputs(
    hole_cards: Tuple[Card, ...], board: Tuple[Card, ...], street: Street
) -> None:
    """
    Validate all inputs for equity calculation.

    Args:
        hole_cards: Tuple of hole cards
        board: Tuple of board cards
        street: Current street

    Raises:
        ValidationError: If any inputs are invalid
    """
    validate_hole_cards(hole_cards)
    validate_board(board, street)
    validate_cards_compatible(hole_cards, board)


def validate_bucket_inputs(
    hole_cards: Tuple[Card, ...], board: Tuple[Card, ...], street: Street
) -> None:
    """
    Validate all inputs for bucketing.

    Args:
        hole_cards: Tuple of hole cards
        board: Tuple of board cards
        street: Current street

    Raises:
        ValidationError: If any inputs are invalid
    """
    # Same validation as equity
    validate_equity_inputs(hole_cards, board, street)

    # Additional check: postflop bucketing requires postflop street
    if street == Street.PREFLOP:
        raise ValidationError("Cannot bucket preflop - use preflop hand string instead")
