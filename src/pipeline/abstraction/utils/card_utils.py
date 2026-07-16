"""Card rank/suit helpers built on the numeric eval7 accessors.

These deliberately avoid parsing ``repr(card)`` strings, which would break if
the display format ever changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.game.state import Card

# eval7 rank encoding (0=2, ..., 12=A) -> rank character.
_RANK_CHARS = "23456789TJQKA"


def get_rank_char(card: Card) -> str:
    """Rank character of a card: '2'..'9', 'T', 'J', 'Q', 'K', or 'A'."""
    return _RANK_CHARS[card.rank_eval7()]


def cards_have_same_suit(card1: Card, card2: Card) -> bool:
    """True if both cards share a suit."""
    return card1.suit_eval7() == card2.suit_eval7()
