"""
Utility functions for card manipulation.

Provides common operations for extracting rank and suit from cards,
eliminating duplicate code across the abstraction system.
"""

from src.game.state import Card


def get_rank_char(card: Card) -> str:
    """
    Extract rank character from card.

    Args:
        card: Card object

    Returns:
        Rank character ('A', 'K', 'Q', ..., '2')

    Examples:
        >>> card = Card.new('As')
        >>> get_rank_char(card)
        'A'
        >>> card = Card.new('Th')
        >>> get_rank_char(card)
        'T'
    """
    card_str = repr(card)
    # Rank is first character(s)
    # Handle both "A♠" and "10♠"
    if card_str.startswith("10"):
        return "T"
    return card_str[0]


def get_rank_value(card: Card) -> int:
    """
    Extract numeric rank value from card.

    Args:
        card: Card object

    Returns:
        Rank value (2-14, where 14 = Ace)

    Examples:
        >>> card = Card.new('As')
        >>> get_rank_value(card)
        14
        >>> card = Card.new('2h')
        >>> get_rank_value(card)
        2
    """
    rank_char = get_rank_char(card)

    rank_map = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }

    return rank_map[rank_char]


def get_suit(card: Card) -> str:
    """
    Extract suit symbol from card.

    Args:
        card: Card object

    Returns:
        Suit symbol ('♠', '♥', '♦', '♣')

    Examples:
        >>> card = Card.new('As')
        >>> get_suit(card)
        '♠'
        >>> card = Card.new('Th')
        >>> get_suit(card)
        '♥'
    """
    card_str = repr(card)

    # Suit is last character(s)
    # Handle both "A♠" and "10♠"
    if len(card_str) == 2:
        return card_str[1]
    else:
        return card_str[2]


def cards_have_same_suit(card1: Card, card2: Card) -> bool:
    """
    Check if two cards have the same suit.

    Args:
        card1: First card
        card2: Second card

    Returns:
        True if cards have same suit

    Examples:
        >>> card1 = Card.new('As')
        >>> card2 = Card.new('Ks')
        >>> cards_have_same_suit(card1, card2)
        True
    """
    return get_suit(card1) == get_suit(card2)


def cards_have_same_rank(card1: Card, card2: Card) -> bool:
    """
    Check if two cards have the same rank.

    Args:
        card1: First card
        card2: Second card

    Returns:
        True if cards have same rank

    Examples:
        >>> card1 = Card.new('As')
        >>> card2 = Card.new('Ah')
        >>> cards_have_same_rank(card1, card2)
        True
    """
    return get_rank_char(card1) == get_rank_char(card2)
