"""Tests for Card public accessors that replace private _card usage."""

from src.core.game.state import Card


def test_card_eval7_accessors():
    card = Card.new("As")
    eval7_card = card.to_eval7()

    assert eval7_card.mask == card.mask
    assert card.rank_eval7() == eval7_card.rank
    assert card.suit_eval7() == eval7_card.suit
