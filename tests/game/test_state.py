"""Tests for game state and primitives."""

import pytest

from src.game.actions import bet
from src.game.state import Card, GameState, Street


class TestStreet:
    """Tests for Street enum."""

    def test_is_preflop(self):
        assert Street.PREFLOP.is_preflop()
        assert not Street.FLOP.is_preflop()
        assert not Street.TURN.is_preflop()
        assert not Street.RIVER.is_preflop()

    def test_is_postflop(self):
        assert not Street.PREFLOP.is_postflop()
        assert Street.FLOP.is_postflop()
        assert Street.TURN.is_postflop()
        assert Street.RIVER.is_postflop()

    def test_next_street(self):
        assert Street.PREFLOP.next_street() == Street.FLOP
        assert Street.FLOP.next_street() == Street.TURN
        assert Street.TURN.next_street() == Street.RIVER
        assert Street.RIVER.next_street() is None

    def test_str(self):
        assert str(Street.PREFLOP) == "preflop"
        assert str(Street.FLOP) == "flop"
        assert str(Street.TURN) == "turn"
        assert str(Street.RIVER) == "river"


class TestCard:
    """Tests for Card class."""

    def test_create_card(self):
        card = Card.new("As")
        assert card is not None

    def test_card_equality(self):
        card1 = Card.new("As")
        card2 = Card.new("As")
        assert card1 == card2

    def test_card_inequality(self):
        card1 = Card.new("As")
        card2 = Card.new("Kh")
        assert card1 != card2

    def test_card_hash(self):
        card1 = Card.new("As")
        card2 = Card.new("As")
        assert hash(card1) == hash(card2)

    def test_card_repr(self):
        card = Card.new("As")
        assert repr(card) == "As"

    def test_card_ordering(self):
        # Test that cards can be compared (actual ordering depends on treys internals)
        ace = Card.new("As")
        king = Card.new("Kh")
        deuce = Card.new("2s")
        # We just verify comparison works and is consistent
        assert (ace < king) or (king < ace)  # One must be less than the other
        assert deuce < ace or ace < deuce  # Comparison works

    def test_all_ranks(self):
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        for rank in ranks:
            card = Card.new(f"{rank}s")
            assert card is not None

    def test_all_suits(self):
        suits = ["s", "h", "d", "c"]
        for suit in suits:
            card = Card.new(f"A{suit}")
            assert card is not None


class TestGameState:
    """Tests for GameState class."""

    def test_create_basic_state(self):
        state = GameState(
            street=Street.PREFLOP,
            pot=100,
            stacks=(200, 200),
            board=tuple(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=False,
        )
        assert state.street == Street.PREFLOP
        assert state.pot == 100
        assert state.stacks == (200, 200)

    def test_state_validation_invalid_button(self):
        with pytest.raises(ValueError, match="Invalid button_position"):
            GameState(
                street=Street.PREFLOP,
                pot=100,
                stacks=(200, 200),
                board=tuple(),
                hole_cards=(
                    (Card.new("As"), Card.new("Kh")),
                    (Card.new("Qd"), Card.new("Jc")),
                ),
                betting_history=tuple(),
                button_position=2,  # Invalid
                current_player=0,
                is_terminal=False,
            )

    def test_state_validation_negative_stack(self):
        with pytest.raises(ValueError, match="Negative stack"):
            GameState(
                street=Street.PREFLOP,
                pot=100,
                stacks=(-10, 200),  # Invalid
                board=tuple(),
                hole_cards=(
                    (Card.new("As"), Card.new("Kh")),
                    (Card.new("Qd"), Card.new("Jc")),
                ),
                betting_history=tuple(),
                button_position=0,
                current_player=0,
                is_terminal=False,
            )

    def test_state_validation_negative_pot(self):
        with pytest.raises(ValueError, match="Negative pot"):
            GameState(
                street=Street.PREFLOP,
                pot=-10,  # Invalid
                stacks=(200, 200),
                board=tuple(),
                hole_cards=(
                    (Card.new("As"), Card.new("Kh")),
                    (Card.new("Qd"), Card.new("Jc")),
                ),
                betting_history=tuple(),
                button_position=0,
                current_player=0,
                is_terminal=False,
            )

    def test_state_validation_wrong_board_size(self):
        with pytest.raises(ValueError, match="Board should have 3 cards"):
            GameState(
                street=Street.FLOP,  # Flop needs 3 cards
                pot=100,
                stacks=(200, 200),
                board=(Card.new("As"),),  # Only 1 card - invalid
                hole_cards=(
                    (Card.new("Kh"), Card.new("Qd")),
                    (Card.new("Jc"), Card.new("Th")),
                ),
                betting_history=tuple(),
                button_position=0,
                current_player=0,
                is_terminal=False,
            )

    def test_opponent(self):
        state = GameState(
            street=Street.PREFLOP,
            pot=100,
            stacks=(200, 200),
            board=tuple(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=False,
        )
        assert state.opponent(0) == 1
        assert state.opponent(1) == 0

    def test_is_all_in(self):
        state = GameState(
            street=Street.PREFLOP,
            pot=100,
            stacks=(0, 200),  # Player 0 all-in
            board=tuple(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=1,
            is_terminal=False,
        )
        assert state.is_all_in()

    def test_effective_stack(self):
        state = GameState(
            street=Street.PREFLOP,
            pot=100,
            stacks=(150, 200),
            board=tuple(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=False,
        )
        assert state.effective_stack() == 150

    def test_can_check(self):
        state = GameState(
            street=Street.FLOP,
            pot=100,
            stacks=(200, 200),
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=0,
        )
        assert state.can_check()

    def test_cannot_check_facing_bet(self):
        state = GameState(
            street=Street.FLOP,
            pot=100,
            stacks=(200, 200),
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=(bet(50),),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=50,
        )
        assert not state.can_check()

    def test_can_bet(self):
        state = GameState(
            street=Street.FLOP,
            pot=100,
            stacks=(200, 200),
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=0,
        )
        assert state.can_bet()

    def test_can_raise(self):
        state = GameState(
            street=Street.FLOP,
            pot=150,
            stacks=(150, 200),  # Player 0 has chips to raise
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=(bet(50),),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=50,
        )
        assert state.can_raise()

    def test_cannot_raise_not_enough_chips(self):
        state = GameState(
            street=Street.FLOP,
            pot=150,
            stacks=(50, 200),  # Player 0 has exactly to_call chips
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=(bet(50),),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=50,
        )
        assert not state.can_raise()

    def test_state_immutable(self):
        state = GameState(
            street=Street.PREFLOP,
            pot=100,
            stacks=(200, 200),
            board=tuple(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=False,
        )
        with pytest.raises(AttributeError):
            setattr(state, "pot", 200)

    def test_state_str(self):
        state = GameState(
            street=Street.FLOP,
            pot=100,
            stacks=(200, 150),
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=50,
        )
        state_str = str(state)
        assert "P1 to act" in state_str
        assert "flop" in state_str
        assert "Pot: 100" in state_str
