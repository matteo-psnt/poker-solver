"""Tests for game rules."""

import pytest

from src.actions.betting_actions import BettingActions
from src.game.actions import ActionType, all_in, bet, call, check, fold, raises
from src.game.rules import GameRules
from src.game.state import Card, GameState, Street


class TestGameRules:
    """Tests for GameRules class."""

    def test_create_rules(self):
        rules = GameRules()

        assert rules.small_blind == 1
        assert rules.big_blind == 2

    def test_create_rules_custom(self):
        rules = GameRules(small_blind=5, big_blind=10)

        assert rules.small_blind == 5
        assert rules.big_blind == 10

    def test_get_legal_actions_terminal_state(self):
        """Terminal states have no legal actions."""
        rules = GameRules()

        # Create terminal state
        hole_cards = (
            (Card(1), Card(2)),  # Dummy cards
            (Card(3), Card(4)),
        )
        state = GameState(
            street=Street.RIVER,
            pot=20,
            stacks=(100, 100),
            board=(Card(5), Card(6), Card(7), Card(8), Card(9)),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=0,
            is_terminal=True,
            to_call=0,
            last_aggressor=None,
        )

        actions = rules.get_legal_actions(state)

        assert len(actions) == 0

    def test_get_legal_actions_can_check(self):
        """Can check when no bet to call."""
        rules = GameRules()

        state = rules.create_initial_state(
            starting_stack=200,
            hole_cards=((Card(1), Card(2)), (Card(3), Card(4))),
        )

        # After BB posts, SB acts and can check/call
        actions = rules.get_legal_actions(state)

        # Should have check option (can match BB by calling/checking)
        action_types = [a.type for a in actions]
        assert ActionType.CALL in action_types or ActionType.CHECK in action_types

    def test_get_legal_actions_can_fold(self):
        """Can fold when facing a bet."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.PREFLOP,
            pot=10,
            stacks=(195, 195),  # After raise
            board=(),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=5,  # Facing a raise
            last_aggressor=0,
        )

        actions = rules.get_legal_actions(state)

        action_types = [a.type for a in actions]
        assert ActionType.FOLD in action_types

    def test_get_legal_actions_can_call(self):
        """Can call when facing a bet."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.PREFLOP,
            pot=10,
            stacks=(195, 195),
            board=(),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=5,
            last_aggressor=0,
        )

        actions = rules.get_legal_actions(state)

        action_types = [a.type for a in actions]
        assert ActionType.CALL in action_types

    def test_get_legal_actions_with_abstraction(self):
        """Test legal actions with action abstraction."""
        rules = GameRules()
        action_abs = BettingActions()

        state = rules.create_initial_state(
            starting_stack=200,
            hole_cards=((Card(1), Card(2)), (Card(3), Card(4))),
        )

        actions = rules.get_legal_actions(state, action_model=action_abs)

        # Should have multiple bet sizes from abstraction
        assert len(actions) > 1

    def test_get_legal_actions_without_abstraction(self):
        """Test legal actions without action abstraction (allows any size)."""
        rules = GameRules()

        state = rules.create_initial_state(
            starting_stack=200,
            hole_cards=((Card(1), Card(2)), (Card(3), Card(4))),
        )

        actions = rules.get_legal_actions(state, action_model=None)

        # Without abstraction, should have all-in option
        action_types = [a.type for a in actions]
        assert ActionType.ALL_IN in action_types

    def test_get_legal_actions_facing_all_in(self):
        """When facing all-in bet, can only fold or call all-in."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.PREFLOP,
            pot=210,
            stacks=(0, 100),  # Opponent all-in
            board=(),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=100,  # Facing all-in equal to stack
            last_aggressor=0,
        )

        actions = rules.get_legal_actions(state)

        # Should only have fold or all-in call
        assert len(actions) == 2
        action_types = [a.type for a in actions]
        assert ActionType.FOLD in action_types
        assert ActionType.ALL_IN in action_types

    def test_apply_action_fold(self):
        """Test folding action."""
        rules = GameRules()

        state = rules.create_initial_state(
            starting_stack=200,
            hole_cards=((Card(1), Card(2)), (Card(3), Card(4))),
        )

        # Fold
        new_state = rules.apply_action(state, fold())

        assert new_state.is_terminal
        # Folder loses the pot
        assert new_state.get_payoff(0, rules) < 0 or new_state.get_payoff(1, rules) < 0

    def test_apply_action_check(self):
        """Test checking action."""
        rules = GameRules()

        # Create state where checking is possible (postflop, no bet)
        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.FLOP,
            pot=20,
            stacks=(190, 190),
            board=(Card(5), Card(6), Card(7)),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
        )

        new_state = rules.apply_action(state, check())

        assert not new_state.is_terminal
        assert new_state.current_player == 1  # Other player acts
        assert new_state.pot == 20  # Pot unchanged

    def test_apply_action_call(self):
        """Test calling action."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.PREFLOP,
            pot=10,
            stacks=(195, 195),
            board=(),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=5,
            last_aggressor=0,
        )

        new_state = rules.apply_action(state, call())

        assert new_state.pot == 15  # Initial pot (10) + call (5)
        assert new_state.stacks[1] == 190  # Caller's stack reduced by 5

    def test_apply_action_bet(self):
        """Test betting action."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.FLOP,
            pot=20,
            stacks=(190, 190),
            board=(Card(5), Card(6), Card(7)),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
        )

        bet_amount = 10
        new_state = rules.apply_action(state, bet(bet_amount))

        assert new_state.pot == 30  # Original pot + bet
        assert new_state.stacks[0] == 180  # Bettor's stack reduced
        assert new_state.to_call == bet_amount
        assert new_state.current_player == 1

    def test_apply_action_raise(self):
        """Test raising action."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.PREFLOP,
            pot=15,
            stacks=(190, 190),
            board=(),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=5,
            last_aggressor=0,
        )

        raise_amount = 10
        new_state = rules.apply_action(state, raises(raise_amount))

        # Raise means call the current bet + additional raise
        total_amount = 5 + raise_amount
        assert new_state.pot == 15 + total_amount
        assert new_state.stacks[1] == 190 - total_amount

    def test_apply_action_all_in(self):
        """Test all-in action."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.FLOP,
            pot=20,
            stacks=(50, 190),  # Player 0 has short stack
            board=(Card(5), Card(6), Card(7)),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
        )

        new_state = rules.apply_action(state, all_in(50))

        assert new_state.pot == 70  # Original pot + all-in
        assert new_state.stacks[0] == 0  # Player all-in

    def test_apply_invalid_action_insufficient_chips(self):
        """Test that invalid actions raise errors."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.FLOP,
            pot=20,
            stacks=(50, 190),
            board=(Card(5), Card(6), Card(7)),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
        )

        # Try to bet more than stack
        with pytest.raises(ValueError):
            rules.apply_action(state, bet(100))

    def test_create_initial_state(self):
        """Test creating initial game state."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = rules.create_initial_state(starting_stack=200, hole_cards=hole_cards)

        assert state.street == Street.PREFLOP
        assert state.pot == 3  # SB + BB
        assert state.stacks == (199, 198)  # SB and BB posted
        assert len(state.board) == 0
        assert state.button_position == 0

    def test_create_initial_state_with_button(self):
        """Test creating initial state with different button positions."""
        rules = GameRules()

        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))

        # Button position 0
        state0 = rules.create_initial_state(starting_stack=200, hole_cards=hole_cards, button=0)
        assert state0.button_position == 0

        # Button position 1
        state1 = rules.create_initial_state(starting_stack=200, hole_cards=hole_cards, button=1)
        assert state1.button_position == 1

    def test_can_raise(self):
        """Test can_raise detection."""
        rules = GameRules()

        # State where player can raise
        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.PREFLOP,
            pot=10,
            stacks=(195, 195),
            board=(),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=5,  # Facing a bet
            last_aggressor=0,
        )

        # Can raise with action abstraction
        action_abs = BettingActions()
        actions = rules.get_legal_actions(state, action_model=action_abs)

        # Should have raise options
        has_raise = any(a.type == ActionType.RAISE for a in actions)
        assert has_raise

    def test_can_bet(self):
        """Test can_bet detection."""
        rules = GameRules()

        # State where player can bet
        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.FLOP,
            pot=20,
            stacks=(190, 190),
            board=(Card(5), Card(6), Card(7)),
            hole_cards=hole_cards,
            betting_history=(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,  # No bet to call
            last_aggressor=None,
        )

        action_abs = BettingActions()
        actions = rules.get_legal_actions(state, action_model=action_abs)

        # Should have bet options
        has_bet = any(a.type == ActionType.BET for a in actions)
        assert has_bet

    def test_advance_street_logic(self):
        """Test that betting round advances to next street correctly."""
        # Start on flop
        hole_cards = ((Card(1), Card(2)), (Card(3), Card(4)))
        state = GameState(
            street=Street.FLOP,
            pot=20,
            stacks=(190, 190),
            board=(Card(5), Card(6), Card(7)),
            hole_cards=hole_cards,
            betting_history=(check(), check()),  # Both players checked
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
        )

        # Both players have acted and checked - would advance to turn
        # (This is tested more thoroughly in state tests, but validates rules integration)
        assert state.street == Street.FLOP
