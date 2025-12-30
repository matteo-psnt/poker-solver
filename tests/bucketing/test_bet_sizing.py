"""Tests for action abstraction."""

from src.actions.betting_actions import BettingActions
from src.game.actions import ActionType, all_in, bet, call, raises
from src.game.state import Card, GameState, Street
from src.utils.config import ActionAbstractionConfig


class TestActionAbstraction:
    """Tests for BettingActions class."""

    def test_default_config(self):
        abstraction = BettingActions()
        assert abstraction.preflop_raises == [2.5, 3.5, 5.0]
        assert abstraction.postflop_bets == {
            "flop": [0.33, 0.66, 1.25],
            "turn": [0.50, 1.0, 1.5],
            "river": [0.50, 1.0, 2.0],
        }

    def test_get_bet_sizes_preflop(self):
        abstraction = BettingActions()
        state = GameState(
            street=Street.PREFLOP,
            pot=3,
            stacks=(197, 198),
            board=tuple(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
        )

        sizes = abstraction.get_bet_sizes(state)
        # Preflop: 2.5bb = 5 chips, 3.5bb = 7 chips, 5bb = 10 chips
        assert 5 in sizes
        assert 7 in sizes
        assert 10 in sizes

    def test_get_bet_sizes_postflop(self):
        abstraction = BettingActions()
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

        sizes = abstraction.get_bet_sizes(state)
        # Postflop (flop): 33% pot = 33, 66% pot = 66, 125% pot = 125
        assert 33 in sizes
        assert 66 in sizes
        assert 125 in sizes

    def test_get_bet_sizes_facing_bet(self):
        """Cannot bet when facing a bet."""
        abstraction = BettingActions()
        state = GameState(
            street=Street.FLOP,
            pot=150,
            stacks=(150, 200),
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

        sizes = abstraction.get_bet_sizes(state)
        assert sizes == []

    def test_get_raise_sizes_postflop(self):
        abstraction = BettingActions()
        state = GameState(
            street=Street.FLOP,
            pot=150,
            stacks=(150, 200),
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

        sizes = abstraction.get_raise_sizes(state)
        # Postflop raises: 33% pot, 75% pot, all-in
        # Pot is 150, so 33% = 49, 75% = 112
        # Raise sizes (on top of call) should include these
        assert len(sizes) > 0
        # All-in raise = 150 - 50 = 100
        assert 100 in sizes

    def test_get_legal_actions_can_check(self):
        abstraction = BettingActions()
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

        actions = abstraction.get_legal_actions(state)

        # Should have check and multiple bet sizes
        assert any(a.type == ActionType.CHECK for a in actions)
        assert any(a.type == ActionType.BET for a in actions)
        assert not any(a.type == ActionType.FOLD for a in actions)  # Can check for free

    def test_get_legal_actions_facing_bet(self):
        abstraction = BettingActions()
        state = GameState(
            street=Street.FLOP,
            pot=150,
            stacks=(150, 200),
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

        actions = abstraction.get_legal_actions(state)

        # Should have fold, call, and raises
        assert any(a.type == ActionType.FOLD for a in actions)
        assert any(a.type == ActionType.CALL for a in actions)
        # Should have some raise options or all-in
        assert any(a.is_aggressive() for a in actions)

    def test_discretize_bet_action(self):
        abstraction = BettingActions()
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

        # Try to bet 40 chips (not in abstraction)
        raw_action = bet(40)
        discretized = abstraction.discretize_action(state, raw_action)

        # Should map to nearest abstracted bet (33 or 75)
        assert discretized.type in (ActionType.BET, ActionType.ALL_IN)
        # Should be one of the abstracted sizes
        legal_actions = abstraction.get_legal_actions(state)
        assert discretized in legal_actions

    def test_str_representation(self):
        abstraction = BettingActions()
        s = str(abstraction)
        assert "BettingActions" in s
        assert "preflop" in s.lower()
        assert "postflop" in s.lower()

    def test_init_validation_errors(self):
        try:
            BettingActions(config="bad")  # type: ignore[arg-type]
            assert False, "Expected TypeError"
        except TypeError:
            pass

        try:
            BettingActions(
                ActionAbstractionConfig(
                    postflop={"flop": [0.33], "turn": [0.5]},
                )
            )
            assert False, "Expected ValueError for missing river config"
        except ValueError:
            pass

        try:
            BettingActions(
                ActionAbstractionConfig(
                    preflop_raises=[],
                )
            )
            assert False, "Expected ValueError for empty preflop raises"
        except ValueError:
            pass

        try:
            BettingActions(
                ActionAbstractionConfig(
                    postflop={"flop": [], "turn": [0.5], "river": [0.5]},
                )
            )
            assert False, "Expected ValueError for empty street sizes"
        except ValueError:
            pass

    def test_bet_sizes_respect_raise_cap(self):
        abstraction = BettingActions(ActionAbstractionConfig(max_raises_per_street=0))
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
            current_player=0,
            is_terminal=False,
            to_call=0,
        )
        assert abstraction.get_bet_sizes(state) == []

    def test_raise_sizes_when_not_facing_bet(self):
        abstraction = BettingActions()
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
            current_player=0,
            is_terminal=False,
            to_call=0,
        )
        assert abstraction.get_raise_sizes(state) == []

    def test_legal_actions_terminal_state(self):
        abstraction = BettingActions()
        state = GameState(
            street=Street.RIVER,
            pot=100,
            stacks=(0, 0),
            board=(
                Card.new("As"),
                Card.new("Kh"),
                Card.new("Qd"),
                Card.new("Jc"),
                Card.new("Th"),
            ),
            hole_cards=(
                (Card.new("2c"), Card.new("3c")),
                (Card.new("4d"), Card.new("5d")),
            ),
            betting_history=(all_in(100), call()),
            button_position=0,
            current_player=0,
            is_terminal=True,
            to_call=0,
            _skip_validation=True,
        )
        assert abstraction.get_legal_actions(state) == []

    def test_discretize_passthrough_for_non_aggressive_actions(self):
        abstraction = BettingActions()
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
            current_player=0,
            is_terminal=False,
            to_call=50,
        )
        assert abstraction.discretize_action(state, call()) == call()

    def test_discretize_bet_without_bet_actions_falls_back_to_all_in(self):
        abstraction = BettingActions()
        state = GameState(
            street=Street.FLOP,
            pot=150,
            stacks=(150, 200),
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
        assert abstraction.discretize_action(state, bet(20)).type == ActionType.ALL_IN

    def test_discretize_raise_without_raise_actions_falls_back_to_all_in(self):
        abstraction = BettingActions()
        state = GameState(
            street=Street.FLOP,
            pot=300,
            stacks=(40, 200),
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=(bet(80),),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=80,
            _skip_validation=True,
        )
        assert abstraction.discretize_action(state, raises(10)).type == ActionType.ALL_IN

    def test_discretize_all_in_branches(self):
        abstraction = BettingActions()

        # Existing all-in action available.
        facing_all_in = GameState(
            street=Street.FLOP,
            pot=300,
            stacks=(40, 200),
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=(bet(80),),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=80,
            _skip_validation=True,
        )
        assert abstraction.discretize_action(facing_all_in, all_in(40)).type == ActionType.ALL_IN

        # No all-in action, but aggressive actions exist -> return max aggressive.
        open_state = GameState(
            street=Street.FLOP,
            pot=100,
            stacks=(400, 400),
            board=(Card.new("As"), Card.new("Kh"), Card.new("Qd")),
            hole_cards=(
                (Card.new("Jc"), Card.new("Th")),
                (Card.new("9s"), Card.new("8h")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
        )
        mapped = abstraction.discretize_action(open_state, all_in(400))
        assert mapped.is_aggressive()

        # No legal actions at all -> return original action.
        terminal_state = GameState(
            street=Street.RIVER,
            pot=0,
            stacks=(0, 0),
            board=(
                Card.new("As"),
                Card.new("Kh"),
                Card.new("Qd"),
                Card.new("Jc"),
                Card.new("Th"),
            ),
            hole_cards=(
                (Card.new("2c"), Card.new("3c")),
                (Card.new("4d"), Card.new("5d")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=True,
            to_call=0,
            _skip_validation=True,
        )
        action = all_in(1)
        assert abstraction.discretize_action(terminal_state, action) == action
