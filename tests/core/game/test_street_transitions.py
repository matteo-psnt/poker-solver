"""
Tests for street transition logic.

Verifies that streets advance correctly based on betting actions,
specifically the fix for check-check detection.
"""

from src.core.game.actions import bet, call, check, raises
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street


class TestStreetTransitions:
    """Test street advancement logic."""

    def test_single_check_does_not_advance_street(self):
        """A single check should not advance the street."""
        rules = GameRules()

        state = GameState(
            street=Street.FLOP,
            pot=200,
            stacks=(900, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,  # Out of position acts first
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            betting_history=(),
        )

        # Player 1 checks
        new_state = rules.apply_action(state, check())

        assert new_state.street == Street.FLOP, "Should still be on flop"
        assert new_state.current_player == 0, "Should be player 0's turn"

    def test_check_check_advances_street(self):
        """Check-check should advance to next street."""
        rules = GameRules()

        state = GameState(
            street=Street.FLOP,
            pot=200,
            stacks=(900, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            betting_history=(),
        )

        # Player 1 checks
        state = rules.apply_action(state, check())
        # Player 0 checks
        state = rules.apply_action(state, check())

        assert state.street == Street.TURN, "Should advance to turn after check-check"

    def test_call_advances_street(self):
        """Calling a bet should advance the street."""
        rules = GameRules()

        state = GameState(
            street=Street.FLOP,
            pot=200,
            stacks=(850, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=50,
            last_aggressor=0,
            betting_history=(bet(50),),
        )

        # Player 1 calls
        new_state = rules.apply_action(state, call())

        assert new_state.street == Street.TURN, "Should advance to turn after call"

    def test_bet_check_check_advances_correctly(self):
        """
        Sequence: bet -> call should advance to turn.

        This test verifies that after a call, the street advances.
        We can't test further without dealing cards (which MCCFR handles).
        """
        rules = GameRules()

        # Flop: Player 1 faces a bet
        state = GameState(
            street=Street.FLOP,
            pot=200,
            stacks=(850, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=50,
            last_aggressor=0,
            betting_history=(bet(50),),
        )

        # Player 1 calls → should advance to turn
        state = rules.apply_action(state, call())
        assert state.street == Street.TURN, "Should be on turn after call"

    def test_check_check_advances_from_flop(self):
        """Verify check-check advances from flop to turn."""
        rules = GameRules()

        state = GameState(
            street=Street.FLOP,
            pot=200,
            stacks=(900, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            betting_history=(),
        )

        # Flop: check-check should advance to turn
        state = rules.apply_action(state, check())
        state = rules.apply_action(state, check())
        assert state.street == Street.TURN, "Should advance to turn after flop check-check"

    def test_check_check_on_river_goes_to_showdown(self):
        """Check-check on river should create terminal state."""
        rules = GameRules()

        state = GameState(
            street=Street.RIVER,
            pot=200,
            stacks=(900, 900),
            board=(Card.new("Ah"), Card.new("Kh"), Card.new("Qh"), Card.new("Jd"), Card.new("Td")),
            hole_cards=((Card.new("2c"), Card.new("3c")), (Card.new("4d"), Card.new("5d"))),
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            betting_history=(),
        )

        # River: check-check goes to showdown
        state = rules.apply_action(state, check())
        state = rules.apply_action(state, check())
        assert state.is_terminal, "Should be terminal after river check-check"

    def test_bet_raise_call_advances(self):
        """Bet -> raise -> call should advance street."""
        rules = GameRules()

        state = GameState(
            street=Street.FLOP,
            pot=200,
            stacks=(850, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=50,
            last_aggressor=0,
            betting_history=(bet(50),),
        )

        # Player 1 raises
        state = rules.apply_action(state, raises(100))
        assert state.street == Street.FLOP, "Should still be on flop after raise"

        # Player 0 calls
        state = rules.apply_action(state, call())
        assert state.street == Street.TURN, "Should advance to turn after call"

    def test_get_actions_on_current_street_isolated(self):
        """Test the _get_actions_on_current_street helper directly."""
        rules = GameRules()

        # Test with check-check on previous street, then bet on current
        betting_history = [check(), check(), bet(50)]

        actions = rules._get_actions_on_current_street(betting_history)

        assert len(actions) == 1, "Should only have bet from current street"
        assert actions[0].type.name == "BET"

    def test_get_actions_on_current_street_after_call(self):
        """Test street detection after a call."""
        rules = GameRules()

        # Previous street: bet, call
        # Current street: bet
        betting_history = [bet(50), call(), bet(75)]

        actions = rules._get_actions_on_current_street(betting_history)

        assert len(actions) == 1, "Should only have current street bet"
        assert actions[0].amount == 75

    def test_consecutive_check_check_streets_do_not_straddle(self):
        """Two streets that both end check-check must each be isolated.

        Regression: a backward scan mis-matched the boundary-straddling pair
        (previous street's last check + current street's first check), so the
        current street's second check saw only itself and could not advance.
        Here betting_history is CALL (preflop limp) + flop check-check + the
        two turn checks; the current (turn) street must contain both checks.
        """
        rules = GameRules()

        # Preflop limp-call, flop check-check, turn check-check (second check pending).
        betting_history = [call(), check(), check(), check(), check()]

        actions = rules._get_actions_on_current_street(betting_history)

        assert [a.type.name for a in actions] == ["CHECK", "CHECK"], (
            "Current (turn) street must contain both turn checks so it can advance"
        )

    def test_full_check_down_hand_reaches_showdown(self):
        """End-to-end: a pure check-down driven through the real engine must
        terminate, not loop forever on the turn.

        This mirrors the MCCFR traversal path (apply_action + external board
        dealing) from a real ``create_initial_state``. Before the fix this hung
        on the turn (perpetual check-check that never advanced), affecting both
        evaluation and training. A step cap turns any regression into a clean
        assertion failure instead of a real hang.
        """
        rules = GameRules()
        board = [Card.new(c) for c in ("2c", "7d", "Th", "Js", "Ac")]
        expected_board = {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
        }

        hole = (
            (Card.new("Kd"), Card.new("Qc")),
            (Card.new("8s"), Card.new("9h")),
        )
        state = rules.create_initial_state(starting_stack=100, hole_cards=hole, button=0)

        streets_seen: list[Street] = []
        for _ in range(40):  # generous cap; a real check-down needs ~9 steps
            if state.is_terminal:
                break
            if len(state.board) < expected_board[state.street]:
                # Deal the next board cards exactly as the traversal layer does.
                state = GameState(
                    street=state.street,
                    pot=state.pot,
                    stacks=state.stacks,
                    board=tuple(board[: expected_board[state.street]]),
                    hole_cards=state.hole_cards,
                    betting_history=state.betting_history,
                    button_position=state.button_position,
                    current_player=1 - state.button_position,
                    is_terminal=False,
                    to_call=0,
                    last_aggressor=None,
                )
                continue
            streets_seen.append(state.street)
            # Pure check-down: check when possible, else limp the preflop blind.
            action = check() if state.to_call == 0 else call()
            state = rules.apply_action(state, action)

        assert state.is_terminal, "Pure check-down must reach a terminal state, not hang"
        assert state.street == Street.RIVER, "Check-down should terminate on the river"
        assert Street.TURN in streets_seen, "Hand must actually reach and pass the turn"
        # CALL (limp) + three check-check pairs = 7 actions.
        assert len(state.betting_history) == 7
