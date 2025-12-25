"""
Tests for betting sequence normalization.

Verifies that actions are normalized with the pot size at the time
they were made, not the current pot size.
"""

from src.game.actions import bet, call, check, raises
from src.game.state import GameState, Street


class TestBettingNormalization:
    """Test betting sequence normalization correctness."""

    def test_normalize_bet_with_correct_pot(self):
        """Bet should be normalized with pot at time of bet."""
        # Bet 50 into pot of 100
        state = GameState(
            street=Street.FLOP,
            pot=150,  # After bet: 100 + 50
            stacks=(850, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=50,
            last_aggressor=0,
            betting_history=(bet(50),),
            street_start_pot=100,  # Pot at start of betting round
        )

        normalized = state._normalize_betting_sequence()
        # Bet 50 into pot of 100 (before bet) = 0.50
        assert normalized == "b0.50", f"Expected 'b0.50', got '{normalized}'"

    def test_normalize_bet_raise_with_correct_pots(self):
        """Bet and raise should each use their respective pot sizes."""
        # Initial pot: 100
        # After bet 50: pot = 150
        # After raise 75: pot = 275 (raise adds to_call + raise_amount = 50 + 75 = 125)
        state = GameState(
            street=Street.FLOP,
            pot=275,  # Correct pot: 100 + 50 + 125 = 275
            stacks=(850, 775),  # Player 0: 900 - 50 = 850, Player 1: 900 - 125 = 775
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=75,
            last_aggressor=1,
            betting_history=(bet(50), raises(75)),
            street_start_pot=100,  # Pot at start of betting round
        )

        normalized = state._normalize_betting_sequence()
        # Bet 50 into 100 = 0.50
        # Raise 75 into 150 = 0.50
        assert normalized == "b0.50-r0.50", f"Expected 'b0.50-r0.50', got '{normalized}'"

    def test_normalize_different_amounts_same_fractions(self):
        """Different absolute amounts but same pot fractions should normalize identically."""
        # Scenario 1: Bet 50 into 100, raise 75 into 150 → pot 275
        state1 = GameState(
            street=Street.FLOP,
            pot=275,  # 100 + 50 + 125 = 275
            stacks=(850, 775),  # Player 0: 900 - 50, Player 1: 900 - 125
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=75,
            last_aggressor=1,
            betting_history=(bet(50), raises(75)),
            street_start_pot=100,
        )

        # Scenario 2: Bet 100 into 200, raise 150 into 300 → pot 550
        state2 = GameState(
            street=Street.FLOP,
            pot=550,  # 200 + 100 + 250 = 550
            stacks=(700, 550),  # Player 0: 800 - 100, Player 1: 800 - 250
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=150,
            last_aggressor=1,
            betting_history=(bet(100), raises(150)),
            street_start_pot=200,
        )

        norm1 = state1._normalize_betting_sequence()
        norm2 = state2._normalize_betting_sequence()

        assert norm1 == norm2 == "b0.50-r0.50", (
            f"Different amounts with same fractions should normalize identically: "
            f"state1={norm1}, state2={norm2}"
        )

    def test_normalize_mixed_actions(self):
        """Test normalization with mix of action types."""
        # Check, bet, call sequence
        # Initial pot: 100
        # After check: 100
        # After bet 50: 150
        # After call 50: 200
        state = GameState(
            street=Street.FLOP,
            pot=200,  # 100 + 50 + 50
            stacks=(900, 850),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            betting_history=(check(), bet(50), call()),
            street_start_pot=100,
        )

        normalized = state._normalize_betting_sequence()
        # Check = x, bet 50 into 100 = b0.50, call = c
        assert normalized == "x-b0.50-c", f"Expected 'x-b0.50-c', got '{normalized}'"

    def test_normalize_empty_history(self):
        """Empty betting history should return empty string."""
        state = GameState(
            street=Street.FLOP,
            pot=100,
            stacks=(900, 900),
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            betting_history=(),
            street_start_pot=100,
        )

        normalized = state._normalize_betting_sequence()
        assert normalized == "", f"Expected empty string, got '{normalized}'"

    def test_normalize_complex_sequence(self):
        """Test complex betting sequence with multiple actions."""
        # Initial pot: 100
        # bet 30 into 100 → pot = 130
        # raise 60 into 130 → pot = 220 (adds to_call 30 + raise 60 = 90)
        # call 60 → pot = 280
        state = GameState(
            street=Street.FLOP,
            pot=280,  # 100 + 30 + 90 + 60 = 280
            stacks=(810, 810),  # Player 0: 900 - 30 - 60, Player 1: 900 - 90
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=1,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            betting_history=(bet(30), raises(60), call()),
            street_start_pot=100,
        )

        normalized = state._normalize_betting_sequence()
        # bet 30 into 100 = 0.30
        # raise 60 into 130 = 0.46 (rounded)
        assert "b0.30" in normalized, f"Expected 'b0.30' in sequence, got '{normalized}'"
        assert "r0.46" in normalized, f"Expected 'r0.46' in sequence, got '{normalized}'"
        assert normalized.endswith("-c"), f"Expected sequence to end with '-c', got '{normalized}'"

    def test_pot_evolution_with_multiple_bets(self):
        """Verify pot evolution is tracked correctly through multiple bets."""
        # Initial: 100
        # After bet 25: 125
        # After raise 50: 200 (adds to_call 25 + raise 50 = 75)
        # After raise 75: 325 (adds to_call 50 + raise 75 = 125)
        state = GameState(
            street=Street.FLOP,
            pot=325,  # 100 + 25 + 75 + 125 = 325
            stacks=(750, 825),  # P0: 900 - 25 - 125, P1: 900 - 75
            board=("Ah", "Kh", "Qh"),  # type: ignore
            hole_cards=(("2c", "3c"), ("4d", "5d")),  # type: ignore
            button_position=0,
            current_player=0,
            is_terminal=False,
            to_call=75,
            last_aggressor=1,
            betting_history=(bet(25), raises(50), raises(75)),
            street_start_pot=100,
        )

        normalized = state._normalize_betting_sequence()

        # bet 25 into 100 = 0.25
        # raise 50 into 125 = 0.40
        # raise 75 into 200 = 0.375 ≈ 0.38 (rounded)
        assert normalized.startswith("b0.25"), f"First action should be 'b0.25', got '{normalized}'"
        assert "r0.40" in normalized, f"Should contain 'r0.40', got '{normalized}'"
        assert "r0.38" in normalized or "r0.37" in normalized, (
            f"Should contain 'r0.38' or 'r0.37' (rounding), got '{normalized}'"
        )
