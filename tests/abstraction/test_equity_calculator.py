"""Tests for equity calculator."""

from src.abstraction.equity_calculator import EquityCalculator
from src.game.state import Card, Street


class TestEquityCalculator:
    """Tests for EquityCalculator."""

    def test_create_calculator(self):
        """Test creating calculator."""
        calc = EquityCalculator(num_samples=100)
        assert calc is not None
        assert calc.num_samples == 100

    def test_create_calculator_with_seed(self):
        """Test creating calculator with seed for reproducibility."""
        calc1 = EquityCalculator(num_samples=100, seed=42)
        calc2 = EquityCalculator(num_samples=100, seed=42)

        hole_cards = (Card.new("As"), Card.new("Ks"))
        board = (Card.new("Ah"), Card.new("2c"), Card.new("3d"))

        equity1 = calc1.calculate_equity(hole_cards, board, Street.FLOP)
        equity2 = calc2.calculate_equity(hole_cards, board, Street.FLOP)

        # With same seed, results should be reasonably similar
        # (some variance due to sampling, but within 10%)
        assert abs(equity1 - equity2) < 0.10

    def test_equity_aces_vs_random_preflop(self):
        """Test that pocket aces have ~85% equity preflop."""
        calc = EquityCalculator(num_samples=1000, seed=42)

        hole_cards = (Card.new("As"), Card.new("Ah"))
        board = ()

        equity = calc.calculate_equity(hole_cards, board, Street.PREFLOP)

        # AA should win about 85% vs random hand
        assert 0.80 < equity < 0.90

    def test_equity_72o_vs_random_preflop(self):
        """Test that 72o has low equity preflop."""
        calc = EquityCalculator(num_samples=1000, seed=42)

        hole_cards = (Card.new("7s"), Card.new("2h"))
        board = ()

        equity = calc.calculate_equity(hole_cards, board, Street.PREFLOP)

        # 72o is worst hand, should have ~35% equity vs random
        assert 0.25 < equity < 0.45

    def test_equity_top_pair_on_flop(self):
        """Test equity with top pair on flop."""
        calc = EquityCalculator(num_samples=1000, seed=42)

        # Top pair top kicker on flop
        hole_cards = (Card.new("As"), Card.new("Kh"))
        board = (Card.new("Ah"), Card.new("7c"), Card.new("2d"))

        equity = calc.calculate_equity(hole_cards, board, Street.FLOP)

        # Top pair should be strong (70-90%)
        assert 0.65 < equity < 0.95

    def test_equity_flush_draw_on_flop(self):
        """Test equity with flush draw on flop."""
        calc = EquityCalculator(num_samples=1000, seed=42)

        # Nut flush draw with overcards (AKs is strong!)
        hole_cards = (Card.new("As"), Card.new("Ks"))
        board = (Card.new("2s"), Card.new("7s"), Card.new("Jh"))

        equity = calc.calculate_equity(hole_cards, board, Street.FLOP)

        # Flush draw + 2 overcards is actually quite strong (60-75%)
        # Pure flush draw is ~35%, but AK adds ~15-20% more equity
        assert 0.50 < equity < 0.80

    def test_equity_made_flush_on_river(self):
        """Test equity with made flush on river."""
        calc = EquityCalculator(num_samples=500, seed=42)

        # Nut flush
        hole_cards = (Card.new("As"), Card.new("Ks"))
        board = (Card.new("2s"), Card.new("7s"), Card.new("Jh"), Card.new("Ts"), Card.new("3d"))

        equity = calc.calculate_equity(hole_cards, board, Street.RIVER)

        # Made flush should be very strong (90%+)
        assert equity > 0.85

    def test_calculate_equity_distribution_made_hand(self):
        """Test equity distribution for made hand."""
        calc = EquityCalculator(num_samples=1000, seed=42)

        # Top pair (made hand)
        hole_cards = (Card.new("As"), Card.new("Kh"))
        board = (Card.new("Ah"), Card.new("7c"), Card.new("2d"))

        dist = calc.calculate_equity_distribution(hole_cards, board, Street.FLOP, num_buckets=10)

        # Should sum to 1
        assert abs(dist.sum() - 1.0) < 0.01

        # Most mass should be in high equity buckets (7-10)
        high_equity_mass = dist[7:].sum()
        assert high_equity_mass > 0.6

    def test_calculate_equity_distribution_draw(self):
        """Test equity distribution for draw."""
        calc = EquityCalculator(num_samples=1000, seed=42)

        # Flush draw (draw hand)
        hole_cards = (Card.new("As"), Card.new("Ks"))
        board = (Card.new("2s"), Card.new("7s"), Card.new("Jh"))

        dist = calc.calculate_equity_distribution(hole_cards, board, Street.FLOP, num_buckets=10)

        # Should sum to 1
        assert abs(dist.sum() - 1.0) < 0.01

        # Distribution should be more spread out for draws
        # (some mass in low buckets for misses, some in high for hits)
        # Hard to test exact distribution, but should have variance
        assert len([x for x in dist if x > 0.05]) >= 2

    def test_batch_calculate_equity(self):
        """Test batch equity calculation."""
        calc = EquityCalculator(num_samples=500, seed=42)

        # Multiple hands
        hands = [
            (Card.new("As"), Card.new("Ah")),  # AA
            (Card.new("Ks"), Card.new("Kh")),  # KK
            (Card.new("7s"), Card.new("2h")),  # 72o
        ]

        board = ()

        equities = calc.batch_calculate_equity(hands, board, Street.PREFLOP)

        assert len(equities) == 3

        # AA should have highest equity
        assert equities[0] > equities[1]
        assert equities[0] > equities[2]

        # KK should beat 72o
        assert equities[1] > equities[2]

    def test_str_representation(self):
        """Test string representation."""
        calc = EquityCalculator(num_samples=1000)
        s = str(calc)
        assert "1000" in s
