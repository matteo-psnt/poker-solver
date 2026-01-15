"""Tests for the exact range-vs-range equity engine."""

import itertools

import numpy as np
import pytest

from src.core.game.evaluator import get_evaluator
from src.core.game.state import Card
from src.pipeline.abstraction.utils.equity import RangeEquityEngine

RIVER_BOARD = (Card.new("2s"), Card.new("7s"), Card.new("Jh"), Card.new("Ts"), Card.new("3d"))
TURN_BOARD = (Card.new("2s"), Card.new("7s"), Card.new("Jh"), Card.new("Ts"))
FLOP_BOARD = (Card.new("Ts"), Card.new("9s"), Card.new("8c"))


def brute_force_river_equity(hole: tuple[Card, Card], board: tuple[Card, ...]) -> float:
    """Reference implementation: enumerate every opponent combo pairwise."""
    evaluator = get_evaluator()
    used = set(board) | set(hole)
    deck = [c for c in Card.get_full_deck() if c not in used]

    wins = ties = total = 0
    for opponent in itertools.combinations(deck, 2):
        result = evaluator.compare_hands(hole, opponent, board)
        total += 1
        if result < 0:
            wins += 1
        elif result == 0:
            ties += 1
    return (wins + 0.5 * ties) / total


class TestRiverExact:
    """River equities are exact (single runout, full opponent enumeration)."""

    @pytest.fixture(scope="class")
    def river_table(self):
        return RangeEquityEngine().board_equities(RIVER_BOARD)

    def test_combo_count(self, river_table):
        # 47 unseen cards -> C(47, 2) combos
        assert len(river_table) == 1081

    def test_matches_brute_force(self, river_table):
        hands = [
            (Card.new("As"), Card.new("Ks")),  # nut flush
            (Card.new("Jd"), Card.new("Tc")),  # two pair
            (Card.new("5h"), Card.new("4h")),  # air
        ]
        for hole in hands:
            assert river_table.equity(hole) == pytest.approx(
                brute_force_river_equity(hole, RIVER_BOARD), abs=1e-12
            )

    def test_mean_equity_is_exactly_half(self, river_table):
        # Every combo faces the same number of opponents, and every ordered
        # matchup contributes symmetrically, so the mean is exactly 0.5.
        assert river_table.equities.mean() == pytest.approx(0.5, abs=1e-12)

    def test_nut_hand_has_equity_one(self, river_table):
        assert river_table.equity((Card.new("As"), Card.new("Ks"))) == 1.0

    def test_invalid_combo_raises(self, river_table):
        with pytest.raises(KeyError):
            river_table.equity((Card.new("2s"), Card.new("Ah")))  # 2s is on the board


class TestTurnExact:
    """Turn equities enumerate all rivers exactly."""

    @pytest.fixture(scope="class")
    def turn_table(self):
        return RangeEquityEngine().board_equities(TURN_BOARD)

    def test_combo_count(self, turn_table):
        assert len(turn_table) == 1128  # C(48, 2)

    def test_mean_equity_is_exactly_half(self, turn_table):
        assert turn_table.equities.mean() == pytest.approx(0.5, abs=1e-12)

    def test_nut_flush_is_strong(self, turn_table):
        equity = turn_table.equity((Card.new("As"), Card.new("Ks")))
        assert equity > 0.9

    def test_no_nan_equities(self, turn_table):
        assert not np.isnan(turn_table.equities).any()


class TestFlop:
    """Flop equities: exact enumeration or deterministic runout sampling."""

    def test_sampled_runouts_deterministic_per_board(self):
        table1 = RangeEquityEngine(max_runouts=100, seed=7).board_equities(FLOP_BOARD)
        table2 = RangeEquityEngine(max_runouts=100, seed=7).board_equities(FLOP_BOARD)
        assert np.array_equal(table1.equities, table2.equities)

    def test_different_seed_changes_sample(self):
        table1 = RangeEquityEngine(max_runouts=100, seed=7).board_equities(FLOP_BOARD)
        table2 = RangeEquityEngine(max_runouts=100, seed=8).board_equities(FLOP_BOARD)
        assert not np.array_equal(table1.equities, table2.equities)

    def test_flush_draw_with_overcards_equity(self):
        # AKs with nut flush draw + overcards vs a random hand.
        engine = RangeEquityEngine(max_runouts=200)
        equity = engine.hand_equity(
            (Card.new("As"), Card.new("Ks")), (Card.new("2s"), Card.new("7s"), Card.new("Jh"))
        )
        assert 0.6 < equity < 0.85

    @pytest.mark.slow
    @pytest.mark.timeout(30)
    def test_exact_flop_mean_half_and_sampling_agreement(self):
        exact = RangeEquityEngine().board_equities(FLOP_BOARD)
        assert len(exact) == 1176  # C(49, 2)
        assert exact.equities.mean() == pytest.approx(0.5, abs=1e-12)

        sampled = RangeEquityEngine(max_runouts=200).board_equities(FLOP_BOARD)
        errors = np.abs(sampled.equities - exact.equities)
        # Each sampled runout is an exact river pass, so 200 runouts already
        # pin equities down tightly.
        assert np.max(errors) < 0.06
        assert np.mean(errors) < 0.02


class TestHistograms:
    """Equity-realization histograms for potential-aware bucketing."""

    @pytest.fixture(scope="class")
    def flop_table(self):
        return RangeEquityEngine(max_runouts=150).board_equities(FLOP_BOARD, histogram_bins=10)

    def test_rows_sum_to_one(self, flop_table):
        assert flop_table.histograms is not None
        row_sums = flop_table.histograms.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)

    def test_draw_has_spread_distribution(self, flop_table):
        # A bare flush draw either hits (very high river equity) or misses
        # (low), so its realization distribution must span multiple bins.
        histogram = flop_table.histogram((Card.new("As"), Card.new("2s")))
        assert (histogram > 0.05).sum() >= 2

    def test_histogram_without_bins_raises(self):
        table = RangeEquityEngine().board_equities(RIVER_BOARD)
        with pytest.raises(ValueError):
            table.histogram((Card.new("As"), Card.new("Ks")))


class TestPreflop:
    """Preflop is supported only via runout sampling."""

    def test_requires_max_runouts(self):
        with pytest.raises(ValueError):
            RangeEquityEngine().board_equities(())

    @pytest.mark.timeout(15)
    def test_pocket_aces_equity(self):
        engine = RangeEquityEngine(max_runouts=200)
        equity = engine.hand_equity((Card.new("As"), Card.new("Ah")), ())
        assert 0.8 < equity < 0.9


class TestEngineApi:
    def test_invalid_board_size_raises(self):
        with pytest.raises(ValueError):
            RangeEquityEngine().board_equities((Card.new("As"), Card.new("Ks")))

    def test_hand_equity_uses_cached_table(self):
        engine = RangeEquityEngine()
        first = engine.hand_equity((Card.new("As"), Card.new("Ks")), RIVER_BOARD)
        second = engine.hand_equity((Card.new("Jd"), Card.new("Tc")), RIVER_BOARD)
        assert len(engine._table_cache) == 1
        assert first == 1.0
        assert 0.0 <= second <= 1.0
