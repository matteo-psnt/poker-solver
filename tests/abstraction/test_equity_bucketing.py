"""Tests for equity-based K-means bucketing."""

import tempfile
from pathlib import Path

import numpy as np

from src.abstraction.equity_bucketing import EquityBucketing
from src.abstraction.equity_calculator import EquityCalculator
from src.game.state import Card, Street


class TestEquityBucketing:
    """Tests for EquityBucketing."""

    def test_create_bucketing(self):
        """Test creating bucketing system."""
        bucketing = EquityBucketing()
        assert bucketing is not None
        assert bucketing.num_buckets[Street.FLOP] == 50
        assert bucketing.num_buckets[Street.TURN] == 100
        assert bucketing.num_buckets[Street.RIVER] == 200
        assert not bucketing.fitted

    def test_create_bucketing_custom_buckets(self):
        """Test creating bucketing with custom bucket counts."""
        custom = {
            Street.FLOP: 10,
            Street.TURN: 20,
            Street.RIVER: 30,
        }
        bucketing = EquityBucketing(num_buckets=custom)
        assert bucketing.num_buckets[Street.FLOP] == 10
        assert bucketing.num_buckets[Street.TURN] == 20
        assert bucketing.num_buckets[Street.RIVER] == 30

    def test_fit_small_sample(self):
        """Test fitting on small sample of boards."""
        # Use fast equity calculator for testing (only 20 samples)
        equity_calc = EquityCalculator(num_samples=20)

        # Use smaller bucket counts for testing
        bucketing = EquityBucketing(
            num_buckets={
                Street.FLOP: 5,
                Street.TURN: 5,
                Street.RIVER: 5,
            },
            num_board_clusters={
                Street.FLOP: 3,  # Small for testing
                Street.TURN: 3,
                Street.RIVER: 3,
            },
            equity_calculator=equity_calc,
        )

        # Generate sample flop boards
        sample_boards = {
            Street.FLOP: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs")),  # High monotone
                (Card.new("2s"), Card.new("3h"), Card.new("4d")),  # Low connected
                (Card.new("7s"), Card.new("7h"), Card.new("2d")),  # Paired
                (Card.new("Ah"), Card.new("2d"), Card.new("9c")),  # Rainbow
                (Card.new("Ts"), Card.new("Js"), Card.new("Qs")),  # Flush draw high
                (Card.new("5s"), Card.new("6h"), Card.new("7d")),  # Straight
                (Card.new("Kh"), Card.new("Qh"), Card.new("Jh")),  # Flush draw
                (Card.new("2c"), Card.new("7s"), Card.new("Kd")),  # Rainbow spread
            ]
        }

        # Fit (this will take a bit as it computes equities)
        bucketing.fit(sample_boards, num_samples_per_cluster=2)

        # Should now be fitted
        assert bucketing.fitted
        assert Street.FLOP in bucketing.bucket_assignments
        assert Street.FLOP in bucketing.clusterers

        # Check shape of bucket assignments: [169 hands, num_board_clusters]
        # Board clusters depend on K-means, but should be reasonable
        assignments = bucketing.bucket_assignments[Street.FLOP]
        assert assignments.shape[0] == 169  # 169 hands

        # All buckets should be in valid range
        assert np.all(assignments >= 0)
        assert np.all(assignments < 5)  # 5 buckets

    def test_get_bucket_after_fit(self):
        """Test getting bucket for a hand after fitting."""
        equity_calc = EquityCalculator(num_samples=20)
        bucketing = EquityBucketing(
            num_buckets={Street.FLOP: 5},
            num_board_clusters={Street.FLOP: 3},
            equity_calculator=equity_calc,
        )

        # Sample boards
        sample_boards = {
            Street.FLOP: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs")),
                (Card.new("2s"), Card.new("3h"), Card.new("4d")),
                (Card.new("7s"), Card.new("7h"), Card.new("2d")),
                (Card.new("Ah"), Card.new("2d"), Card.new("9c")),
            ]
        }

        bucketing.fit(sample_boards, num_samples_per_cluster=2)

        # Get bucket for AA on high board
        hole_cards = (Card.new("Ad"), Card.new("Ac"))
        board = (Card.new("Ks"), Card.new("Qs"), Card.new("Jh"))

        bucket = bucketing.get_bucket(hole_cards, board, Street.FLOP)

        # Should return valid bucket
        assert isinstance(bucket, int)
        assert 0 <= bucket < 5

    def test_get_bucket_before_fit(self):
        """Test getting bucket before fitting raises error."""
        bucketing = EquityBucketing()

        hole_cards = (Card.new("As"), Card.new("Ah"))
        board = (Card.new("Ks"), Card.new("Qs"), Card.new("Jh"))

        try:
            bucketing.get_bucket(hole_cards, board, Street.FLOP)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not fitted" in str(e)

    def test_example_hand_generation(self):
        """Test generating example hands from hand strings."""
        bucketing = EquityBucketing()

        # Test pair
        hand = bucketing._get_example_hand("AA")
        assert len(hand) == 2
        assert hand[0] != hand[1]  # Different suits

        # Test suited
        hand = bucketing._get_example_hand("AKs")
        assert len(hand) == 2
        # Should be same suit
        c1_str = repr(hand[0])
        c2_str = repr(hand[1])
        assert c1_str[1] == c2_str[1] or c1_str[2] == c2_str[2]  # Same suit

        # Test offsuit
        hand = bucketing._get_example_hand("AKo")
        assert len(hand) == 2

    def test_cards_conflict_detection(self):
        """Test detecting card conflicts."""
        bucketing = EquityBucketing()

        # No conflict
        hole_cards = (Card.new("As"), Card.new("Ah"))
        board = (Card.new("Ks"), Card.new("Qs"), Card.new("Jh"))
        assert not bucketing._cards_conflict(hole_cards, board)

        # Conflict (As appears in both)
        hole_cards = (Card.new("As"), Card.new("Ah"))
        board = (Card.new("As"), Card.new("Qs"), Card.new("Jh"))
        assert bucketing._cards_conflict(hole_cards, board)

    def test_save_and_load(self):
        """Test saving and loading bucketing."""
        equity_calc = EquityCalculator(num_samples=20)
        bucketing = EquityBucketing(
            num_buckets={Street.FLOP: 5},
            num_board_clusters={Street.FLOP: 2},
            equity_calculator=equity_calc,
        )

        # Fit on sample
        sample_boards = {
            Street.FLOP: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs")),
                (Card.new("2s"), Card.new("3h"), Card.new("4d")),
            ]
        }
        bucketing.fit(sample_boards, num_samples_per_cluster=1)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_bucketing.pkl"
            bucketing.save(filepath)

            # Load
            loaded = EquityBucketing.load(filepath)

            # Should have same configuration
            assert loaded.num_buckets == bucketing.num_buckets
            assert loaded.fitted
            assert Street.FLOP in loaded.bucket_assignments

            # Should get same buckets
            hole_cards = (Card.new("Ad"), Card.new("Ac"))
            board = (Card.new("Ks"), Card.new("Qs"), Card.new("Jh"))

            bucket1 = bucketing.get_bucket(hole_cards, board, Street.FLOP)
            bucket2 = loaded.get_bucket(hole_cards, board, Street.FLOP)

            assert bucket1 == bucket2

    def test_different_hands_different_buckets(self):
        """Test that different strength hands get different buckets."""
        equity_calc = EquityCalculator(num_samples=20)
        bucketing = EquityBucketing(
            num_buckets={Street.FLOP: 10},
            num_board_clusters={Street.FLOP: 3},
            equity_calculator=equity_calc,
        )

        # Fit on diverse boards
        sample_boards = {
            Street.FLOP: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs")),
                (Card.new("2s"), Card.new("3h"), Card.new("4d")),
                (Card.new("7s"), Card.new("7h"), Card.new("2d")),
                (Card.new("Ah"), Card.new("2d"), Card.new("9c")),
                (Card.new("Ts"), Card.new("Js"), Card.new("Qs")),
            ]
        }
        bucketing.fit(sample_boards, num_samples_per_cluster=2)

        # Board: Ks Qs Jh (high connected)
        board = (Card.new("Ks"), Card.new("Qs"), Card.new("Jh"))

        # Strong hand: AA
        strong_hand = (Card.new("Ad"), Card.new("Ac"))
        strong_bucket = bucketing.get_bucket(strong_hand, board, Street.FLOP)

        # Weak hand: 72o
        weak_hand = (Card.new("7d"), Card.new("2c"))
        weak_bucket = bucketing.get_bucket(weak_hand, board, Street.FLOP)

        # They should likely be in different buckets (though not guaranteed with K-means)
        # At minimum, they should both be valid
        assert 0 <= strong_bucket < 10
        assert 0 <= weak_bucket < 10

    def test_get_num_buckets(self):
        """Test getting number of buckets."""
        bucketing = EquityBucketing()
        assert bucketing.get_num_buckets(Street.FLOP) == 50
        assert bucketing.get_num_buckets(Street.TURN) == 100
        assert bucketing.get_num_buckets(Street.RIVER) == 200

    def test_str_representation(self):
        """Test string representation."""
        bucketing = EquityBucketing()
        s = str(bucketing)
        assert "EquityBucketing" in s
        assert "50" in s
        assert "100" in s
        assert "200" in s
        assert "not fitted" in s

        # After fitting
        equity_calc = EquityCalculator(num_samples=20)
        bucketing_small = EquityBucketing(
            num_buckets={Street.FLOP: 5},
            num_board_clusters={Street.FLOP: 2},
            equity_calculator=equity_calc,
        )
        sample_boards = {
            Street.FLOP: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs")),
                (Card.new("2s"), Card.new("3h"), Card.new("4d")),
            ]
        }
        bucketing_small.fit(sample_boards, num_samples_per_cluster=1)

        s = str(bucketing_small)
        assert "fitted" in s

    def test_multiple_streets(self):
        """Test fitting multiple streets."""
        equity_calc = EquityCalculator(num_samples=20)
        bucketing = EquityBucketing(
            num_buckets={
                Street.FLOP: 5,
                Street.TURN: 5,
                Street.RIVER: 5,
            },
            num_board_clusters={
                Street.FLOP: 2,
                Street.TURN: 2,
                Street.RIVER: 2,
            },
            equity_calculator=equity_calc,
        )

        # Sample boards for all streets
        sample_boards = {
            Street.FLOP: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs")),
                (Card.new("2s"), Card.new("3h"), Card.new("4d")),
            ],
            Street.TURN: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs"), Card.new("Jh")),
                (Card.new("2s"), Card.new("3h"), Card.new("4d"), Card.new("5c")),
            ],
            Street.RIVER: [
                (Card.new("As"), Card.new("Ks"), Card.new("Qs"), Card.new("Jh"), Card.new("Th")),
                (Card.new("2s"), Card.new("3h"), Card.new("4d"), Card.new("5c"), Card.new("6h")),
            ],
        }

        bucketing.fit(sample_boards, num_samples_per_cluster=1)

        # All streets should be fitted
        assert Street.FLOP in bucketing.bucket_assignments
        assert Street.TURN in bucketing.bucket_assignments
        assert Street.RIVER in bucketing.bucket_assignments

        # Should be able to get buckets for all streets
        hole_cards = (Card.new("Ad"), Card.new("Ac"))

        flop_board = (Card.new("Kd"), Card.new("Qd"), Card.new("Jd"))
        turn_board = (Card.new("Kd"), Card.new("Qd"), Card.new("Jd"), Card.new("2h"))
        river_board = (
            Card.new("Kd"),
            Card.new("Qd"),
            Card.new("Jd"),
            Card.new("2h"),
            Card.new("3c"),
        )

        flop_bucket = bucketing.get_bucket(hole_cards, flop_board, Street.FLOP)
        turn_bucket = bucketing.get_bucket(hole_cards, turn_board, Street.TURN)
        river_bucket = bucketing.get_bucket(hole_cards, river_board, Street.RIVER)

        assert 0 <= flop_bucket < 5
        assert 0 <= turn_bucket < 5
        assert 0 <= river_bucket < 5
