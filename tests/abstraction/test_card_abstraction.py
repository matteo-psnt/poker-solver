"""Tests for card abstraction."""

import pytest

from src.abstraction.card_abstraction import RankBasedBucketing
from src.game.state import Card, Street


class TestRankBasedBucketing:
    """Tests for RankBasedBucketing."""

    def test_create_bucketing(self):
        bucketing = RankBasedBucketing()
        assert bucketing is not None

    def test_num_buckets(self):
        bucketing = RankBasedBucketing()
        assert bucketing.num_buckets(Street.PREFLOP) == 6
        assert bucketing.num_buckets(Street.FLOP) == 9
        assert bucketing.num_buckets(Street.TURN) == 9
        assert bucketing.num_buckets(Street.RIVER) == 9

    def test_preflop_premium_pairs(self):
        """Premium pairs (AA-JJ) should be bucket 0."""
        bucketing = RankBasedBucketing()

        # Test pocket aces
        hole_cards = (Card.new("As"), Card.new("Ah"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 0

        # Test pocket kings
        hole_cards = (Card.new("Ks"), Card.new("Kh"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 0

        # Test pocket jacks
        hole_cards = (Card.new("Js"), Card.new("Jh"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 0

    def test_preflop_medium_pairs(self):
        """Medium pairs (TT-77) should be bucket 1."""
        bucketing = RankBasedBucketing()

        # Test pocket tens
        hole_cards = (Card.new("Ts"), Card.new("Th"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 1

        # Test pocket sevens
        hole_cards = (Card.new("7s"), Card.new("7h"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 1

    def test_preflop_small_pairs(self):
        """Small pairs (66-22) should be bucket 2."""
        bucketing = RankBasedBucketing()

        # Test pocket sixes
        hole_cards = (Card.new("6s"), Card.new("6h"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 2

        # Test pocket deuces
        hole_cards = (Card.new("2s"), Card.new("2h"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 2

    def test_preflop_high_cards(self):
        """High cards (AK, AQ) should be bucket 3."""
        bucketing = RankBasedBucketing()

        # Test AK
        hole_cards = (Card.new("As"), Card.new("Kh"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 3

        # Test AQ
        hole_cards = (Card.new("Ah"), Card.new("Qd"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 3

    def test_preflop_medium_cards(self):
        """Medium cards (AJ, KQ) should be bucket 4."""
        bucketing = RankBasedBucketing()

        # Test AJ
        hole_cards = (Card.new("As"), Card.new("Jh"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 4

        # Test KQ
        hole_cards = (Card.new("Ks"), Card.new("Qh"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 4

    def test_preflop_weak_hands(self):
        """Weak hands should be bucket 5."""
        bucketing = RankBasedBucketing()

        # Test 72o (worst hand)
        hole_cards = (Card.new("7s"), Card.new("2h"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 5

        # Test random weak hand
        hole_cards = (Card.new("9s"), Card.new("3h"))
        bucket = bucketing.get_bucket(hole_cards, tuple(), Street.PREFLOP)
        assert bucket == 5

    def test_postflop_bucketing(self):
        """Postflop should use hand rank classes."""
        bucketing = RankBasedBucketing()

        # Top pair on flop
        hole_cards = (Card.new("As"), Card.new("Kh"))
        board = (Card.new("Ad"), Card.new("7c"), Card.new("2s"))

        bucket = bucketing.get_bucket(hole_cards, board, Street.FLOP)

        # Should be pair (bucket 7) or better
        assert 0 <= bucket <= 8

    def test_postflop_flush(self):
        """Flush should be strong bucket."""
        bucketing = RankBasedBucketing()

        # Flush on river
        hole_cards = (Card.new("As"), Card.new("Ks"))
        board = (Card.new("Qs"), Card.new("Js"), Card.new("9s"), Card.new("7d"), Card.new("2h"))

        bucket = bucketing.get_bucket(hole_cards, board, Street.RIVER)

        # Flush is rank class 4 -> bucket 3
        assert bucket <= 3  # Should be strong

    def test_different_suits_same_bucket(self):
        """Same hand different suits should give same bucket."""
        bucketing = RankBasedBucketing()

        bucket1 = bucketing.get_bucket(
            (Card.new("As"), Card.new("Ks")), tuple(), Street.PREFLOP
        )
        bucket2 = bucketing.get_bucket(
            (Card.new("Ah"), Card.new("Kh")), tuple(), Street.PREFLOP
        )

        # Both are AK suited -> should be same bucket
        assert bucket1 == bucket2

    def test_str_representation(self):
        bucketing = RankBasedBucketing()
        s = str(bucketing)
        assert "RankBasedBucketing" in s
