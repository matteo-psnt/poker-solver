"""Tests for preflop hand mapping."""

from src.abstraction.preflop_hands import PreflopHandMapper
from src.game.state import Card


class TestPreflopHandMapper:
    """Tests for PreflopHandMapper."""

    def test_create_mapper(self):
        """Test creating mapper."""
        mapper = PreflopHandMapper()
        assert mapper is not None

    def test_get_hand_string_pairs(self):
        """Test mapping pairs."""
        mapper = PreflopHandMapper()

        # Aces
        assert mapper.get_hand_string((Card.new("As"), Card.new("Ah"))) == "AA"
        assert mapper.get_hand_string((Card.new("Ad"), Card.new("Ac"))) == "AA"

        # Kings
        assert mapper.get_hand_string((Card.new("Ks"), Card.new("Kh"))) == "KK"

        # Deuces
        assert mapper.get_hand_string((Card.new("2s"), Card.new("2h"))) == "22"

    def test_get_hand_string_suited(self):
        """Test mapping suited hands."""
        mapper = PreflopHandMapper()

        # High cards
        assert mapper.get_hand_string((Card.new("As"), Card.new("Ks"))) == "AKs"
        assert mapper.get_hand_string((Card.new("Kd"), Card.new("Ad"))) == "AKs"  # Order matters

        # Medium cards
        assert mapper.get_hand_string((Card.new("Th"), Card.new("9h"))) == "T9s"
        assert mapper.get_hand_string((Card.new("9h"), Card.new("Th"))) == "T9s"  # Order matters

        # Low cards
        assert mapper.get_hand_string((Card.new("3c"), Card.new("2c"))) == "32s"

    def test_get_hand_string_offsuit(self):
        """Test mapping offsuit hands."""
        mapper = PreflopHandMapper()

        # High cards
        assert mapper.get_hand_string((Card.new("As"), Card.new("Kh"))) == "AKo"
        assert mapper.get_hand_string((Card.new("Kh"), Card.new("Ad"))) == "AKo"

        # Medium cards
        assert mapper.get_hand_string((Card.new("Ts"), Card.new("9h"))) == "T9o"

        # Low cards
        assert mapper.get_hand_string((Card.new("7s"), Card.new("2h"))) == "72o"

    def test_get_hand_index_pairs(self):
        """Test getting index for pairs."""
        mapper = PreflopHandMapper()

        # Aces should be 0
        assert mapper.get_hand_index((Card.new("As"), Card.new("Ah"))) == 0

        # Kings should be 1
        assert mapper.get_hand_index((Card.new("Ks"), Card.new("Kh"))) == 1

        # Deuces should be 12
        assert mapper.get_hand_index((Card.new("2s"), Card.new("2h"))) == 12

    def test_get_hand_index_suited(self):
        """Test getting index for suited hands."""
        mapper = PreflopHandMapper()

        # AKs should be first suited (index 13)
        assert mapper.get_hand_index((Card.new("As"), Card.new("Ks"))) == 13

        # AQs should be second suited (index 14)
        assert mapper.get_hand_index((Card.new("As"), Card.new("Qs"))) == 14

    def test_get_hand_index_offsuit(self):
        """Test getting index for offsuit hands."""
        mapper = PreflopHandMapper()

        # AKo should be first offsuit (index 91 = 13 pairs + 78 suited)
        assert mapper.get_hand_index((Card.new("As"), Card.new("Kh"))) == 91

        # AQo should be second offsuit (index 92)
        assert mapper.get_hand_index((Card.new("As"), Card.new("Qh"))) == 92

    def test_get_all_hands(self):
        """Test getting all 169 hands."""
        hands = PreflopHandMapper.get_all_hands()

        # Should have exactly 169 hands
        assert len(hands) == 169

        # Should have unique hands
        assert len(set(hands)) == 169

        # First should be AA
        assert hands[0] == "AA"

        # 13th should be first suited (AKs)
        assert hands[13] == "AKs"

        # 91st should be first offsuit (AKo)
        assert hands[91] == "AKo"

        # Last should be 32o
        assert hands[168] == "32o"

    def test_all_hands_can_be_indexed(self):
        """Test that all hands map to unique indices."""
        hands = PreflopHandMapper.get_all_hands()
        mapper = PreflopHandMapper()

        # Create sample cards for each hand
        indices = set()
        for hand_str in hands:
            # Convert hand string to cards (just need example)
            if len(hand_str) == 2:
                # Pair
                rank = hand_str[0]
                c1 = Card.new(f"{rank}s")
                c2 = Card.new(f"{rank}h")
            elif hand_str[2] == "s":
                # Suited
                high, low = hand_str[0], hand_str[1]
                c1 = Card.new(f"{high}s")
                c2 = Card.new(f"{low}s")
            else:
                # Offsuit
                high, low = hand_str[0], hand_str[1]
                c1 = Card.new(f"{high}s")
                c2 = Card.new(f"{low}h")

            idx = mapper.get_hand_index((c1, c2))
            indices.add(idx)

        # Should have 169 unique indices
        assert len(indices) == 169

        # Indices should be 0-168
        assert min(indices) == 0
        assert max(indices) == 168

    def test_str_representation(self):
        """Test string representation."""
        mapper = PreflopHandMapper()
        s = str(mapper)
        assert "169" in s
