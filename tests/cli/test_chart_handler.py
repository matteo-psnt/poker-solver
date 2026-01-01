"""Tests for chart handler."""

from src.chart.data import _ranks_to_hand_string


def test_ranks_to_hand_string_pair():
    """Test _ranks_to_hand_string for pairs."""
    result = _ranks_to_hand_string("A", "A", False, True)
    assert result == "AA"

    result = _ranks_to_hand_string("K", "K", False, True)
    assert result == "KK"


def test_ranks_to_hand_string_suited():
    """Test _ranks_to_hand_string for suited hands."""
    result = _ranks_to_hand_string("A", "K", True, False)
    assert result == "AKs"


def test_ranks_to_hand_string_offsuit():
    """Test _ranks_to_hand_string for offsuit hands."""
    result = _ranks_to_hand_string("A", "K", False, False)
    assert result == "AKo"
