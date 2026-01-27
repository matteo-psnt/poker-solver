"""Tests for chip -> bb/mbb conversions."""

from src.shared.units import chips_to_bb, chips_to_mbb, pair_mean_mbb


def test_chips_to_bb():
    assert chips_to_bb(200.0, 2) == 100.0


def test_chips_to_mbb():
    assert chips_to_mbb(2.0, 2) == 1000.0
    assert chips_to_mbb(-1.0, 2) == -500.0
    assert chips_to_mbb(0.0, 2) == 0.0


def test_mbb_is_bb_scaled_by_1000():
    assert chips_to_mbb(37.0, 2) == chips_to_bb(37.0, 2) * 1000.0


def test_pair_mean_mbb_averages_the_two_seats():
    """The factor of 2 is a mean over seats, not a sum — the drift this guards."""
    assert pair_mean_mbb(10.0, 0.0, 2) == chips_to_mbb(5.0, 2)
    assert pair_mean_mbb(4.0, 4.0, 2) == chips_to_mbb(4.0, 2)


def test_pair_mean_mbb_cancels_a_zero_sum_pair():
    """Winning x from one seat and losing x from the other nets zero."""
    assert pair_mean_mbb(12.5, -12.5, 2) == 0.0


def test_pair_mean_mbb_is_seat_order_independent():
    assert pair_mean_mbb(9.0, -3.0, 2) == pair_mean_mbb(-3.0, 9.0, 2)
