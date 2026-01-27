"""Conversions from chips into the units results are reported in.

Every published number (exploitability, head-to-head win rate) goes through here
so the scale factor and the two-seat averaging are defined exactly once.
"""

MBB_PER_BB = 1000.0


def chips_to_bb(chips: float, big_blind: int) -> float:
    """Convert a chip quantity to big blinds."""
    return chips / big_blind


def chips_to_mbb(chips: float, big_blind: int) -> float:
    """Convert a chip quantity to milli-big-blinds."""
    return chips / big_blind * MBB_PER_BB


def pair_mean_mbb(seat0_chips: float, seat1_chips: float, big_blind: int) -> float:
    """Mean mbb/hand over a seat-swapped pair of games sharing one deal.

    Both arguments are the SAME player's payoff, once from each seat, so the deal's
    card luck cancels in the average. The pair is two hands, not one: the halving
    belongs here rather than at call sites, where it is a factor-of-2 error waiting
    to happen.
    """
    return chips_to_mbb((seat0_chips + seat1_chips) / 2.0, big_blind)
