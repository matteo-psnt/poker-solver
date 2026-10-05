"""Two blueprints ranked at one table depth, whatever depth each was cut for.

Both seats share one strategy stream per game, so a self-duel must be all-zero
pairs: that is the evidence a non-zero pair is the strategies and not the luck.
"""

from __future__ import annotations

import math

import pytest

from src.pipeline.evaluation.estimators.cross_depth import CrossDepthPlayer
from src.pipeline.evaluation.estimators.depth_duel import play_depth_duel
from tests.test_helpers import build_trained_test_solver

BB = 100
DEALS = 12


@pytest.fixture(scope="module")
def shallow():
    return build_trained_test_solver(iterations=6, starting_stack=4 * BB)


@pytest.fixture(scope="module")
def deep():
    return build_trained_test_solver(iterations=6, starting_stack=12 * BB)


def _duel(a, b, *, table_bb: int):
    return play_depth_duel(
        CrossDepthPlayer(a),
        CrossDepthPlayer(b),
        rules=a.rules,
        starting_stack=table_bb * BB,
        num_deals=DEALS,
        seed=3,
    )


def test_a_self_duel_is_all_zero_pairs(deep):
    result = _duel(deep, deep, table_bb=8)
    assert all(sample == 0.0 for sample in result.pair_samples_mbb)


def test_swapping_the_sides_flips_the_sign(shallow, deep):
    """The attribution test: a seat bug leaves the magnitude right and the sign wrong."""
    forward = _duel(shallow, deep, table_bb=8)
    backward = _duel(deep, shallow, table_bb=8)
    assert forward.a_mbb_per_hand == pytest.approx(-backward.a_mbb_per_hand)
    assert any(sample != 0.0 for sample in forward.pair_samples_mbb)


def test_a_table_neither_was_cut_for_plays_to_a_number(shallow, deep):
    result = _duel(shallow, deep, table_bb=7)
    assert result.num_hands == 2 * DEALS
    assert math.isfinite(result.a_mbb_per_hand)
