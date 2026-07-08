"""Choosing which blueprint answers a hand.

NEAREST IN LOG SPACE. Depth error is proportional -- a 25 bb blueprint is as
wrong at 50 bb as a 50 bb one is at 100 -- so the midpoint between two rungs is
their geometric mean, not their average. Measured across the 25-100 bb gap the
crossover lands near sqrt(25 x 100) = 50, and the rule calls all four sampled
tables correctly.

The earlier rule snapped DOWN, and a live probe caught it answering an 80 bb
table with the 25 bb rung -- worth -140 mbb/hand.
"""

from __future__ import annotations

import pytest

from src.interfaces.chipzen.ladder import DepthLadder
from tests.test_helpers import build_trained_test_solver

BB = 100


@pytest.fixture(scope="module")
def ladder():
    """Rungs at 4, 10 and 40 bb -- widely spaced, so a wrong pick is obvious."""
    return DepthLadder(
        [build_trained_test_solver(iterations=2, starting_stack=d * BB) for d in (40, 4, 10)]
    )


class TestSelection:
    def test_the_rungs_are_ordered_however_they_arrive(self, ladder):
        assert [r.depth for r in ladder.rungs] == [4.0, 10.0, 40.0]

    def test_an_exact_depth_takes_its_own_rung(self, ladder):
        assert ladder.select(10.0).depth == 10.0

    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            (39.9, 40.0),  # all but on the deep rung
            (10.1, 10.0),  # all but on the middle one
            (4.5, 4.0),  # nearer 4 than 10 either way you measure
            (6.4, 10.0),  # sqrt(4 x 10) = 6.32, so just above it is the DEEP side
            (6.2, 4.0),  # and just below it is the shallow one
            (20.1, 40.0),  # sqrt(10 x 40) = 20, so the deep side
            (19.9, 10.0),  # and the shallow side
        ],
    )
    def test_the_nearest_rung_in_log_space_answers(self, ladder, table, expected):
        # The boundary is the GEOMETRIC mean of the neighbouring rungs. Snapping
        # down instead answered an 80 bb table with a 25 bb blueprint, measured
        # at -140 mbb/hand against the 100 bb one.
        assert ladder.select(table).depth == expected

    def test_a_table_in_a_wide_gap_is_not_stranded_on_the_far_rung(self, ladder):
        # The failure the live probe found: rungs at 10 and 40 with a table at
        # 35 must not be answered by 10 merely because 10 is below it.
        assert ladder.select(35.0).depth == 40.0

    def test_a_table_deeper_than_every_rung_takes_the_deepest(self, ladder):
        assert ladder.select(500.0).depth == 40.0

    def test_a_table_shallower_than_every_rung_takes_the_shallowest(self, ladder):
        # Answering is better than not: the alternative is a safe default, which
        # at 1 bb folds every hand.
        assert ladder.select(0.5).depth == 4.0

    def test_a_zero_or_negative_depth_does_not_blow_up(self, ladder):
        # `log(0)` is a crash, and a frame CAN report a zero effective stack.
        assert ladder.select(0.0).depth == 4.0
        assert ladder.select(-1.0).depth == 4.0

    def test_the_deepest_is_reachable_for_the_default_seat(self, ladder):
        assert ladder.deepest.depth == 40.0


class TestConstruction:
    def test_an_empty_ladder_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            DepthLadder([])

    def test_two_rungs_at_one_depth_are_refused(self):
        # Not fastidiousness: `select` returns the last match, so the other would
        # be loaded, held in memory for the life of the seat, and never played.
        pair = [build_trained_test_solver(iterations=2, starting_stack=8 * BB) for _ in range(2)]
        with pytest.raises(ValueError, match="share a depth"):
            DepthLadder(pair)

    def test_one_blueprint_is_a_legal_ladder(self):
        # The seat must behave identically to today when handed a single rung.
        solo = DepthLadder([build_trained_test_solver(iterations=2, starting_stack=8 * BB)])
        assert solo.select(100.0).depth == 8.0
        assert solo.select(1.0).depth == 8.0
