"""Choosing which blueprint answers a hand.

Snap-down is the rule, and the tests state it as behaviour rather than as
arithmetic: what matters is that a rung is never asked about a table SHALLOWER
than the one it was cut for, because that is the direction the depth measurement
showed hurts.
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

    @pytest.mark.parametrize(("table", "expected"), [(39.9, 10.0), (10.1, 10.0), (4.5, 4.0)])
    def test_a_depth_between_rungs_snaps_downward(self, ladder, table, expected):
        # Never up: a blueprint that thinks it holds more than the table does
        # plans a bet it cannot complete and strands itself halfway through.
        assert ladder.select(table).depth == expected

    def test_a_table_deeper_than_every_rung_takes_the_deepest(self, ladder):
        assert ladder.select(500.0).depth == 40.0

    def test_a_table_shallower_than_every_rung_takes_the_shallowest(self, ladder):
        # There is nothing below to snap to, and answering is better than not:
        # the alternative is a safe default, which at 1 bb folds every hand.
        assert ladder.select(0.5).depth == 4.0

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
