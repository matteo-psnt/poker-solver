"""Whole elimination matches -- the only format Chipzen deals.

A fixed-depth duel measures a game the platform never plays. These pin the parts
of the real shape that are easy to get subtly wrong: that blinds climb, that the
ladder tracks the depth down as they do, and that a pair cancels the seats.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.rules import GameRules
from src.pipeline.evaluation.estimators.cross_depth import CrossDepthPlayer
from src.pipeline.evaluation.estimators.elimination import (
    LadderSeat,
    play_elimination,
    play_match,
)
from tests.test_helpers import build_trained_test_solver

BB = 100
RULES = GameRules(BB // 2, BB)  # the blueprints' own blinds, always


def _player(depth: int) -> CrossDepthPlayer:
    return CrossDepthPlayer(
        build_trained_test_solver(iterations=4, starting_stack=depth * BB), seed=depth
    )


@pytest.fixture(scope="module")
def rungs():
    return [_player(d) for d in (2, 8, 32)]


def _match(seats, **kwargs):
    return play_match(
        seats, rules=RULES, starting_stack=20 * BB, cards=np.random.default_rng(3), **kwargs
    )


class TestLadderSelection:
    def test_it_takes_the_nearest_rung_in_log_space(self, rungs):
        seat = LadderSeat(rungs)
        # sqrt(8 x 32) = 16, so 17 bb is on the deep side and 15 bb the shallow.
        assert seat.for_depth(17.0).depth_in_blinds == 32.0
        assert seat.for_depth(15.0).depth_in_blinds == 8.0

    def test_snap_up_takes_the_next_rung_up_within_twice_the_table(self):
        seat = LadderSeat([_player(d) for d in (8, 12, 32)], "up")
        assert seat.for_depth(9.0).depth_in_blinds == 12.0  # log-nearest would say 8
        assert seat.for_depth(8.0).depth_in_blinds == 8.0
        assert seat.for_depth(13.0).depth_in_blinds == 12.0  # 32 is beyond 2x: nearest
        assert seat.for_depth(40.0).depth_in_blinds == 32.0  # deeper than every rung

    def test_an_unknown_rule_is_refused(self, rungs):
        with pytest.raises(ValueError, match="selection rule"):
            LadderSeat(rungs, "down")

    def test_it_opens_on_the_deepest(self, rungs):
        assert LadderSeat(rungs).chosen.depth_in_blinds == 32.0

    def test_a_zero_depth_does_not_blow_up(self, rungs):
        assert LadderSeat(rungs).for_depth(0.0).depth_in_blinds == 2.0

    def test_switches_are_counted_only_on_a_change(self, rungs):
        seat = LadderSeat(rungs)
        seat.select(32.0)
        seat.select(30.0)
        assert seat.switches == 0
        seat.select(2.0)
        assert seat.switches == 1


class TestPlayMatch:
    def test_a_match_ends_with_one_winner(self, rungs):
        out = _match((LadderSeat([rungs[2]]), LadderSeat([rungs[1]])))
        assert out.winner in (0, 1)
        assert out.hands > 0

    def test_the_depth_actually_falls(self, rungs):
        """Escalation is shrinking stacks, so DEPTH is what moves -- and the table
        stays on the blinds the blueprints were cut for."""
        out = _match(
            (LadderSeat([rungs[2]]), LadderSeat([rungs[1]])), escalate_every=2, escalate_by=2.0
        )
        assert out.final_depth < 20.0, "a long match must have come off its opening depth"

    def test_the_ladder_tracks_the_depth_down(self, rungs):
        seat = LadderSeat(rungs)
        _match((seat, LadderSeat([rungs[1]])), escalate_every=2, escalate_by=2.0)
        assert seat.switches > 0


class TestPlayElimination:
    def test_identical_ladders_split_every_pair(self):
        """One card sequence and one strategy stream per pair, ladders swapped
        between seats: identical ladders must mirror exactly, so seat and card
        luck cancel by construction rather than on average."""
        result = play_elimination(
            LadderSeat([_player(8), _player(32)]),
            LadderSeat([_player(8), _player(32)]),
            rules=RULES,
            starting_stack=20 * BB,
            num_pairs=3,
        )
        assert result.pair_samples == [0.5, 0.5, 0.5]
        assert result.a_win_rate == 0.5

    def test_a_run_is_reproducible_from_fresh_players(self):
        def run():
            return play_elimination(
                LadderSeat([_player(32)]),
                LadderSeat([_player(8)]),
                rules=RULES,
                starting_stack=20 * BB,
                num_pairs=3,
                seed=5,
            ).pair_samples

        assert run() == run()
