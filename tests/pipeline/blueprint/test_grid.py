"""The per-combo grid: complete, board-aware, and honest about what is untrained.

The load-bearing claim is that combos sharing a bucket are identical *by
construction* -- the grid is computed one row per bucket precisely because that
is true. If it ever stopped being true the optimisation would be silently wrong,
so it is asserted directly here rather than assumed.
"""

from __future__ import annotations

import pytest

from src.core.game.actions import Action, ActionType
from src.core.game.state import Card
from src.engine.search.range_inference import ALL_COMBOS, NUM_COMBOS
from src.pipeline.blueprint.grid import strategy_grid
from src.pipeline.blueprint.paths import PathError, encode_path, replay
from tests.test_helpers import build_trained_test_solver

FLOP = (Card.new("2c"), Card.new("7d"), Card.new("9h"))


@pytest.fixture(scope="module")
def blueprint():
    """Trained enough that some infosets are visited and plenty are not."""
    return build_trained_test_solver(iterations=40)


@pytest.fixture(scope="module")
def root(blueprint):
    return replay(blueprint, "")


@pytest.fixture(scope="module")
def flop_node(blueprint):
    """A line that gets to the flop, so the board actually blocks combos."""
    call = next(a for a in replay(blueprint, "").legal_actions if a.type is ActionType.CALL)
    path = encode_path((call, Action(ActionType.CHECK, 0)))
    return replay(blueprint, path, board=FLOP)


class TestTheGridIsComplete:
    def test_every_combo_gets_a_verdict(self, blueprint, root):
        grid = strategy_grid(blueprint, root)

        assert len(grid.combo_buckets) == NUM_COMBOS

    def test_preflop_blocks_nothing(self, blueprint, root):
        grid = strategy_grid(blueprint, root)

        assert grid.blocked == 0
        assert all(bucket >= 0 for bucket in grid.combo_buckets)

    def test_the_board_blocks_exactly_the_combos_sharing_a_card(self, blueprint, flop_node):
        grid = strategy_grid(blueprint, flop_node)
        board = set(FLOP)
        expected = sum(1 for c1, c2 in ALL_COMBOS if c1 in board or c2 in board)

        assert grid.blocked == expected
        assert grid.blocked > 0, "a three-card flop must block something"

    def test_a_blocked_combo_is_marked_rather_than_dropped(self, blueprint, flop_node):
        """Dropping them would shift every later index in a rendered grid."""
        grid = strategy_grid(blueprint, flop_node)
        blocked = [i for i, bucket in enumerate(grid.combo_buckets) if bucket < 0]

        assert len(blocked) == grid.blocked
        assert grid.for_combo(blocked[0]) is None

    def test_the_actions_match_what_is_playable_there(self, blueprint, root):
        grid = strategy_grid(blueprint, root)

        assert len(grid.actions) == len(root.legal_actions)
        assert grid.actor == root.actor


class TestBucketsAreTheUnit:
    def test_combos_in_one_bucket_share_a_row(self, blueprint, root):
        """The claim the per-bucket computation rests on."""
        grid = strategy_grid(blueprint, root)
        seen: dict[int, int] = {}
        for index, bucket in enumerate(grid.combo_buckets):
            if bucket < 0:
                continue
            if bucket in seen:
                assert grid.for_combo(index) is grid.for_combo(seen[bucket])
            else:
                seen[bucket] = index

        assert len(seen) < NUM_COMBOS, "an abstraction that separated every combo proves nothing"

    def test_there_are_fewer_buckets_than_combos(self, blueprint, root):
        grid = strategy_grid(blueprint, root)

        assert 0 < len(grid.buckets) <= NUM_COMBOS


class TestUntrainedIsNotUniform:
    def test_an_untrained_bucket_carries_no_strategy_at_all(self, blueprint, root):
        """`trained=False` and a strategy must never coexist -- that pairing is
        exactly the confident-looking uniform this surface exists to avoid."""
        grid = strategy_grid(blueprint, root)

        for entry in grid.buckets.values():
            assert (entry.strategy is None) == (not entry.trained)

    def test_a_trained_row_is_a_distribution_over_the_offered_actions(self, blueprint, root):
        grid = strategy_grid(blueprint, root)
        trained = [e for e in grid.buckets.values() if e.trained]

        assert trained, "40 iterations should have visited at least one preflop infoset"
        for entry in trained:
            assert entry.strategy is not None
            assert len(entry.strategy) == len(grid.actions)
            assert sum(entry.strategy) == pytest.approx(1.0)
            assert all(p >= 0.0 for p in entry.strategy)

    def test_reach_count_is_reported_for_trained_rows(self, blueprint, root):
        grid = strategy_grid(blueprint, root)
        trained = [e for e in grid.buckets.values() if e.trained]

        assert any(e.reach_count > 0 for e in trained)
        assert grid.trained_buckets == len(trained)


class TestAverageAndCurrentAreDifferentQuestions:
    def test_both_are_available_and_the_choice_is_explicit(self, blueprint, root):
        average = strategy_grid(blueprint, root, use_average=True)
        current = strategy_grid(blueprint, root, use_average=False)

        assert average.combo_buckets == current.combo_buckets
        assert average.actions == current.actions


class TestRefusals:
    def test_a_terminal_node_has_no_strategy(self, blueprint):
        fold = replay(blueprint, "f")

        assert fold.actor is None
        with pytest.raises(PathError, match="ends the hand"):
            strategy_grid(blueprint, fold)
