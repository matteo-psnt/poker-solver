"""A mixture of boards is the smallest game with a chance layer in it.

The single-board kernel solves a game where both players see the river before
acting. That is enough to validate arithmetic and not enough to say anything
about real poker, whose whole difficulty is chance. These tests pin the two
places the mixture has to be *joint* across boards rather than per board — a
per-board version of either would still converge to something, just not to the
mixture's equilibrium, and nothing else would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.vector.compiled_tree import compile_tree
from src.engine.solver.vector.kernel import VectorCFR
from src.engine.solver.vector.mixture import BoardMixtureCFR
from tests.engine.solver.vector.contexts import (
    MIN_SHOWDOWN_SIGNAL,
    ordered_context,
    prefix_consistent_contexts,
    showdown_signal,
)
from tests.test_helpers import make_test_config

# A 36-card deck: 465 holdings instead of 1,081, which roughly halves the cost of
# every pass. The showdown signal stays at 0.88 (floor 0.5), so the game these
# tests solve is still a real one — the deck size is a speed knob, the bucket
# ORDERING is what carries the meaning.
DECK = 36
BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}
STACK = 12


ALL_COUNTS = {Street.PREFLOP: 169, **BUCKETS}


def _context(rng):
    """Strength-ordered, so showdowns carry signal — see ``contexts``."""
    context = ordered_context(rng, ALL_COUNTS, num_cards=DECK)
    return context, float((~context.blocks).sum())


@pytest.fixture(scope="module")
def parts():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(small_blind=1, big_blind=2)
    tree = BettingTree(rules, ActionModel(config), starting_stack=STACK, buckets_per_street=BUCKETS)
    compiled = compile_tree(tree, rules)

    rng = np.random.default_rng(23)
    made = [_context(rng) for _ in range(3)]
    contexts = [context for context, _ in made]
    pairs = float(np.mean([count for _, count in made]))
    return compiled, contexts, pairs


class TestSharedStorage:
    def test_all_boards_write_one_regret_table(self, parts):
        """One abstraction row is the same row whatever board produced it."""
        compiled, contexts, _ = parts
        mixture = BoardMixtureCFR(compiled, contexts)
        for board in mixture.boards:
            assert board.regrets is mixture.regrets
            assert board.strategy_sum is mixture.strategy_sum

    def test_an_iteration_is_the_summed_contribution_of_independent_boards(self, parts):
        """Every board reads one strategy, and their increments sum before the floor.

        Run each board alone from the same starting regrets and add the results:
        that is what one mixture iteration must equal. It fails if boards update
        the table in sequence (later boards would respond to earlier ones — a
        convergent algorithm, but not this one) and it fails if the CFR+ floor is
        applied per board, since that clips a negative another board was about to
        cancel.
        """
        compiled, contexts, _ = parts
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)

        mixture = BoardMixtureCFR(compiled, contexts)
        mixture.iterate(initial)

        # Each solo board starts from the same zeroed table the mixture did, so
        # they all regret-match the identical (uniform) strategy.
        contributions = []
        for context in contexts:
            solo = VectorCFR(compiled, context, cfr_plus=False)
            solo.iterate(initial)
            contributions.append(solo.regrets)
        expected = np.maximum(np.sum(contributions, axis=0), 0.0)

        assert np.allclose(mixture.regrets, expected, rtol=1e-4, atol=1e-3)


class TestMixtureConvergence:
    # Three boards means three full tree passes per iteration, and the ordered
    # buckets make it a real game rather than one of coin-flip showdowns, so it
    # needs 32 iterations to halve twice. Deliberately slower than the default
    # gate allows, with a bound still tight enough to catch a stall.
    @pytest.mark.timeout(40)
    def test_exploitability_falls_across_the_whole_mixture(self, parts):
        compiled, contexts, pairs = parts
        mixture = BoardMixtureCFR(compiled, contexts)
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)

        mixture.iterate(initial)
        early = mixture.exploitability(initial, pairs)
        while mixture.iteration < 32:
            mixture.iterate(initial)
        later = mixture.exploitability(initial, pairs)

        assert 0 < later < early / 4

    # Cheap on an idle box (~0.7s), but it builds the same kernels as its
    # sibling above and so scales with whatever else holds the CPU. Bounded for
    # the same reason that one is, and tight enough to still catch a stall.
    @pytest.mark.timeout(30)
    def test_a_single_board_mixture_matches_the_plain_kernel(self, parts):
        """With one board the mixture must be the single-board kernel exactly.

        Same strategy, same updates, only the CFR+ floor moved from inside the
        pass to the end of the iteration — which with one contribution is the
        same operation.
        """
        compiled, contexts, pairs = parts
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)

        mixture = BoardMixtureCFR(compiled, contexts[:1])
        plain = VectorCFR(compiled, contexts[0], cfr_plus=True)
        for _ in range(6):
            mixture.iterate(initial)
            plain.iterate(initial)

        assert np.allclose(mixture.regrets, plain.regrets, rtol=1e-4, atol=1e-3)
        assert mixture.exploitability(initial, pairs) == pytest.approx(
            plain.exploitability(initial, pairs), rel=1e-3
        )


class TestUnconstrainedBestResponse:
    """Lifting the CARD abstraction must not also lift the deal.

    The constrained responder picks one action per ``(node, bucket)``; the
    unconstrained one is told its exact holding. The gap between them is what an
    abstraction costs. But "told its holding" is not "told the runout" — a
    responder standing on the turn cannot see the river, and a per-board
    maximisation would hand it exactly that. The bug inflates the gap in the
    direction that flatters the abstraction's critic, so it is pinned here.
    """

    @pytest.fixture(scope="class")
    def shared_prefix(self, parts):
        """Two runouts identical but for the river, and one unrelated board.

        The first two are the same observation on every street up to the turn,
        so nothing above the river may tell them apart.
        """
        compiled, _, _ = parts
        prefix = [0, 1, 2, 3]
        boards = [[*prefix, 4], [*prefix, 5], [10, 11, 12, 13, 14]]
        made = prefix_consistent_contexts(boards, ALL_COUNTS, num_cards=DECK)
        # Guard the power before anything asserts behaviour: an exploitability
        # gap can only show up where showdowns actually favour someone.
        assert showdown_signal(made, ALL_COUNTS) > MIN_SHOWDOWN_SIGNAL
        pairs = float(np.mean([(~c.blocks).sum() for c in made]))
        return compiled, made, boards, pairs

    def test_boards_are_required(self, parts):
        """Without the deal the mixture cannot say who may be distinguished."""
        compiled, contexts, pairs = parts
        mixture = BoardMixtureCFR(compiled, contexts)
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)
        mixture.iterate(initial)

        with pytest.raises(ValueError, match="board cards"):
            mixture.exploitability(initial, pairs, unconstrained=True)

    def test_partition_merges_boards_until_the_card_that_separates_them(self, shared_prefix):
        """Two boards differing only on the river are one observation until it."""
        compiled, contexts, boards, _ = shared_prefix
        mixture = BoardMixtureCFR(compiled, contexts, boards=boards)

        assert [sorted(g) for g in mixture._visible_partition(Street.PREFLOP)] == [[0, 1, 2]]
        for street in (Street.FLOP, Street.TURN):
            groups = sorted(sorted(g) for g in mixture._visible_partition(street))
            assert groups == [[0, 1], [2]], f"{street} must not separate a shared prefix"
        assert sorted(sorted(g) for g in mixture._visible_partition(Street.RIVER)) == [
            [0],
            [1],
            [2],
        ]

    # Six full best-response passes over three boards. It sits ~3s against the
    # 5s default, which is close enough to lose under 12-worker contention.
    @pytest.mark.timeout(30)
    def test_seeing_the_future_is_worth_more_than_seeing_your_cards(
        self, shared_prefix, monkeypatch
    ):
        """The naive per-board max is strictly larger, so it is not the same number.

        Monkeypatching the partition to singletons IS the old per-board
        behaviour, so this compares the two directly on one trained strategy. If
        they were equal the fix would be measuring nothing and this test would
        be free to delete.
        """
        compiled, contexts, boards, pairs = shared_prefix
        mixture = BoardMixtureCFR(compiled, contexts, boards=boards)
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)
        for _ in range(12):
            mixture.iterate(initial)

        constrained = mixture.exploitability(initial, pairs)
        honest = mixture.exploitability(initial, pairs, unconstrained=True)

        singletons = [[i] for i in range(mixture.num_boards)]
        monkeypatch.setattr(mixture, "_visible_partition", lambda street: singletons)
        clairvoyant = mixture.exploitability(initial, pairs, unconstrained=True)

        assert constrained <= honest, "knowing your own cards cannot hurt"
        assert honest < clairvoyant, "seeing the runout early must be worth something"


class TestPerBoardDecomposition:
    """The un-collapsed score, which is what lets a comparison carry an interval.

    The aggregate must be EXACTLY the mean of the parts. If it were merely close,
    the parts would be a second estimate of the same quantity rather than the one
    estimate un-collapsed, and differencing them across arms would be measuring
    the decomposition instead of the strategies.
    """

    @pytest.fixture(scope="class")
    def scored(self, parts):
        compiled, _, _ = parts
        boards = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
        contexts = prefix_consistent_contexts(boards, ALL_COUNTS, num_cards=DECK)
        assert showdown_signal(contexts, ALL_COUNTS) > MIN_SHOWDOWN_SIGNAL
        pairs = float(np.mean([(~c.blocks).sum() for c in contexts]))
        mixture = BoardMixtureCFR(compiled, contexts, boards=boards)
        initial = np.ones(contexts[0].num_hands, dtype=np.float32)
        mixture.iterate(initial)
        return mixture, initial, pairs, len(boards)

    @pytest.mark.parametrize("unconstrained", [False, True])
    def test_the_parts_average_to_the_whole(self, scored, unconstrained: bool) -> None:
        mixture, initial, pairs, count = scored

        whole = mixture.exploitability(initial, pairs, unconstrained=unconstrained)
        per_board = mixture.exploitability_per_board(initial, pairs, unconstrained=unconstrained)

        assert per_board.shape == (count,)
        assert float(per_board.mean()) == pytest.approx(whole, rel=1e-9)

    def test_the_parts_differ_from_each_other(self, scored) -> None:
        """Guards the point of the exercise: a constant decomposition would carry
        no information about its own precision, and every interval built on it
        would be zero-width and wrong."""
        mixture, initial, pairs, _ = scored

        per_board = mixture.exploitability_per_board(initial, pairs, unconstrained=True)

        assert float(per_board.std()) > 0.0
