"""Playing a blueprint at a depth it was not cut for.

The identity case carries the whole file: at a MATCHED depth the replay
reproduces the state it was handed action for action, so the state it rebuilds
must equal the one it was given and it must never truncate. That is the only
evidence the fiction behaves as claimed when the depths genuinely differ -- there
is no oracle for the mismatched case, which is the reason it needs measuring.
"""

from __future__ import annotations

import pytest

from src.core.game.actions import Action, ActionType
from src.core.game.state import FULL_DECK
from src.pipeline.evaluation.estimators.cross_depth import (
    CrossDepthPlayer,
    nearest_legal,
    safe_default,
)
from tests.test_helpers import build_trained_test_solver

BB = 100


@pytest.fixture(scope="module")
def shallow():
    """4 bb deep."""
    return build_trained_test_solver(iterations=6, starting_stack=4 * BB)


@pytest.fixture(scope="module")
def deep():
    """12 bb deep -- the same game with three times the money."""
    return build_trained_test_solver(iterations=6, starting_stack=12 * BB)


def _opening(blueprint):
    """The first decision of a hand, with real hole cards."""
    return blueprint.rules.create_initial_state(
        starting_stack=blueprint.config.game.starting_stack,
        hole_cards=((FULL_DECK[0], FULL_DECK[1]), (FULL_DECK[2], FULL_DECK[3])),
        button=0,
    )


def _legal(blueprint, state):
    return blueprint.rules.get_legal_actions(state, action_model=blueprint.action_model)


class TestMatchedDepthIsTheIdentity:
    def test_the_replay_lands_on_our_seat_at_every_decision(self, deep):
        player = CrossDepthPlayer(deep)
        state = _opening(deep)
        for _ in range(6):
            if state.is_terminal:
                break
            player.act(state)
            state = state.apply_action(_legal(deep, state)[-1], deep.rules)
        assert player.tally.decisions > 0
        assert player.tally.truncated == 0

    def test_the_replayed_state_is_the_state_it_was_given(self, deep):
        """The identity claim, stated where it is deterministic.

        Comparing two `act` calls does NOT test this -- the blueprint SAMPLES,
        so two draws at one state differ legitimately. What must hold is that the
        public state the replay rebuilds is the one it was handed; the policy is
        then reading the same infoset by construction.
        """
        player = CrossDepthPlayer(deep)
        state = _opening(deep)
        state = state.apply_action(_legal(deep, state)[-1], deep.rules)
        rebuilt = player._replay(state)
        assert rebuilt is not None
        for field in ("street", "pot", "stacks", "board", "betting_history", "current_player"):
            assert getattr(rebuilt, field) == getattr(state, field), field

    def test_the_same_seed_gives_the_same_action(self, deep):
        # `sample_action_from_strategy` draws from GLOBAL np.random, so a duel
        # built on it is unreproducible and two arms share one stream.
        state = _opening(deep)
        first = [CrossDepthPlayer(deep, seed=11).act(state) for _ in range(4)]
        second = [CrossDepthPlayer(deep, seed=11).act(state) for _ in range(4)]
        assert first == second

    def test_nothing_is_snapped(self, deep):
        player = CrossDepthPlayer(deep)
        player.act(_opening(deep))
        assert player.tally.snapped == 0


class TestMismatchedDepth:
    def test_a_shallow_blueprint_answers_a_deep_table_legally(self, shallow, deep):
        # The whole point: `shallow` has never seen a 12 bb stack, and must still
        # return something the 12 bb game will accept.
        player = CrossDepthPlayer(shallow)
        state = _opening(deep)
        assert player.act(state) in _legal(deep, state)

    def test_a_deep_blueprint_answers_a_shallow_table_legally(self, shallow, deep):
        player = CrossDepthPlayer(deep)
        state = _opening(shallow)
        assert player.act(state) in _legal(shallow, state)

    def test_the_depth_it_reports_is_its_own_not_the_tables(self, shallow):
        assert CrossDepthPlayer(shallow).depth_in_blinds == 4.0


class TestNearestLegal:
    def test_an_offered_type_is_kept(self, deep):
        legal = _legal(deep, _opening(deep))
        folds = [a for a in legal if a.type is ActionType.FOLD]
        if not folds:
            pytest.skip("no fold at the opening spot")
        assert nearest_legal(folds[0], legal) is folds[0]

    def test_an_unreachable_size_falls_to_the_nearest_of_its_type(self, deep):
        legal = _legal(deep, _opening(deep))
        raises = [a for a in legal if a.is_aggressive()]
        if not raises:
            pytest.skip("no aggressive action at the opening spot")
        biggest = max(raises, key=lambda a: a.amount)
        # Ask for more than the table holds; the largest on offer is the answer.
        assert nearest_legal(Action(biggest.type, biggest.amount * 10), legal) is biggest

    def test_a_safe_default_is_always_legal(self, deep):
        legal = _legal(deep, _opening(deep))
        assert safe_default(legal) in legal


class TestTruncation:
    """The replay CAN fail, and the counter says so.

    MEASURED at 2 bb against 60 bb: a raise-then-jam line leaves the 2 bb game
    with no legal action to replay into, and `_replay` returns None. It does NOT
    fire at 4 bb against 12 bb -- the snap keeps the shallow game consistent --
    so the ratio here is the one that actually produces the failure.
    """

    @pytest.fixture(scope="class")
    def paper(self):
        """2 bb: barely more than the blinds."""
        return build_trained_test_solver(iterations=6, starting_stack=2 * BB)

    @pytest.fixture(scope="class")
    def tower(self):
        """60 bb: room for a line the paper stack cannot follow."""
        return build_trained_test_solver(iterations=6, starting_stack=60 * BB)

    def _long_history(self, blueprint):
        """A deep state several SMALL raises in.

        Smallest-aggressive, not first-aggressive: the first is a jam, which ends
        the hand in one action and leaves no history to diverge.
        """
        state = _opening(blueprint)
        for _ in range(6):
            legal = _legal(blueprint, state)
            aggressive = [a for a in legal if a.is_aggressive()]
            if not aggressive:
                break
            nxt = state.apply_action(min(aggressive, key=lambda a: a.amount), blueprint.rules)
            if nxt.is_terminal:
                break
            state = nxt
        return state

    def test_a_shallow_blueprint_truncates_on_a_deep_history(self, paper, tower):
        state = self._long_history(tower)
        assert not state.is_terminal
        player = CrossDepthPlayer(paper)
        player.act(state)
        assert player.tally.truncated == 1, (
            "a 2 bb stack cannot hold a 60 bb betting line; the replay must say so"
        )

    def test_it_still_returns_a_legal_action(self, paper, tower):
        # Truncation is answered with a safe default, never with an exception and
        # never from whatever seat the broken replay happened to land on.
        state = self._long_history(tower)
        assert CrossDepthPlayer(paper).act(state) in _legal(tower, state)

    def test_the_native_blueprint_does_not(self, tower):
        state = self._long_history(tower)
        player = CrossDepthPlayer(tower)
        player.act(state)
        assert player.tally.truncated == 0


class TestUnitsMustMatch:
    """The replay converts STACKS, never currency.

    Each action's amount is carried across unchanged, so a 1/2 blueprint handed a
    5/10 table reads every raise in the wrong units -- and returns a legal,
    plausible, WRONG answer. This was a live defect in an elimination experiment
    before the guard existed, so it is a measured failure and not a hypothetical.
    """

    def test_a_table_on_other_blinds_is_refused(self, deep):
        from src.core.game.rules import GameRules

        theirs = GameRules(5, 10)  # Chipzen's units against our 50/100
        state = theirs.create_initial_state(
            starting_stack=1000,
            hole_cards=((FULL_DECK[0], FULL_DECK[1]), (FULL_DECK[2], FULL_DECK[3])),
            button=0,
        )
        with pytest.raises(ValueError, match="wrong units"):
            CrossDepthPlayer(deep).act(state)

    def test_the_blueprints_own_table_is_fine(self, deep):
        CrossDepthPlayer(deep).act(_opening(deep))

    def test_a_different_stack_on_the_same_blinds_is_still_fine(self, shallow, deep):
        # Stacks are exactly what this is for; only the currency is refused.
        CrossDepthPlayer(shallow).act(_opening(deep))
