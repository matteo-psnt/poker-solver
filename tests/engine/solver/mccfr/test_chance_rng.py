"""The dealing helpers' randomness: injectable, and unchanged when it isn't.

Two properties, and the first is the one that would be expensive to get wrong.
Training passes no ``rng`` and must keep drawing from the process-global
``random`` in the same order it always has -- a shifted stream would move every
golden number at once, which reads as a lineage break rather than as the
refactor that caused it.

The second is why the parameter exists: a process dealing several hands at once
needs each to draw from its own source, or no session is reproducible from its
own seed and their draws interleave by arrival time.
"""

from __future__ import annotations

import random

from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street
from src.engine.solver.mccfr.chance import (
    deal_remaining_cards,
    draw_cards,
    sample_chance_outcome,
)


def _preflop_state() -> GameState:
    """Blinds posted, holes fixed so only what gets drawn varies between runs."""
    rules = GameRules(small_blind=1, big_blind=2)
    return rules.create_initial_state(
        starting_stack=200,
        hole_cards=((Card.new("As"), Card.new("Kd")), (Card.new("Qh"), Card.new("Jc"))),
        button=0,
    )


def _awaiting_flop() -> GameState:
    """The mid-deal state a chance node actually is: on the flop, board not yet dealt.

    Built with ``validate=False`` because the validator enforces the *settled*
    invariant (a flop shows three cards) and this is the transient the dealer is
    called on -- the same shape ``sample_chance_outcome`` branches on.
    """
    return _preflop_state().replace(validate=False, street=Street.FLOP, board=(), to_call=0)


def _fold_terminal() -> GameState:
    """A hand that ended on a fold. ``ended_by_fold`` is derived from the history,
    so it has to be reached by folding rather than by setting a flag."""
    state = _preflop_state()
    rules = GameRules(small_blind=1, big_blind=2)
    return state.apply_action(Action(ActionType.FOLD, 0), rules)


class TestDefaultStreamIsUnchanged:
    """No ``rng`` means the module-global ``random``, called exactly as before."""

    def test_draw_cards_follows_the_global_seed(self):
        state = _preflop_state()

        random.seed(1234)
        first = draw_cards(state, 3)
        random.seed(1234)
        second = draw_cards(state, 3)

        assert first == second, "seeding the global must still determine the deal"

    def test_the_two_paths_are_the_same_algorithm(self):
        """Seeding the global and passing an equally-seeded ``Random`` must agree.

        The module-level functions are bound to a hidden ``Random`` instance, so
        this is the check that injection did not quietly substitute a different
        generator for the one training has always drawn from.
        """
        state = _preflop_state()

        random.seed(99)
        from_the_global = draw_cards(state, 3)
        from_an_instance = draw_cards(state, 3, random.Random(99))

        assert from_the_global == from_an_instance


class TestAnExplicitSourceIsolates:
    """An injected ``Random`` is the only thing that determines its own draws."""

    def test_two_sources_with_one_seed_agree(self):
        state = _preflop_state()

        assert draw_cards(state, 3, random.Random(7)) == draw_cards(state, 3, random.Random(7))

    def test_a_seeded_source_ignores_the_global(self):
        """The point of threading rather than reseeding: no cross-talk either way."""
        state = _preflop_state()

        random.seed(1)
        mine = draw_cards(state, 3, random.Random(42))
        random.seed(2)
        mine_again = draw_cards(state, 3, random.Random(42))

        assert mine == mine_again

    def test_interleaved_sessions_do_not_disturb_each_other(self):
        """Two sessions dealing in turn, as concurrent requests would."""
        state = _preflop_state()
        undisturbed = random.Random(5)
        alone = [draw_cards(state, 1, undisturbed) for _ in range(3)]

        rng, other = random.Random(5), random.Random(6)
        interleaved = []
        for _ in range(3):
            interleaved.append(draw_cards(state, 1, rng))
            draw_cards(state, 1, other)

        assert interleaved == alone


class TestTheDealtCardsAreLegal:
    """Injection must not weaken the constraint that a deal avoids known cards."""

    def test_sample_chance_outcome_avoids_hole_cards(self):
        state = _awaiting_flop()
        dealt = sample_chance_outcome(state, random.Random(3))

        known = {c for hand in state.hole_cards for c in hand}
        assert len(dealt.board) == 3
        assert not known.intersection(dealt.board)
        assert len(set(dealt.board)) == 3

    def test_deal_remaining_cards_completes_to_five(self):
        state = _awaiting_flop().replace(
            validate=False,
            board=(Card.new("2c"), Card.new("7d"), Card.new("9h")),
            street=Street.FLOP,
        )
        completed = deal_remaining_cards(state, random.Random(11))

        assert len(completed.board) == 5
        assert len(set(completed.board)) == 5
        assert completed.street == Street.RIVER

    def test_a_fold_is_returned_untouched_whatever_the_source(self):
        """Folds skip dealing entirely, so they must not consume from the source."""
        state = _fold_terminal()
        rng = random.Random(4)

        assert deal_remaining_cards(state, rng) is state
        assert rng.randrange(52) == random.Random(4).randrange(52), "nothing was drawn"
