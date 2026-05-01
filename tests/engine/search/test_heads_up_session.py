"""One hand against the blueprint: pausing, dealing, and reproducibility.

The property worth the most here is that a seed fixes the WHOLE hand. It did not
before -- the board came from the process-global `random`, so two sessions in one
process interleaved their draws and no hand could be replayed. A server that
holds several sessions makes that a correctness question rather than a
convenience one.
"""

from __future__ import annotations

import pytest

from src.core.game.actions import Action, ActionType
from src.engine.search.heads_up_session import HeadsUpHand
from tests.test_helpers import build_trained_test_solver


@pytest.fixture(scope="module")
def blueprint():
    return build_trained_test_solver(iterations=40)


def play_out(hand: HeadsUpHand, *, prefer: ActionType = ActionType.CALL) -> HeadsUpHand:
    """Drive the human seat to a terminal, taking ``prefer`` where it is offered."""
    while not hand.is_over:
        legal = hand.legal_actions()
        assert legal, "not over, but the human has nothing to do"
        chosen = next((a for a in legal if a.type is prefer), legal[0])
        hand.submit(chosen)
    return hand


class TestTheHandPauses:
    def test_it_stops_on_the_human_and_not_before(self, blueprint):
        hand = HeadsUpHand(blueprint, human_seat=0, button=0, seed=1)

        assert hand.is_over or hand.state.current_player == 0
        assert hand.legal_actions()

    def test_the_bot_seat_never_asks_the_caller(self, blueprint):
        hand = HeadsUpHand(blueprint, human_seat=1, button=0, seed=2)

        while not hand.is_over:
            assert hand.state.current_player == 1
            hand.submit(hand.legal_actions()[0])

    def test_acting_after_the_hand_ends_is_refused(self, blueprint):
        hand = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=3))

        with pytest.raises(ValueError, match="already over"):
            hand.submit(Action(ActionType.FOLD, 0))


class TestReproducibility:
    def test_one_seed_fixes_the_whole_hand(self, blueprint):
        """Hole cards, every board card, and each bot sample."""
        first = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=7))
        second = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=7))

        assert first.state.board == second.state.board
        assert first.state.hole_cards == second.state.hole_cards
        assert first.payoffs == second.payoffs
        assert [(e.actor, e.action_type, e.amount) for e in first.log] == [
            (e.actor, e.action_type, e.amount) for e in second.log
        ]

    def test_interleaved_hands_do_not_disturb_each_other(self, blueprint):
        """Two sessions in flight, as two browser tabs would be."""
        alone = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=11))

        mine = HeadsUpHand(blueprint, human_seat=0, button=0, seed=11)
        other = HeadsUpHand(blueprint, human_seat=0, button=0, seed=99)
        while not mine.is_over:
            mine.submit(
                next(
                    (a for a in mine.legal_actions() if a.type is ActionType.CALL),
                    mine.legal_actions()[0],
                )
            )
            if not other.is_over:
                other.submit(other.legal_actions()[0])

        assert mine.state.board == alone.state.board
        assert mine.payoffs == alone.payoffs

    def test_different_seeds_give_different_hands(self, blueprint):
        seeds = [HeadsUpHand(blueprint, human_seat=0, button=0, seed=s) for s in range(8)]

        assert len({h.state.hole_cards for h in seeds}) > 1


class TestSettlement:
    def test_the_hand_settles_to_a_zero_sum(self, blueprint):
        hand = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=5))

        assert hand.payoffs is not None
        assert hand.payoffs[0] + hand.payoffs[1] == pytest.approx(0.0)
        assert hand.human_payoff() == hand.payoffs[0]

    def test_a_showdown_runs_the_board_out_to_five(self, blueprint):
        for seed in range(25):
            hand = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=seed))
            if hand.showdown:
                assert len(hand.state.board) == 5
                return
        pytest.fail("25 called-down hands produced no showdown")

    def test_payoffs_are_refused_before_the_end(self, blueprint):
        hand = HeadsUpHand(blueprint, human_seat=0, button=0, seed=4)
        if hand.is_over:
            pytest.skip("this seed ended before the human acted")

        with pytest.raises(ValueError, match="only defined once"):
            hand.human_payoff()


class TestTheUntrainedSignal:
    def test_every_bot_decision_is_accounted_for(self, blueprint):
        hand = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=6))
        bot_events = [e for e in hand.log if e.actor == "bot"]

        assert len(bot_events) == hand.bot_decisions
        assert hand.bot_untrained_decisions == sum(1 for e in bot_events if e.untrained)
        assert hand.bot_untrained_decisions <= hand.bot_decisions

    def test_a_trained_decision_carries_the_mix_it_sampled(self, blueprint):
        """The mix is what makes a post-hand reveal possible."""
        found = False
        for seed in range(12):
            hand = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=seed))
            for event in hand.log:
                if event.actor == "bot" and not event.untrained:
                    assert event.mix is not None
                    assert sum(weight for _, weight in event.mix) == pytest.approx(1.0)
                    found = True
        assert found, "no trained bot decision in 12 hands"

    def test_an_untrained_fallback_carries_no_mix(self, blueprint):
        """There was no distribution -- which is the thing worth surfacing."""
        for seed in range(12):
            hand = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=seed))
            for event in hand.log:
                if event.actor == "bot" and event.untrained:
                    assert event.mix is None

    def test_human_moves_are_never_marked_untrained(self, blueprint):
        hand = play_out(HeadsUpHand(blueprint, human_seat=0, button=0, seed=8))

        assert all(not e.untrained and e.mix is None for e in hand.log if e.actor == "human")


class TestConstruction:
    @pytest.mark.parametrize(
        ("seat", "button", "names"),
        [(2, 0, "human_seat"), (0, 5, "button"), (-1, 0, "human_seat")],
    )
    def test_a_bad_seat_or_button_is_refused(self, blueprint, seat, button, names):
        """The message names the offending argument -- both are plain ints, so a
        bare 'invalid' leaves no way to tell which one was wrong."""
        with pytest.raises(ValueError, match=names):
            HeadsUpHand(blueprint, human_seat=seat, button=button, seed=1)
