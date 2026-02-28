"""Tests for the interactive heads-up hand driver."""

import numpy as np

from src.core.game.actions import Action, ActionType
from src.engine.search.heads_up_session import HeadsUpHand
from tests.test_helpers import build_trained_test_solver


def _pick_passive(actions: tuple[Action, ...]) -> Action:
    """Never fold: check/call when possible, otherwise call the jam (ALL_IN).

    Facing an all-in, the "call" is itself encoded as an ALL_IN action (calling a
    jam), so the passive move is the non-fold action rather than a CALL.
    """
    for action in actions:
        if action.type in (ActionType.CHECK, ActionType.CALL):
            return action
    for action in actions:
        if action.type != ActionType.FOLD:
            return action
    return actions[0]


def _play_out(hand: HeadsUpHand, chooser=_pick_passive) -> None:
    while not hand.is_over:
        hand.submit(chooser(hand.legal_actions()))


class TestHeadsUpHand:
    def test_hand_terminates_with_zero_sum_payoffs(self):
        blueprint = build_trained_test_solver(30, session_id="hu-zero-sum")
        for seed in range(20):
            hand = HeadsUpHand(
                blueprint,
                human_seat=seed % 2,
                button=(seed // 2) % 2,
                rng=np.random.default_rng(seed),
            )
            _play_out(hand)

            assert hand.is_over
            assert hand.payoffs is not None
            # Heads-up is zero-sum: one seat's win is the other's loss.
            assert hand.payoffs[0] + hand.payoffs[1] == 0
            # At a showdown the board is fully run out (all-in runout completed).
            if hand.showdown:
                assert len(hand.state.board) == 5

    def test_human_fold_loses_only_the_blind(self):
        # The helper trains at sb=50/bb=100: the button posts the small blind and
        # acts first preflop.
        blueprint = build_trained_test_solver(10, session_id="hu-fold")
        hand = HeadsUpHand(blueprint, human_seat=0, button=0, rng=np.random.default_rng(1))

        assert hand.state.current_player == 0  # human (button/SB) acts first
        fold = next(a for a in hand.legal_actions() if a.type == ActionType.FOLD)
        hand.submit(fold)

        assert hand.is_over
        assert hand.state.ended_by_fold
        assert hand.human_payoff() == -50  # only the posted small blind is lost

    def test_untrained_blueprint_flags_every_bot_decision(self):
        # A 0-iteration blueprint has an empty table: every lookup misses, so the
        # bot plays a uniform-random fallback and each decision is flagged.
        blueprint = build_trained_test_solver(0, session_id="hu-untrained")
        seen_bot_decision = False
        for seed in range(10):
            hand = HeadsUpHand(
                blueprint,
                human_seat=seed % 2,
                button=0,
                rng=np.random.default_rng(seed),
            )
            _play_out(hand)
            if hand.bot_decisions > 0:
                seen_bot_decision = True
                assert hand.bot_untrained_decisions == hand.bot_decisions

        assert seen_bot_decision, "expected at least one hand where the bot acted"

    def test_log_records_actor_and_untrained(self):
        blueprint = build_trained_test_solver(0, session_id="hu-log")
        hand = HeadsUpHand(blueprint, human_seat=0, button=0, rng=np.random.default_rng(3))
        _play_out(hand)

        assert hand.log
        actors = {event.actor for event in hand.log}
        assert actors <= {"human", "bot"}
        # Human moves are never flagged untrained; untrained bot moves count matches.
        assert all(not e.untrained for e in hand.log if e.actor == "human")
        bot_untrained = sum(1 for e in hand.log if e.actor == "bot" and e.untrained)
        assert bot_untrained == hand.bot_untrained_decisions
