"""Unit tests for the on-tree shadow state (rigorous off-tree LBR).

These lock the mirroring rules in isolation, on constructed states — no
blueprint needed. The end-to-end structural invariants (street/player/all-in
parity, shadow never terminal first) are enforced by always-on asserts inside
``ShadowTracker`` and exercised over full hands in
``test_hunl_local_best_response.py``.
"""

from __future__ import annotations

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType, all_in, bet, call, check, fold, raises
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street
from src.pipeline.evaluation.shadow_state import ShadowTracker
from tests.test_helpers import make_test_config

RULES = GameRules(small_blind=50, big_blind=100)
HOLE = ((Card.new("As"), Card.new("Kh")), (Card.new("Qd"), Card.new("Jc")))
BOARD = (Card.new("2c"), Card.new("7d"), Card.new("9s"))


def _model() -> ActionModel:
    return ActionModel(make_test_config(small_blind=50, big_blind=100))


def _tracker() -> ShadowTracker:
    return ShadowTracker(RULES, _model())


def _initial(stack: int = 2000) -> GameState:
    return RULES.create_initial_state(starting_stack=stack, hole_cards=HOLE, button=0)


def _flop_state(
    pot: int,
    stacks: tuple[int, int],
    *,
    to_call: int = 0,
    current_player: int = 0,
    history: tuple[Action, ...] = (call(),),
    last_aggressor: int | None = None,
) -> GameState:
    return GameState(
        street=Street.FLOP,
        pot=pot,
        stacks=stacks,
        board=BOARD,
        hole_cards=HOLE,
        betting_history=history,
        button_position=0,
        current_player=current_player,
        is_terminal=False,
        to_call=to_call,
        last_aggressor=last_aggressor,
        blind_to_call=50,
    )


def _forced_diverged(shadow_state: GameState) -> ShadowTracker:
    """A tracker whose shadow is ``shadow_state`` and which considers itself
    diverged — unit-test setup for the mirror/map-back rules without driving a
    full divergent line."""
    tracker = _tracker()
    tracker.start(shadow_state)
    tracker._diverged = True
    return tracker


def _diverge_preflop(tracker: ShadowTracker, state: GameState) -> GameState:
    """Commit an on-tree real raise with a different on-tree shadow raise."""
    real_action, shadow_action = raises(200), raises(300)  # 2.5bb vs 3.5bb totals
    real_next = state.apply_action(real_action, RULES)
    tracker.commit(real_action, real_next, shadow_action)
    return real_next


class TestAliasing:
    def test_alias_until_divergence(self):
        tracker = _tracker()
        state = _initial()
        tracker.start(state)
        assert tracker.state is state
        action = raises(200)
        real_next = state.apply_action(action, RULES)
        tracker.commit(action, real_next, action)
        assert tracker.state is real_next
        assert not tracker.diverged

    def test_divergence_detaches_and_counts(self):
        tracker = _tracker()
        state = _initial()
        tracker.start(state)
        real_next = _diverge_preflop(tracker, state)
        assert tracker.diverged
        assert tracker.divergence_count == 1
        assert tracker.state is not real_next
        assert tracker.state.pot != real_next.pot
        tracker.assert_sync(real_next)  # structure stays in lockstep

    def test_commit_asserts_shadow_legality(self):
        tracker = _tracker()
        state = _initial()
        tracker.start(state)
        real_next = state.apply_action(raises(200), RULES)
        with pytest.raises(AssertionError, match="off the abstract tree"):
            tracker.commit(raises(200), real_next, raises(37))


class TestCounterpart:
    def test_passthrough_fold_check_call(self):
        real = _flop_state(300, (1000, 1000), to_call=100, current_player=1, last_aggressor=0)
        tracker = _forced_diverged(
            _flop_state(200, (1100, 1100), to_call=50, current_player=1, last_aggressor=0)
        )
        assert tracker.counterpart(real, fold()) == fold()
        assert tracker.counterpart(real, call()) == call()
        real_check = _flop_state(300, (1000, 1000))
        shadow_check = _forced_diverged(_flop_state(200, (1100, 1100)))
        assert shadow_check.counterpart(real_check, check()) == check()

    def test_all_in_mirrors_to_shadow_stack(self):
        real = _flop_state(300, (1000, 1000))
        tracker = _forced_diverged(_flop_state(200, (1150, 1050)))
        assert tracker.counterpart(real, all_in(1000)) == all_in(1150)

    def test_bet_counterpart_nearest_pot_fraction_postflop(self):
        # Real menu on pot 300: bets 99/198/375; shadow on pot 200: 66/132/250.
        real = _flop_state(300, (1000, 1000))
        tracker = _forced_diverged(_flop_state(200, (1100, 1100)))
        assert tracker.counterpart(real, bet(198)) == bet(132)  # 0.66 pot -> 0.66 pot

    def test_preflop_raise_counterpart_nearest_bb_total(self):
        # Shadow first-in totals: 250/350/500 chips (2.5/3.5/5bb). A real raise
        # to 500 total must mirror to the 5bb size, not the nearest raw amount.
        shadow = _initial()
        real = shadow.replace(pot=170, to_call=70)
        tracker = _forced_diverged(shadow)
        assert tracker.counterpart(real, raises(430)) == raises(450)

    def test_counterpart_none_when_no_same_type_action(self):
        # Shadow stacks too small for any template bet: every size converts to
        # all-in, so no BET exists on the shadow and the mirror is gated.
        real = _flop_state(300, (500, 500))
        tracker = _forced_diverged(_flop_state(200, (60, 60)))
        assert tracker.counterpart(real, bet(99)) is None


class TestOffTreeDist:
    def test_pseudo_harmonic_pair_between_on_tree_sizes(self):
        state = _flop_state(200, (1000, 1000))
        tracker = _tracker()
        tracker.start(state)  # pre-divergence: shadow is the state itself
        dist = tracker.off_tree_dist(state, bet(100))  # between 66 (0.33) and 132 (0.66)
        assert dist is not None
        actions = [action for action, _ in dist]
        assert actions == [bet(66), bet(132)]
        assert abs(sum(weight for _, weight in dist) - 1.0) < 1e-9

    def test_jam_bracket_clamps_to_largest_non_jam_size(self):
        # Shadow bets on pot 200 with 90 stacks: only bet(66) stays a true BET
        # (larger sizes convert to all-in). A jam proxy would break all-in
        # parity, so 80 clamps to the largest structure-preserving size instead
        # of interpolating against the jam.
        state = _flop_state(200, (90, 90))
        tracker = _tracker()
        tracker.start(state)
        assert tracker.off_tree_dist(state, bet(80)) == ((bet(66), 1.0),)

    def test_clamps_below_smallest_size_to_singleton(self):
        state = _flop_state(200, (90, 90))
        tracker = _tracker()
        tracker.start(state)
        assert tracker.off_tree_dist(state, bet(40)) == ((bet(66), 1.0),)

    def test_gated_when_no_same_type_candidate_exists(self):
        # 60-chip stacks: every template bet reaches the stack and becomes
        # ALL_IN, so no structure-preserving BET proxy exists at all.
        state = _flop_state(200, (60, 60))
        tracker = _tracker()
        tracker.start(state)
        assert tracker.off_tree_dist(state, bet(40)) is None

    def test_single_raise_menu_still_supports_off_tree_raises(self):
        # Facing a bet the abstract menu is [min_raise, jam]; an off-tree raise
        # between them must map to the min_raise, not be gated by the jam.
        state = _flop_state(
            300,
            (900, 1000),
            to_call=100,
            current_player=0,
            last_aggressor=1,
            history=(call(), bet(100)),
        )
        tracker = _tracker()
        tracker.start(state)
        dist = tracker.off_tree_dist(state, raises(264))
        assert dist is not None
        assert all(proxy.type == ActionType.RAISE for proxy, _ in dist)


class TestMapBack:
    def test_identity_before_divergence(self):
        state = _flop_state(200, (1000, 1000))
        tracker = _tracker()
        tracker.start(state)
        assert tracker.map_back(state, bet(66)) == bet(66)

    def test_scales_bet_by_pot_ratio(self):
        real = _flop_state(300, (1000, 1000))
        tracker = _forced_diverged(_flop_state(200, (1100, 1100)))
        assert tracker.map_back(real, bet(132)) == bet(198)

    def test_call_promotes_to_all_in_on_covering_jam(self):
        # Real: facing a jam that exactly covers (to_call == stack); calling is
        # encoded ALL_IN and is terminal — the allowed real-terminal-first case.
        real = _flop_state(
            600,
            (200, 0),
            to_call=200,
            current_player=0,
            last_aggressor=1,
            history=(call(), Action(ActionType.ALL_IN, 400)),
        )
        tracker = _forced_diverged(
            _flop_state(
                500,
                (300, 100),
                to_call=100,
                current_player=0,
                last_aggressor=1,
                history=(call(), bet(100)),
            )
        )
        assert tracker.map_back(real, call()) == all_in(200)

    def test_raise_promotes_to_all_in_when_stack_short(self):
        # Shadow raise 400 on (pot 200, to_call 100) is 4/3 of the after-call
        # pot; real after-call pot is 150 so the scaled raise is 200, but
        # to_call + 200 >= the 220 real stack -> ALL_IN.
        real = _flop_state(
            100,
            (220, 500),
            to_call=50,
            current_player=0,
            last_aggressor=1,
            history=(call(), bet(50)),
        )
        tracker = _forced_diverged(
            _flop_state(
                200,
                (1000, 1000),
                to_call=100,
                current_player=0,
                last_aggressor=1,
                history=(call(), bet(100)),
            )
        )
        assert tracker.map_back(real, raises(400)) == all_in(220)


class TestAllInParity:
    def test_real_side_all_in_ahead_of_shadow_is_tolerated(self):
        # A map-back promotion can put the REAL side all-in one decision before
        # the hand ends (the exploiter's response to an all-in is terminal);
        # assert_sync must tolerate that transient asymmetry.
        real = _flop_state(
            700,
            (200, 0),
            to_call=200,
            current_player=0,
            last_aggressor=1,
            history=(call(), Action(ActionType.ALL_IN, 500)),
        )
        tracker = _forced_diverged(
            _flop_state(
                500,
                (300, 100),
                to_call=100,
                current_player=0,
                last_aggressor=1,
                history=(call(), bet(100)),
            )
        )
        tracker.assert_sync(real)

    def test_shadow_side_all_in_without_real_is_a_hard_error(self):
        real = _flop_state(
            500,
            (300, 100),
            to_call=100,
            current_player=0,
            last_aggressor=1,
            history=(call(), bet(100)),
        )
        tracker = _forced_diverged(
            _flop_state(
                700,
                (200, 0),
                to_call=200,
                current_player=0,
                last_aggressor=1,
                history=(call(), Action(ActionType.ALL_IN, 500)),
            )
        )
        with pytest.raises(AssertionError, match="shadow all-in without real all-in"):
            tracker.assert_sync(real)


class TestChanceAndBreak:
    def test_mirror_chance_copies_board_only(self):
        tracker = _tracker()
        state = _initial()
        tracker.start(state)
        real_next = _diverge_preflop(tracker, state)
        # Both sides close the street with a call -> flop chance node.
        real_pre_deal = real_next.apply_action(call(), RULES)
        tracker.commit(call(), real_pre_deal, call())
        real_after = real_pre_deal.replace(board=BOARD)
        tracker.mirror_chance(real_after)
        assert tracker.state.board == BOARD
        assert tracker.state.pot != real_after.pot  # shadow chips stay its own
        tracker.assert_sync(real_after)

    def test_mark_broken_re_aliases_real(self):
        tracker = _tracker()
        state = _initial()
        tracker.start(state)
        _diverge_preflop(tracker, state)
        real_next = state.apply_action(raises(200), RULES)
        tracker.mark_broken(real_next)
        assert tracker.broken
        assert tracker.state is real_next
        # Post-break commits keep aliasing the real line.
        real_after = real_next.apply_action(call(), RULES)
        tracker.commit(call(), real_after, call())
        assert tracker.state is real_after
