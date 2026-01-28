"""Unit tests for the depth-limited lookahead scorer.

All dependencies are injected (scripted opponent vectors, constant equity), so
every expected value here is hand-computed exact arithmetic — no trained
blueprint, no Monte Carlo. Engine-level behavior (determinism, parallel ==
serial, changes-play-vs-myopic) lives in ``test_hunl_local_best_response.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, all_in, bet, call, check, fold, raises
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street
from src.engine.search.range_inference import NUM_COMBOS
from src.pipeline.evaluation.lookahead_scorer import (
    BlueprintDistMemo,
    LookaheadScorer,
    committed_chips,
)
from src.pipeline.evaluation.shadow_state import MenuCandidate
from tests.test_helpers import make_test_config

RULES = GameRules(small_blind=50, big_blind=100)
MODEL = ActionModel(make_test_config(small_blind=50, big_blind=100))
HOLE = ((Card.new("As"), Card.new("Kh")), (Card.new("Qd"), Card.new("Jc")))
RIVER_BOARD = (Card.new("2c"), Card.new("7d"), Card.new("9s"), Card.new("4h"), Card.new("Ts"))


def _river_state(
    pot: int,
    stacks: tuple[int, int],
    *,
    to_call: int = 0,
    current_player: int = 0,
    history: tuple[Action, ...] = (call(), check(), check(), check(), check(), check(), check()),
    last_aggressor: int | None = None,
) -> GameState:
    return GameState(
        street=Street.RIVER,
        pot=pot,
        stacks=stacks,
        board=RIVER_BOARD,
        hole_cards=HOLE,
        betting_history=history,
        button_position=0,
        current_player=current_player,
        is_terminal=False,
        to_call=to_call,
        last_aggressor=last_aggressor,
        blind_to_call=50,
    )


class ScriptedOpponent:
    """action_matrix stub: per-combo vectors from a (state, legal) -> dict fn."""

    def __init__(self, script):
        self.script = script

    def action_matrix(self, state: GameState, actor: int):
        legal = RULES.get_legal_actions(state, action_model=MODEL)
        probs = self.script(state, legal)
        vecs = {action: np.full(NUM_COMBOS, probs.get(action, 0.0)) for action in legal}
        return legal, vecs


def _scorer(script, *, depth: int, equity: float = 1.0) -> LookaheadScorer:
    return LookaheadScorer(
        blueprint_model=ScriptedOpponent(script),
        rules=RULES,
        action_model=MODEL,
        is_chance_node=lambda state: False,  # river-only trees in these tests
        equity_fn=lambda hand, board, weights: equity,
        depth=depth,
    )


def _point_belief() -> np.ndarray:
    belief = np.zeros(NUM_COMBOS)
    belief[0] = 1.0
    return belief


def _on_tree(action: Action) -> MenuCandidate:
    return MenuCandidate(action, ((action, 1.0),))


def _fold_call_script(fold_prob: float):
    """Opponent folds/calls a bet; checks behind otherwise. Never raises."""

    def script(state: GameState, legal: list[Action]) -> dict[Action, float]:
        if state.to_call > 0:
            return {fold(): fold_prob, call(): 1.0 - fold_prob}
        return {check(): 1.0}

    return script


class TestCommittedChips:
    def test_all_action_types(self):
        state = _river_state(200, (400, 400), to_call=66, current_player=1, last_aggressor=0)
        assert committed_chips(state, fold()) == 0
        assert committed_chips(state, call()) == 66
        assert committed_chips(state, raises(100)) == 166
        assert committed_chips(state, all_in(400)) == 400
        lead = _river_state(200, (400, 400))
        assert committed_chips(lead, check()) == 0
        assert committed_chips(lead, bet(66)) == 66


class TestLeafAndFoldArithmetic:
    def test_bet_fold_call_mixture_hand_computed(self):
        # pot 200, bet 66; opponent folds 0.9 (win pot: +200) or calls 0.1
        # (river call is terminal; eq=1 -> (200+132)*1 - 66 = 266).
        state = _river_state(200, (400, 400))
        scorer = _scorer(_fold_call_script(0.9), depth=2, equity=1.0)
        value = scorer.score(state, state, 1, HOLE[0], _point_belief(), _on_tree(bet(66)))
        assert value == pytest.approx(0.9 * 200 + 0.1 * 266)

    def test_fold_candidate_scores_zero(self):
        state = _river_state(200, (400, 400), to_call=66, current_player=0, last_aggressor=1)
        scorer = _scorer(_fold_call_script(0.5), depth=2)
        assert scorer.score(state, state, 1, HOLE[0], _point_belief(), _on_tree(fold())) == 0.0

    def test_call_closes_and_uses_real_to_call(self):
        # Facing 66 on the river: call -> terminal; eq=1 -> pot(200)+66 won,
        # 66 risked: 266*1 - 66 = 200 == wp*pot - (1-wp)*to_call at wp=1.
        state = _river_state(
            200,
            (400, 466),
            to_call=66,
            current_player=0,
            last_aggressor=1,
            history=(call(), bet(66)),
        )
        scorer = _scorer(_fold_call_script(0.5), depth=2, equity=1.0)
        value = scorer.score(state, state, 1, HOLE[0], _point_belief(), _on_tree(call()))
        assert value == pytest.approx(200.0)

    def test_pending_call_completion_at_budget_leaf(self):
        # depth=1: after CHECK the opponent jams (400); the exploiter node has
        # no budget, so the leaf completes the pending call: pot 200+400+400,
        # mine 400 -> eq=1 gives 600.
        def script(state: GameState, legal: list[Action]) -> dict[Action, float]:
            return {all_in(400): 1.0} if state.to_call == 0 else {call(): 1.0}

        state = _river_state(200, (400, 400))
        scorer = _scorer(script, depth=1, equity=1.0)
        value = scorer.score(state, state, 1, HOLE[0], _point_belief(), _on_tree(check()))
        assert value == pytest.approx(600.0)


class TestDepthOneEqualsMyopic:
    def test_aggressive_candidate_matches_myopic_formula(self):
        # With a fold/call-only opponent response, depth-1 lookahead reduces
        # analytically to the myopic formula fp*pot + (1-fp)*(wp*(pot+s)-(1-wp)*s).
        state = _river_state(200, (400, 400))
        fp, wp, pot, size = 0.7, 0.85, 200.0, 132.0
        scorer = _scorer(_fold_call_script(fp), depth=1, equity=wp)
        value = scorer.score(state, state, 1, HOLE[0], _point_belief(), _on_tree(bet(132)))
        myopic = fp * pot + (1.0 - fp) * (wp * (pot + size) - (1.0 - wp) * size)
        assert value == pytest.approx(myopic)


class TestTrapSpot:
    """The reason this scorer exists: myopic prefers the small bet, the
    lookahead sees that checking induces a jam it can call with the nuts."""

    @staticmethod
    def _script(state: GameState, legal: list[Action]) -> dict[Action, float]:
        if state.to_call > 0:
            return {fold(): 0.9, call(): 0.1}  # folds out vs a lead
        return {all_in(400): 1.0}  # jams after a check

    def test_lookahead_ranks_check_above_bet(self):
        state = _river_state(200, (400, 400))
        belief = _point_belief()
        scorer = _scorer(self._script, depth=2, equity=1.0)
        check_value = scorer.score(state, state, 1, HOLE[0], belief, _on_tree(check()))
        bet_value = scorer.score(state, state, 1, HOLE[0], belief, _on_tree(bet(66)))
        # Check -> jam -> call: 1000 pot, 400 risked = +600.
        assert check_value == pytest.approx(600.0)
        # Bet 66: 0.9*200 + 0.1*266 = 206.6 — what myopic would (correctly)
        # compute for the bet, but myopic scores CHECK at wp*pot = 200 and
        # therefore picks the bet; the lookahead flips the ranking.
        assert bet_value == pytest.approx(206.6)
        myopic_check = 1.0 * 200.0
        assert bet_value > myopic_check  # myopic prefers the bet...
        assert check_value > bet_value  # ...the lookahead prefers the trap


class TestChipSpaceRule:
    def test_deeper_chips_scale_by_pot_ratio(self):
        # Real pot 150, shadow pot 100 -> ratio 1.5. Root uses REAL chips
        # (bet 75); the opponent's shadow call of the bet(33) proxy adds
        # 1.5*33 = 49.5 real chips.
        real = _river_state(150, (400, 400))
        shadow = _river_state(100, (425, 425))
        candidate = MenuCandidate(bet(75), ((bet(33), 1.0),))
        eq = 0.8
        scorer = _scorer(_fold_call_script(0.5), depth=2, equity=eq)
        value = scorer.score(real, shadow, 1, HOLE[0], _point_belief(), candidate)
        fold_branch = 150.0  # pot - mine = (150+75) - 75
        call_branch = eq * (150.0 + 75.0 + 1.5 * 33.0) - 75.0
        assert value == pytest.approx(0.5 * fold_branch + 0.5 * call_branch)

    def test_proxy_mixture_is_weight_averaged(self):
        state = _river_state(200, (400, 400))
        candidate = MenuCandidate(bet(100), ((bet(66), 0.5), (bet(132), 0.5)))
        scorer = _scorer(_fold_call_script(1.0), depth=2, equity=1.0)
        # Opponent always folds: value = pot regardless of proxy, but the walk
        # must traverse both proxies and weight-average.
        value = scorer.score(state, state, 1, HOLE[0], _point_belief(), candidate)
        assert value == pytest.approx(200.0)


class TestDepthValidation:
    def test_depth_below_one_rejected(self):
        with pytest.raises(ValueError, match="depth must be >= 1"):
            _scorer(_fold_call_script(0.5), depth=0)


class TestBlueprintDistMemo:
    def test_key_distinguishes_chip_configurations(self):
        infoset_key = ("same", "key")
        a = _river_state(200, (400, 400))
        b = _river_state(300, (350, 350))
        assert BlueprintDistMemo.key(infoset_key, a) != BlueprintDistMemo.key(infoset_key, b)

    def test_freeze_on_full_keeps_first_entries(self):
        memo = BlueprintDistMemo(max_entries=1)
        state = _river_state(200, (400, 400))
        key_a = BlueprintDistMemo.key("a", state)
        key_b = BlueprintDistMemo.key("b", state)
        memo.put(key_a, {check(): 1.0}, False)
        memo.put(key_b, {check(): 1.0}, True)
        assert len(memo) == 1
        assert memo.get(key_a) == ({check(): 1.0}, False)
        assert memo.get(key_b) is None

    def test_hit_and_miss_counters(self):
        memo = BlueprintDistMemo()
        state = _river_state(200, (400, 400))
        key = BlueprintDistMemo.key("k", state)
        assert memo.get(key) is None
        memo.put(key, {check(): 0.4, bet(66): 0.6}, True)
        assert memo.get(key) == ({check(): 0.4, bet(66): 0.6}, True)
        assert memo.hits == 1 and memo.misses == 1
