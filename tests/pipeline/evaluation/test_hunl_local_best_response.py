"""Tests for the HUNL Local Best Response evaluator.

The generic LBR bridge (``test_local_best_response.py``) already proves the
``LBR <= exact_BR`` and ``>= 0`` properties on Kuhn/Leduc. These tests lock the
HUNL-specific machinery that bridge never exercised: equity-vs-range, the
analytic terminal valuation over the surviving range, the off-tree action
mapping, determinism, and the end-to-end lower bound on a weak blueprint.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.core.game.actions import Action, ActionType, call, check, fold
from src.core.game.state import Card, GameState, Street
from src.engine.search.range_inference import (
    ALL_COMBOS,
    COMBO_MASKS,
    combo_index_for,
    replace_actor_hole_cards,
)
from src.engine.solver.mccfr import MCCFRSolver
from src.pipeline.evaluation.hunl_local_best_response import (
    LBRConfig,
    _deal_initial_state,
    _HUNLLocalBestResponse,
    compute_lbr_exploitability,
)
from src.pipeline.evaluation.statistics import compare_paired_samples
from tests.test_helpers import build_trained_test_solver


def _build_solver(
    iterations: int, *, starting_stack: int = 2000, session_id: str = "lbr-test"
) -> MCCFRSolver:
    return build_trained_test_solver(
        iterations, starting_stack=starting_stack, session_id=session_id
    )


def _rebuild_parallel_test_blueprint() -> MCCFRSolver:
    """Picklable factory: rebuild the deterministic test blueprint inside a worker.

    Must match the serial blueprint's params exactly (same seed/stack/iterations) so
    the strategies are identical; a unique session_id avoids shared-memory collisions.
    """
    return _build_solver(4, starting_stack=400, session_id=f"lbr-par-{os.getpid()}")


def _engine(solver: MCCFRSolver, **cfg) -> _HUNLLocalBestResponse:
    cfg.setdefault("seed", 7)
    cfg.setdefault("equity_runouts", 8)
    config = LBRConfig(**cfg)
    return _HUNLLocalBestResponse(solver, config, np.random.default_rng(config.seed))


def _combo_index(a: str, b: str) -> int:
    return combo_index_for((Card.new(a), Card.new(b)))


class TestEquity:
    """`_equity` must track real hand strength vs a range with board rollout."""

    @pytest.fixture(scope="class")
    def engine(self):
        return _engine(_build_solver(0), equity_runouts=200)

    def test_strength_ordering_on_dry_flop(self, engine):
        board = (Card.new("Kh"), Card.new("8d"), Card.new("3c"))
        uniform = np.ones(len(ALL_COMBOS))
        top_set = engine._equity((Card.new("Kd"), Card.new("Ks")), board, uniform)
        overpair = engine._equity((Card.new("As"), Card.new("Ac")), board, uniform)
        air = engine._equity((Card.new("7s"), Card.new("2h")), board, uniform)
        assert top_set > overpair > air
        assert overpair > 0.80
        assert air < 0.35

    def test_river_is_exact(self, engine):
        # On a complete board equity is a deterministic count over the range.
        river = (Card.new("Kh"), Card.new("8d"), Card.new("3c"), Card.new("Qs"), Card.new("2d"))
        uniform = np.ones(len(ALL_COMBOS))
        nut = engine._equity((Card.new("Ks"), Card.new("Kd")), river, uniform)  # trip kings
        assert nut > 0.95
        # Determinism: no runout sampling on a full board.
        again = engine._equity((Card.new("Ks"), Card.new("Kd")), river, uniform)
        assert nut == again

    def test_delta_range_equity_is_pairwise(self, engine):
        # Against a single opponent combo, equity is a clean win/tie/lose verdict.
        board = (Card.new("Kh"), Card.new("8d"), Card.new("3c"), Card.new("Qs"), Card.new("2d"))
        weights = np.zeros(len(ALL_COMBOS))
        weights[_combo_index("7c", "2s")] = 1.0  # opponent has air
        assert engine._equity((Card.new("Ks"), Card.new("Kd")), board, weights) == 1.0


def _bet_call_to_river(engine, state):
    """Drive a small-bet / call line, guaranteeing a river showdown.

    Every street ends in a CALL (a clean street boundary), so the terminal
    has a full (deterministic) 5-card board — no runout sampling — which lets
    us assert exact analytic values. (A pure check/check line would trigger a
    latent rules bug where a second consecutive check-check street fails to
    advance.)
    """
    s = state
    while not s.is_terminal:
        if engine.blueprint.is_chance_node(s):
            s = engine.blueprint.sample_chance_outcome(s)
            continue
        if s.to_call > 0:
            move = call()
        else:
            legal = engine.rules.get_legal_actions(s, action_model=engine.action_model)
            bets = [a for a in legal if a.type == ActionType.BET]
            move = min(bets, key=lambda a: a.amount)  # smallest bet keeps it non-all-in
        s = s.apply_action(move, engine.rules)
    return s


class TestTerminalValue:
    """`_terminal_value` integrates out the opponent hand analytically."""

    @pytest.fixture(scope="class")
    def engine(self):
        return _engine(_build_solver(5))

    def _bet_call_to_river(self, engine, state):
        return _bet_call_to_river(engine, state)

    def _reference_showdown_value(self, engine, terminal, lbr_player, opp, belief):
        """Belief-weighted mean of per-combo payoffs over the surviving range."""
        known = engine._known_mask(terminal, opp)
        weights = np.where((COMBO_MASKS & known) == 0, belief, 0.0)
        total = weights.sum()
        acc = 0.0
        for idx in np.nonzero(weights)[0]:
            payoff = replace_actor_hole_cards(
                terminal, actor=opp, combo=ALL_COMBOS[idx]
            ).get_payoff(lbr_player, engine.rules)
            acc += float(weights[idx]) * float(payoff)
        return acc / float(total)

    def test_showdown_delta_belief_equals_exact_payoff(self, engine):
        """A point-mass belief must reproduce that exact hand's payoff."""
        rng = np.random.default_rng(3)
        for _ in range(3):
            terminal = self._bet_call_to_river(engine, _deal_initial_state(engine, 2000, 0, rng))
            assert engine._is_showdown(terminal) and len(terminal.board) == 5
            known = engine._known_mask(terminal, opp=1)
            idx = next(i for i in range(len(ALL_COMBOS)) if not (COMBO_MASKS[i] & known))
            belief = np.zeros(len(ALL_COMBOS))
            belief[idx] = 1.0
            expected = float(
                replace_actor_hole_cards(terminal, actor=1, combo=ALL_COMBOS[idx]).get_payoff(
                    0, engine.rules
                )
            )
            assert engine._terminal_value(terminal, 0, 1, belief) == expected

    @pytest.mark.timeout(30)
    def test_showdown_range_value_matches_weighted_mean(self, engine):
        """The range-weighted showdown value equals the direct weighted mean."""
        rng = np.random.default_rng(11)
        terminal = self._bet_call_to_river(engine, _deal_initial_state(engine, 2000, 0, rng))
        belief = engine._initial_belief(terminal, opp=1)
        got = engine._terminal_value(terminal, 0, 1, belief)
        expected = self._reference_showdown_value(engine, terminal, 0, 1, belief)
        assert got == pytest.approx(expected)

    def test_fold_terminal_is_card_independent(self, engine):
        """Fold terminals return the plain payoff regardless of belief."""
        rng = np.random.default_rng(7)
        state = _deal_initial_state(engine, 2000, 0, rng)
        # SB folds preflop immediately.
        terminal = state.apply_action(fold(), engine.rules)
        assert terminal.is_terminal and not engine._is_showdown(terminal)
        belief = engine._initial_belief(terminal, opp=1)
        assert engine._terminal_value(terminal, 0, 1, belief) == float(
            terminal.get_payoff(0, engine.rules)
        )


class TestAllInRunoutAveraging:
    """Early all-in terminals average the analytic value over board runouts."""

    @pytest.fixture(scope="class")
    def engine(self):
        return _engine(_build_solver(5))

    @staticmethod
    def _strip_river(terminal) -> GameState:
        """Rewind a river terminal to a 4-card 'all-in' terminal (same pot/stacks)."""
        return GameState(
            street=Street.TURN,
            pot=terminal.pot,
            stacks=terminal.stacks,
            board=terminal.board[:4],
            hole_cards=terminal.hole_cards,
            betting_history=terminal.betting_history,
            button_position=terminal.button_position,
            current_player=terminal.current_player,
            is_terminal=True,
            to_call=0,
            last_aggressor=terminal.last_aggressor,
            blind_to_call=terminal.blind_to_call,
        )

    @pytest.mark.timeout(30)
    def test_one_missing_card_is_enumerated_exactly(self, engine):
        """With one card missing the value is the exact mean over all river cards."""
        rng = np.random.default_rng(5)
        terminal = _bet_call_to_river(engine, _deal_initial_state(engine, 2000, 0, rng))
        allin_state = self._strip_river(terminal)

        # Narrow belief on three disjoint combos: at most one drops per river card,
        # so the reference never hits the empty-range fallback.
        combo_indices = []
        known = engine._known_mask(allin_state, opp=1)
        used = known
        for idx in range(len(ALL_COMBOS)):
            if not (COMBO_MASKS[idx] & used):
                combo_indices.append(idx)
                used |= int(COMBO_MASKS[idx])
            if len(combo_indices) == 3:
                break
        belief = np.zeros(len(ALL_COMBOS))
        belief[combo_indices] = 1.0 / 3.0

        # Independent reference: enumerate rivers, weight surviving combos.
        values = []
        for card in Card.get_full_deck():
            if card.mask & known:
                continue
            river_state = engine._with_runout(allin_state, (card,))
            acc, total = 0.0, 0.0
            for idx in combo_indices:
                if COMBO_MASKS[idx] & card.mask:
                    continue
                payoff = replace_actor_hole_cards(
                    river_state, actor=1, combo=ALL_COMBOS[idx]
                ).get_payoff(0, engine.rules)
                acc += belief[idx] * float(payoff)
                total += belief[idx]
            values.append(acc / total)
        expected = float(np.mean(values))

        got = engine._terminal_value(allin_state, 0, 1, belief)
        assert got == pytest.approx(expected)

    def test_preflop_allin_beats_dominated_point_mass(self):
        """AA all-in preflop vs a 72o point mass must show a clearly positive value."""
        engine = _engine(_build_solver(0), allin_runouts=50)
        hole_cards = (
            (Card.new("As"), Card.new("Ah")),
            (Card.new("Kd"), Card.new("Kc")),  # dealt hand is a fiction; belief overrides it
        )
        allin_state = GameState(
            street=Street.PREFLOP,
            pot=800,
            stacks=(0, 0),
            board=(),
            hole_cards=hole_cards,
            betting_history=(Action(ActionType.ALL_IN, 400), call()),
            button_position=0,
            current_player=1,
            is_terminal=True,
            to_call=0,
            last_aggressor=0,
            blind_to_call=100,
        )
        belief = np.zeros(len(ALL_COMBOS))
        belief[_combo_index("7c", "2d")] = 1.0

        value = engine._terminal_value(allin_state, 0, 1, belief)
        # AA vs 72o has ~88% equity: EV ~ 0.88*800 - 400 ~ +300 chips. Runout noise
        # is averaged over 50 boards, so the estimate is safely above +100.
        assert value > 100.0

        # Deterministic under the engine RNG seed (fresh engine, same seed).
        engine2 = _engine(_build_solver(0), allin_runouts=50)
        assert engine2._terminal_value(allin_state, 0, 1, belief) == value


class TestTerminalBranching:
    """Opponent decisions whose every action ends the hand are integrated out."""

    @pytest.fixture(scope="class")
    def engine(self):
        return _engine(_build_solver(3), allin_runouts=5)

    def _jam_faced_node(self, engine):
        """A preflop node where the opponent faces an all-in (all responses terminal)."""
        state = _deal_initial_state(engine, 400, 0, np.random.default_rng(2))
        lbr = state.current_player
        legal = engine.rules.get_legal_actions(state, action_model=engine.action_model)
        jam = next(a for a in legal if a.type == ActionType.ALL_IN)
        return state, jam, state.apply_action(jam, engine.rules), lbr

    def test_jam_value_is_probability_weighted_branch_mixture(self, engine):
        state, jam, faced, lbr = self._jam_faced_node(engine)
        opp = 1 - lbr
        belief = engine._initial_belief(state, opp)
        legal, vecs = engine._opp_action_matrix(faced, opp, state, jam)

        engine.rng = np.random.default_rng(0)
        outcome = engine._branch_terminal_decision(faced, lbr, opp, legal, vecs, belief)
        assert outcome is not None
        assert outcome.terminal == "allin"  # the call branch dominates fold in severity

        # Reference mixture with an identically-seeded RNG (runout sampling must align).
        engine.rng = np.random.default_rng(0)
        probs = np.array([float(np.dot(belief, vecs[action])) for action in legal])
        expected = 0.0
        for weight, action in zip(probs / probs.sum(), legal):
            if weight <= 0.0:
                continue
            posterior = belief * vecs[action]
            branch_belief = posterior / posterior.sum()
            next_state = faced.apply_action(action, engine.rules)
            expected += weight * engine._terminal_value(next_state, lbr, opp, branch_belief)
        assert outcome.value == pytest.approx(expected)

    def test_no_branching_when_an_action_continues_the_hand(self, engine):
        # At the first preflop decision a call continues the hand, so the node
        # must fall through to the sampling path.
        state = _deal_initial_state(engine, 400, 0, np.random.default_rng(4))
        player = state.current_player
        legal, vecs = engine._opp_action_matrix(state, player, None, None)
        belief = engine._initial_belief(state, 1 - player)
        assert (
            engine._branch_terminal_decision(state, 1 - player, player, legal, vecs, belief) is None
        )


class TestPerHandRecords:
    """Per-hand outcomes + base seed enable paired CRN comparisons offline."""

    def test_records_match_aggregate_and_seed(self):
        solver = _build_solver(3, starting_stack=400)
        result = compute_lbr_exploitability(
            solver, LBRConfig(num_hands=6, equity_runouts=2, seed=21)
        )
        assert result.base_seed == 21
        assert len(result.hand_outcomes) == 6
        for outcome_p0, outcome_p1 in result.hand_outcomes:
            assert outcome_p0.terminal in {"fold", "showdown", "allin"}
            assert outcome_p1.terminal in {"fold", "showdown", "allin"}
            assert outcome_p0.pot > 0 and outcome_p1.pot > 0
        # The aggregate is exactly the mean of the recorded per-hand samples.
        samples = [(o0.value + o1.value) / 2.0 for o0, o1 in result.hand_outcomes]
        big_blind = solver.config.game.big_blind
        assert result.exploitability_mbb == pytest.approx(
            float(np.mean(samples)) / big_blind * 1000.0
        )

    def test_same_seed_yields_identical_records_and_null_paired_diff(self):
        """CRN foundation: same seed => hand-for-hand identical deals/outcomes."""
        solver = _build_solver(3, starting_stack=400)
        cfg = LBRConfig(num_hands=6, equity_runouts=2, seed=99)
        first = compute_lbr_exploitability(solver, cfg)
        second = compute_lbr_exploitability(solver, cfg)
        assert first.hand_outcomes == second.hand_outcomes

        samples_a = [(o0.value + o1.value) / 2.0 for o0, o1 in first.hand_outcomes]
        samples_b = [(o0.value + o1.value) / 2.0 for o0, o1 in second.hand_outcomes]
        comparison = compare_paired_samples(samples_a, samples_b)
        assert comparison["mean_diff"] == 0.0
        assert comparison["se_diff"] == 0.0
        assert comparison["p_value"] == 1.0
        assert not comparison["is_significant"]


class TestActionMapping:
    """`_map_to_real` projects a proxy response onto the real legal menu."""

    def test_fold_and_call_carry_across(self):
        engine = _engine(_build_solver(0))
        # For actions already in the legal menu the state is unused; any valid
        # state satisfies the signature.
        state = _deal_initial_state(engine, 400, 0, np.random.default_rng(0))
        legal = [fold(), call(), Action(ActionType.RAISE, 300)]
        assert engine._map_to_real(state, fold(), legal) == [(fold(), 1.0)]
        assert engine._map_to_real(state, call(), legal) == [(call(), 1.0)]

    def test_check_maps_to_itself_when_legal(self):
        engine = _engine(_build_solver(0))
        state = _deal_initial_state(engine, 400, 0, np.random.default_rng(0))
        legal = [check(), Action(ActionType.BET, 200)]
        assert engine._map_to_real(state, check(), legal) == [(check(), 1.0)]


class TestExploitability:
    """End-to-end: the leak-free bound is a positive, reproducible number."""

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_weak_blueprint_is_exploitable(self):
        solver = _build_solver(5, starting_stack=600)
        result = compute_lbr_exploitability(
            solver, LBRConfig(num_hands=60, equity_runouts=2, seed=13)
        )
        # A barely trained blueprint is grossly exploitable; the bound is well
        # clear of zero even at this sample size.
        assert result.exploitability_mbb > 0.0
        assert result.num_hands == 60

    def test_deterministic_under_fixed_seed(self):
        solver = _build_solver(3, starting_stack=400)
        cfg = LBRConfig(num_hands=8, equity_runouts=2, seed=99)
        first = compute_lbr_exploitability(solver, cfg)
        second = compute_lbr_exploitability(solver, cfg)
        assert first.exploitability_mbb == second.exploitability_mbb
        assert first.lbr_utility_p0 == second.lbr_utility_p0

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_parallel_matches_serial_bitwise(self):
        """num_workers must not change the result: per-hand seeding + ordered
        aggregation make parallel bitwise-identical to serial. Failure here means a
        global RNG isn't reseeded per-hand or aggregation order leaked."""
        solver = _build_solver(4, starting_stack=400, session_id="lbr-par-serial")
        serial = compute_lbr_exploitability(
            solver, LBRConfig(num_hands=16, equity_runouts=2, seed=123, num_workers=1)
        )
        parallel = compute_lbr_exploitability(
            solver,
            LBRConfig(num_hands=16, equity_runouts=2, seed=123, num_workers=4),
            blueprint_factory=_rebuild_parallel_test_blueprint,
        )
        assert serial.exploitability_mbb == parallel.exploitability_mbb
        assert serial.lbr_utility_p0 == parallel.lbr_utility_p0
        assert serial.lbr_utility_p1 == parallel.lbr_utility_p1
        assert serial.std_error_mbb == parallel.std_error_mbb
        assert serial.num_hands == parallel.num_hands == 16
