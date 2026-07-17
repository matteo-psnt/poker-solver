"""Tests for runtime range inference helpers."""

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import call
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.engine.search.range_inference import combo_index_for, infer_ranges, update_ranges
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.storage.in_memory import InMemoryStorage
from tests.test_helpers import (
    DummyCardAbstraction,
    build_trained_test_solver,
    make_test_config,
    skew_preflop_infoset,
)


def _make_state_and_solver():
    rules = GameRules(small_blind=1, big_blind=2)
    hole_cards = (
        (Card.new("As"), Card.new("Kh")),
        (Card.new("Qd"), Card.new("Jc")),
    )
    state = rules.create_initial_state(starting_stack=200, hole_cards=hole_cards, button=0)
    config = make_test_config(seed=42)
    action_model = ActionModel(config)
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=InMemoryStorage(),
        config=config,
    )
    return state, solver


def test_infer_ranges_returns_normalized_distributions():
    state, solver = _make_state_and_solver()
    ranges = infer_ranges(state, solver)
    assert np.isclose(ranges.p0.sum(), 1.0)
    assert np.isclose(ranges.p1.sum(), 1.0)


def test_update_ranges_preserves_normalization():
    state, solver = _make_state_and_solver()
    ranges = infer_ranges(state, solver)
    updated = update_ranges(state, ranges, call(), solver)
    assert np.isclose(updated.p0.sum(), 1.0)
    assert np.isclose(updated.p1.sum(), 1.0)


def test_update_ranges_updates_only_the_actors_slot():
    """The Bayes update applies to whoever acted; the other slot is untouched.

    The actor's AA infoset is manufactured to put all blueprint mass on the
    observed raise (trained tiny blueprints are near-uniform, which gives the
    update nothing to grip), so the raise provably up-weights AA in the
    actor's slot — deterministically, with no training.
    """
    solver = build_trained_test_solver(0, session_id="range-slots")
    hole_cards = (
        (Card.new("As"), Card.new("Kh")),
        (Card.new("Qd"), Card.new("Jc")),
    )
    state = solver.rules.create_initial_state(starting_stack=400, hole_cards=hole_cards, button=0)
    actor = state.current_player
    legal = solver.rules.get_legal_actions(state, action_model=solver.action_model)
    raise_action = next(a for a in legal if a.is_aggressive())
    aa = (Card.new("Ad"), Card.new("Ac"))
    skew_preflop_infoset(solver, state, actor=actor, combo=aa, action=raise_action)

    ranges = infer_ranges(state, solver)
    updated = update_ranges(state, ranges, raise_action, solver)

    actor_before, other_before = (ranges.p0, ranges.p1) if actor == 0 else (ranges.p1, ranges.p0)
    actor_after, other_after = (updated.p0, updated.p1) if actor == 0 else (updated.p1, updated.p0)
    assert not np.allclose(actor_after, actor_before)
    assert np.allclose(other_after, other_before)
    # The raise up-weights the hand class the blueprint raises with certainty.
    assert actor_after[combo_index_for(aa)] > actor_before[combo_index_for(aa)]
