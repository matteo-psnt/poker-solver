"""Tests for runtime range inference helpers."""

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import call
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.engine.search.range_inference import infer_ranges, update_ranges
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.storage.in_memory import InMemoryStorage
from tests.test_helpers import DummyCardAbstraction, make_test_config


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
