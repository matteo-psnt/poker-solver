"""Tests for HU runtime resolver."""

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.engine.search.resolver import HUResolver
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.storage.in_memory import InMemoryStorage
from tests.test_helpers import DummyCardAbstraction, make_test_config


def _make_initial_state():
    rules = GameRules(small_blind=1, big_blind=2)
    hole_cards = (
        (Card.new("As"), Card.new("Kh")),
        (Card.new("Qd"), Card.new("Jc")),
    )
    state = rules.create_initial_state(starting_stack=200, hole_cards=hole_cards, button=0)
    return state, rules


def test_resolver_returns_legal_action():
    state, rules = _make_initial_state()
    config = make_test_config(seed=42)
    action_model = ActionModel(config)
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=InMemoryStorage(),
        config=config,
    )
    resolver = HUResolver(
        blueprint=solver,
        action_model=action_model,
        rules=rules,
        config=config.resolver,
    )

    action = resolver.act(state, time_budget_ms=50)
    assert action in action_model.get_legal_actions(state)


def test_solver_act_with_resolver_enabled():
    state, _ = _make_initial_state()
    config = make_test_config(seed=42)
    action_model = ActionModel(config)
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=InMemoryStorage(),
        config=config,
    )

    action = solver.act(state, use_resolver=True, time_budget_ms=50)
    assert action in action_model.get_legal_actions(state)


def test_resolver_uses_depth_cutoff_leaves(monkeypatch):
    state, rules = _make_initial_state()
    config = make_test_config(seed=42, **{"resolver.max_depth": 2})
    action_model = ActionModel(config)
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=InMemoryStorage(),
        config=config,
    )
    resolver = HUResolver(
        blueprint=solver,
        action_model=action_model,
        rules=rules,
        config=config.resolver,
    )

    observed = {"leaf_count": 0}

    def _fake_leaf_values(leaves, **_kwargs):
        observed["leaf_count"] = len(leaves)
        return {i: 0.0 for i in range(len(leaves))}

    monkeypatch.setattr("src.engine.search.resolver.estimate_leaf_values", _fake_leaf_values)

    result = resolver.solve(state, time_budget_ms=25)
    assert observed["leaf_count"] >= len(result.root_actions)
    assert observed["leaf_count"] > 0


@pytest.mark.parametrize("field", ["resolver.leaf_value_mode", "resolver.range_update_mode"])
def test_resolver_unknown_field_rejected(field):
    # Removed fields — ResolverConfig uses extra="forbid".
    with pytest.raises(ValidationError):
        make_test_config(seed=42, **{field: "any_value"})


def test_resolver_blend_alpha_zero_returns_blueprint_mix():
    state, rules = _make_initial_state()
    config = make_test_config(seed=42, **{"resolver.policy_blend_alpha": 0.0})
    action_model = ActionModel(config)
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=InMemoryStorage(),
        config=config,
    )
    resolver = HUResolver(
        blueprint=solver,
        action_model=action_model,
        rules=rules,
        config=config.resolver,
    )

    result = resolver.solve(state, time_budget_ms=25)
    assert np.allclose(result.strategy, result.blueprint_strategy)
