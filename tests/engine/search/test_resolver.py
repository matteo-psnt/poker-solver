"""Tests for HU runtime resolver."""

import random as py_random

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.engine.search import resolver as resolver_module
from src.engine.search.range_inference import replace_actor_hole_cards
from src.engine.search.resolver import HUResolver
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.storage.in_memory import InMemoryStorage
from src.engine.solver.storage.shared_array import SharedArrayStorage
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
    assert action in rules.get_legal_actions(state, action_model=action_model)


def test_solver_act_with_resolver_enabled():
    state, rules = _make_initial_state()
    config = make_test_config(seed=42)
    action_model = ActionModel(config)
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=InMemoryStorage(),
        config=config,
    )

    action = solver.act(state, use_resolver=True, time_budget_ms=50)
    assert action in rules.get_legal_actions(state, action_model=action_model)


def test_resolver_solves_subgame_with_per_combo_strategy(monkeypatch):
    """solve() routes through the range-vs-range subgame CFR and picks the
    hero-combo row of the average root strategy."""
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

    observed = {}
    real_solve = resolver_module.solve_subgame

    def _spy(tree, **kwargs):
        solution = real_solve(tree, **kwargs)
        observed["solution"] = solution
        observed["hero"] = kwargs["hero"]
        return solution

    monkeypatch.setattr(resolver_module, "solve_subgame", _spy)

    result = resolver.solve(state, time_budget_ms=25)
    solution = observed["solution"]
    assert solution.iterations >= 8
    assert solution.root_strategy.shape[1] == len(result.root_actions)
    # The played resolver strategy is the hero-combo row of the average strategy.
    assert result.strategy.shape == (len(result.root_actions),)
    assert result.action_values.shape == (len(result.root_actions),)


@pytest.mark.timeout(60)
def test_resolver_is_not_clairvoyant():
    """solve() must be invariant to the opponent's dealt hole cards.

    The resolver may only condition on public state + hero's own cards + the
    tracked range. Same hero hand, same board, same RNG seed, two different
    dealt opponent hands => identical action values. Fails if the solve
    conditions on anything but public state + ranges.
    """
    state, rules = _make_initial_state()
    # Fixed iteration count: budget-driven iterations vary with wall clock and
    # would break bitwise comparison.
    config = make_test_config(seed=42, **{"resolver.max_iterations": 20})
    action_model = ActionModel(config)
    storage = SharedArrayStorage(
        num_workers=1, worker_id=0, session_id="resolver-clair", is_coordinator=True
    )
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=storage,
        config=config,
    )
    for _ in range(10):  # trained strategies give the internal-node path real bite
        solver.train_iteration()

    hero = state.current_player
    opponent = 1 - hero
    state_alt = replace_actor_hole_cards(
        state, actor=opponent, combo=(Card.new("2c"), Card.new("7d"))
    )
    assert state.hole_cards[hero] == state_alt.hole_cards[hero]
    assert state.hole_cards[opponent] != state_alt.hole_cards[opponent]

    def _solve(target_state):
        resolver = HUResolver(
            blueprint=solver, action_model=action_model, rules=rules, config=config.resolver
        )
        # Seed BOTH streams: leaf sampling uses np.random, the blueprint's board
        # dealing inside rollouts uses the global `random` module.
        py_random.seed(123)
        np.random.seed(123)
        return resolver.solve(target_state, time_budget_ms=25)

    result = _solve(state)
    result_alt = _solve(state_alt)

    # Action values are the clairvoyance carrier: rollout payoffs and opponent
    # response predictions both feed them. (The final root strategy also depends
    # on wall-clock iteration counts, so it is not asserted bitwise.)
    np.testing.assert_allclose(result.action_values, result_alt.action_values)
    np.testing.assert_allclose(result.blueprint_strategy, result_alt.blueprint_strategy)


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
