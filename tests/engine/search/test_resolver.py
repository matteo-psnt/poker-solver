"""Tests for HU runtime resolver."""

import random as py_random

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card
from src.engine.search import resolver as resolver_module
from src.engine.search.agent import BlueprintAgent
from src.engine.search.range_inference import (
    combo_index_for,
    infer_ranges,
    replace_actor_hole_cards,
)
from src.engine.search.resolver import HUResolver
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.storage.in_memory import InMemoryStorage
from tests.test_helpers import (
    DummyCardAbstraction,
    build_test_storage,
    make_test_config,
    skew_preflop_infoset,
)


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


def test_agent_act_with_resolver_enabled():
    state, rules = _make_initial_state()
    config = make_test_config(seed=42)
    action_model = ActionModel(config)
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=InMemoryStorage(),
        config=config,
    )

    agent = BlueprintAgent(solver, use_resolver=True)
    action = agent.act(state, time_budget_ms=50)
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
    storage = build_test_storage(
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


def _trained_solver(config, session_id: str):
    """Small trained solver so blueprint lookups have real bite."""
    action_model = ActionModel(config)
    storage = build_test_storage(
        num_workers=1, worker_id=0, session_id=session_id, is_coordinator=True
    )
    solver = MCCFRSolver(
        action_model=action_model,
        card_abstraction=DummyCardAbstraction(),
        storage=storage,
        config=config,
    )
    for _ in range(10):
        solver.train_iteration()
    return solver, action_model


def _fresh_matrix(solver, action_model, rules, config, state):
    """Strategy matrix from a fresh resolver under fixed seeds/iterations."""
    resolver = HUResolver(
        blueprint=solver, action_model=action_model, rules=rules, config=config.resolver
    )
    py_random.seed(123)
    np.random.seed(123)
    return resolver.solve_strategy_matrix(state)


@pytest.mark.timeout(60)
def test_strategy_matrix_rows_are_distributions_and_call_is_pure():
    state, rules = _make_initial_state()
    config = make_test_config(
        seed=42, **{"resolver.max_iterations": 10, "resolver.leaf_rollouts": 2}
    )
    solver, action_model = _trained_solver(config, "resolver-matrix-pure")

    resolver = HUResolver(
        blueprint=solver, action_model=action_model, rules=rules, config=config.resolver
    )
    py_random.seed(123)
    np.random.seed(123)
    actions, matrix = resolver.solve_strategy_matrix(state)

    assert matrix.shape == (1326, len(actions))
    assert np.all(matrix >= 0.0)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    # Pure: no range state was created or mutated by the call.
    assert resolver._ranges is None

    # Reproducible: same seeds + pinned iterations => identical output.
    py_random.seed(123)
    np.random.seed(123)
    actions_again, matrix_again = resolver.solve_strategy_matrix(state)
    assert actions_again == actions
    np.testing.assert_array_equal(matrix_again, matrix)


@pytest.mark.timeout(60)
def test_solve_does_not_mutate_ranges():
    """observe() is the single range-update path: solve() must not write _ranges
    (a driver observing the applied action would otherwise double-count it)."""
    state, rules = _make_initial_state()
    config = make_test_config(
        seed=42, **{"resolver.max_iterations": 8, "resolver.leaf_rollouts": 2}
    )
    solver, action_model = _trained_solver(config, "resolver-solve-pure")
    resolver = HUResolver(
        blueprint=solver, action_model=action_model, rules=rules, config=config.resolver
    )
    np.random.seed(7)
    resolver.solve(state, time_budget_ms=50)
    assert resolver._ranges is None


@pytest.mark.timeout(60)
def test_observe_replays_history_for_both_seats():
    """History-replay range inference: observed actions Bayes-update the acting
    player's slot — including the OPPONENT's actions, which previously never
    reached range inference (the uniform-opponent-range limitation)."""
    state, rules = _make_initial_state()
    config = make_test_config(
        seed=42, **{"resolver.max_iterations": 8, "resolver.leaf_rollouts": 2}
    )
    solver, action_model = _trained_solver(config, "resolver-observe")
    resolver = HUResolver(
        blueprint=solver, action_model=action_model, rules=rules, config=config.resolver
    )

    baseline = infer_ranges(state, solver)
    first_actor = state.current_player
    legal = rules.get_legal_actions(state, action_model=action_model)
    open_raise = next(a for a in legal if a.is_aggressive())
    # Manufactured certainty: the blueprint opens AA with the observed raise
    # (tiny trained blueprints are near-uniform — nothing for Bayes to grip).
    aa = (Card.new("Ad"), Card.new("Ac"))
    skew_preflop_infoset(solver, state, actor=first_actor, combo=aa, action=open_raise)

    resolver.observe(state, open_raise)
    after_first = resolver._ranges
    assert after_first is not None
    first_slot = after_first.p0 if first_actor == 0 else after_first.p1
    first_base = baseline.p0 if first_actor == 0 else baseline.p1
    assert not np.allclose(first_slot, first_base)
    assert first_slot[combo_index_for(aa)] > first_base[combo_index_for(aa)]

    # The responder's action must update the OTHER slot too.
    faced = state.apply_action(open_raise, rules)
    responder = faced.current_player
    faced_legal = rules.get_legal_actions(faced, action_model=action_model)
    response = next((a for a in faced_legal if a.is_aggressive()), faced_legal[0])
    kk = (Card.new("Kd"), Card.new("Kc"))
    skew_preflop_infoset(solver, faced, actor=responder, combo=kk, action=response)
    resolver.observe(faced, response)
    after_second = resolver._ranges
    second_slot = after_second.p0 if responder == 0 else after_second.p1
    second_base = baseline.p0 if responder == 0 else baseline.p1
    assert not np.allclose(second_slot, second_base)
    assert second_slot[combo_index_for(kk)] > second_base[combo_index_for(kk)]


@pytest.mark.timeout(60)
def test_strategy_matrix_is_invariant_to_all_dealt_cards():
    """The matrix answers "what would the system do holding each combo" — it must
    not depend on what EITHER player was actually dealt (solve() only guards the
    opponent's cards; the per-combo matrix must also be free of the hero's)."""
    state, rules = _make_initial_state()
    config = make_test_config(
        seed=42, **{"resolver.max_iterations": 10, "resolver.leaf_rollouts": 2}
    )
    solver, action_model = _trained_solver(config, "resolver-matrix-honest")

    state_alt = replace_actor_hole_cards(
        state, actor=state.current_player, combo=(Card.new("9s"), Card.new("3h"))
    )
    state_alt = replace_actor_hole_cards(
        state_alt, actor=1 - state.current_player, combo=(Card.new("2c"), Card.new("7d"))
    )
    assert state_alt.hole_cards != state.hole_cards

    actions, matrix = _fresh_matrix(solver, action_model, rules, config, state)
    actions_alt, matrix_alt = _fresh_matrix(solver, action_model, rules, config, state_alt)

    assert actions == actions_alt
    np.testing.assert_allclose(matrix, matrix_alt)


@pytest.mark.timeout(60)
def test_strategy_matrix_row_matches_solve_strategy():
    """The deployed system's played strategy is exactly the matrix row of the
    actually-dealt combo — the consistency contract between measurement (matrix)
    and deployment (solve)."""
    state, rules = _make_initial_state()
    config = make_test_config(
        seed=42, **{"resolver.max_iterations": 10, "resolver.leaf_rollouts": 2}
    )
    solver, action_model = _trained_solver(config, "resolver-matrix-row")

    actions, matrix = _fresh_matrix(solver, action_model, rules, config, state)

    resolver = HUResolver(
        blueprint=solver, action_model=action_model, rules=rules, config=config.resolver
    )
    py_random.seed(123)
    np.random.seed(123)
    result = resolver.solve(state)

    assert result.root_actions == actions
    hero_combo_row = matrix[combo_index_for(state.hole_cards[state.current_player])]
    np.testing.assert_allclose(hero_combo_row, result.strategy)


@pytest.mark.timeout(60)
def test_strategy_matrix_alpha_zero_equals_blueprint_rows():
    """alpha=0 (and no probability floor) collapses the matrix to the pure
    per-combo blueprint strategy — the plumbing-only regression anchor."""
    state, rules = _make_initial_state()
    config = make_test_config(
        seed=42,
        **{
            "resolver.max_iterations": 2,
            "resolver.leaf_rollouts": 2,
            "resolver.policy_blend_alpha": 0.0,
            "resolver.min_strategy_prob": 0.0,
        },
    )
    solver, action_model = _trained_solver(config, "resolver-matrix-alpha0")

    resolver = HUResolver(
        blueprint=solver, action_model=action_model, rules=rules, config=config.resolver
    )
    py_random.seed(123)
    np.random.seed(123)
    actions, matrix = resolver.solve_strategy_matrix(state)

    for combo in [
        (Card.new("As"), Card.new("Ah")),
        (Card.new("7c"), Card.new("2d")),
        (Card.new("Ts"), Card.new("9s")),
    ]:
        hypo = replace_actor_hole_cards(state, actor=state.current_player, combo=combo)
        expected = resolver._blueprint_strategy(hypo, actions, use_average=True)
        np.testing.assert_allclose(matrix[combo_index_for(combo)], expected)


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
