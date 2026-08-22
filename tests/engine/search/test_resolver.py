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
from src.engine.search.tree_builder import build_local_tree
from tests.test_helpers import (
    DummyCardAbstraction,
    build_test_solver,
    build_trained_test_solver,
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
    solver, _storage = build_test_solver(config, DummyCardAbstraction())
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
    solver, _storage = build_test_solver(config, DummyCardAbstraction())

    agent = BlueprintAgent(solver, use_resolver=True)
    action = agent.act(state, time_budget_ms=50)
    assert action in rules.get_legal_actions(state, action_model=action_model)


def test_agent_forwards_rng_to_resolver():
    """The injected generator must reach the resolver: it drives leaf-runout
    sampling, so eval harnesses pin it per hand for reproducibility."""
    config = make_test_config(seed=42)
    solver, _storage = build_test_solver(config, DummyCardAbstraction())

    rng = np.random.default_rng(7)
    agent = BlueprintAgent(solver, use_resolver=True, rng=rng)
    assert agent.resolver is not None
    assert agent.resolver.rng is rng


def test_resolver_solves_subgame_with_per_combo_strategy(monkeypatch):
    """solve() routes through the range-vs-range subgame CFR and picks the
    hero-combo row of the average root strategy."""
    state, rules = _make_initial_state()
    config = make_test_config(seed=42, **{"resolver.max_depth": 2})
    action_model = ActionModel(config)
    solver, _storage = build_test_solver(config, DummyCardAbstraction())
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
    solver, _storage = build_test_solver(config, DummyCardAbstraction())
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
            blueprint=solver,
            action_model=action_model,
            rules=rules,
            config=config.resolver,
            rng=np.random.default_rng(123),
        )
        # The blueprint's board dealing inside rollouts uses the global `random`
        # module; the resolver's own sampling comes from the rng passed above.
        py_random.seed(123)
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


def _trained_solver(config):
    """Small trained solver so blueprint lookups have real bite."""
    action_model = ActionModel(config)
    solver, _storage = build_test_solver(config, DummyCardAbstraction())
    for _ in range(10):
        solver.train_iteration()
    return solver, action_model


def _fresh_matrix(solver, action_model, rules, config, state):
    """Strategy matrix from a fresh resolver under fixed seeds/iterations."""
    resolver = HUResolver(
        blueprint=solver,
        action_model=action_model,
        rules=rules,
        config=config.resolver,
        rng=np.random.default_rng(123),
    )
    py_random.seed(123)
    return resolver.solve_strategy_matrix(state)


@pytest.mark.timeout(60)
def test_strategy_matrix_rows_are_distributions_and_call_is_pure():
    state, rules = _make_initial_state()
    config = make_test_config(
        seed=42, **{"resolver.max_iterations": 10, "resolver.leaf_rollouts": 2}
    )
    solver, action_model = _trained_solver(config)

    actions, matrix = _fresh_matrix(solver, action_model, rules, config, state)

    assert matrix.shape == (1326, len(actions))
    assert np.all(matrix >= 0.0)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    # Pure: no range state was created or mutated by the call.
    resolver = HUResolver(
        blueprint=solver,
        action_model=action_model,
        rules=rules,
        config=config.resolver,
        rng=np.random.default_rng(123),
    )
    py_random.seed(123)
    resolver.solve_strategy_matrix(state)
    assert resolver._ranges is None

    # Reproducible: same seeds + pinned iterations => identical output.
    actions_again, matrix_again = _fresh_matrix(solver, action_model, rules, config, state)
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
    solver, action_model = _trained_solver(config)
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
    solver, action_model = _trained_solver(config)
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
    assert after_second is not None
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
    solver, action_model = _trained_solver(config)

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
    solver, action_model = _trained_solver(config)

    actions, matrix = _fresh_matrix(solver, action_model, rules, config, state)

    resolver = HUResolver(
        blueprint=solver,
        action_model=action_model,
        rules=rules,
        config=config.resolver,
        rng=np.random.default_rng(123),
    )
    py_random.seed(123)
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
    solver, action_model = _trained_solver(config)

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
    solver, _storage = build_test_solver(config, DummyCardAbstraction())
    resolver = HUResolver(
        blueprint=solver,
        action_model=action_model,
        rules=rules,
        config=config.resolver,
    )

    result = resolver.solve(state, time_budget_ms=25)
    assert np.allclose(result.strategy, result.blueprint_strategy)


def test_the_leaf_continuation_knob_reaches_the_solve(monkeypatch):
    """`resolver.leaf_continuation_fraction` must actually arrive at the leaf.

    A knob that is declared, threaded and never read is the failure this repo
    has already paid for once (`b99a799` shipped a classification with no reader
    on either surface). Three hops separate the config field from
    `_leaf_values`, so this asserts the value that arrives, not merely that the
    resolver still runs.

    The DEFAULT is asserted too: zero is the shipped behaviour, and a
    continuation that silently applied itself would change every published
    resolver number without anything failing.
    """
    state, rules = _make_initial_state()
    seen: list[float] = []
    real_solve = resolver_module.solve_subgame

    def _spy(tree, **kwargs):
        seen.append(kwargs["continuation"].pot_fraction)
        return real_solve(tree, **kwargs)

    monkeypatch.setattr(resolver_module, "solve_subgame", _spy)

    for expected, overrides in ((0.0, {}), (0.5, {"resolver.leaf_continuation_fraction": 0.5})):
        config = make_test_config(seed=42, **{"resolver.max_depth": 2, **overrides})
        solver, _storage = build_test_solver(config, DummyCardAbstraction())
        HUResolver(
            blueprint=solver,
            action_model=ActionModel(config),
            rules=rules,
            config=config.resolver,
        ).solve(state, time_budget_ms=25)
        assert seen[-1] == expected


class TestStarvedSolveDegradesToTheBlueprint:
    """A truncated subgame solve must fall back on the blueprint, not on uniform.

    MEASURED failure this pins: `_prepare_nodes` starts regrets at ZERO, and
    regret matching from zero regrets IS the uniform strategy. At the shipped
    `time_budget_ms=300` the solve gets ~9 iterations (~32 ms each), which
    barely move it -- the resolver's root row had entropy 1.604 against a 1.609
    maximum, i.e. uniform random over {check, three bet sizes, all-in}. The
    duplicate-deal gate scored that at -486 mbb/hand (-48.6 BB/100) against the
    bare blueprint it wraps, over 20,000 deals, p~0.

    `root_prior_weight` seeds the root's `strategy_sum` with the blueprint as a
    pseudo-count worth that many iterations, so a starved solve returns the
    blueprint instead.
    """

    ITERATIONS = 8

    def _row(self, weight: float):
        """The raw (unblended) resolver root row, and the blueprint's, at ``weight``.

        A TRAINED blueprint, because an untrained one is itself uniform and the
        comparison below would be vacuous.
        """
        solver = build_trained_test_solver(
            300,
            resolver={"max_iterations": self.ITERATIONS, "root_prior_weight": weight},
        )
        state = solver.deal_initial_state()
        resolver = HUResolver(
            blueprint=solver,
            action_model=solver.action_model,
            rules=solver.rules,
            config=solver.config.resolver,
            rng=np.random.default_rng(7),
        )
        actions, solution = resolver._solve_root(
            state, infer_ranges(state, solver), solver.config.resolver.time_budget_ms
        )
        combo = combo_index_for(state.hole_cards[state.current_player])
        blueprint = resolver._blueprint_strategy(state, list(actions), use_average=True)
        return solution.root_strategy[combo], blueprint

    @staticmethod
    def _tv(a, b):
        return 0.5 * float(np.abs(a - b).sum())

    def test_an_unseeded_starved_solve_is_far_from_the_blueprint(self):
        """The defect, stated as what is ROBUST across states.

        WHERE it lands varies: on a flop state it was near-uniform (entropy
        1.604 against a 1.609 maximum, i.e. uniform over check/three bets/all-in),
        while this preflop root concentrates somewhere else entirely. What holds
        either way is that ~8 iterations from zero regrets do not reach the
        blueprint, so the resolver overrides a trained strategy with a
        half-solved one. Do not re-assert "near uniform" -- that was one state.
        """
        unseeded, blueprint = self._row(0.0)
        assert self._tv(unseeded, blueprint) > 0.2

    def test_a_prior_pulls_a_starved_solve_onto_the_blueprint(self):
        unseeded, blueprint = self._row(0.0)
        seeded, _ = self._row(50.0)
        assert self._tv(seeded, blueprint) < self._tv(unseeded, blueprint) / 2.0

    def test_the_prior_does_not_lock_cfr_out(self):
        """It is a pseudo-count, not an override: the solve still moves the row."""
        seeded, blueprint = self._row(50.0)
        assert self._tv(seeded, blueprint) > 0.0

    def test_zero_weight_is_exactly_todays_behaviour(self):
        first, _ = self._row(0.0)
        second, _ = self._row(0.0)
        assert np.array_equal(first, second)

    def test_a_misshaped_prior_is_refused(self):
        """A wrong-shaped prior is a caller bug; silently ignoring it hides the fix."""
        from src.engine.search.subgame_cfr import solve_subgame

        solver = build_trained_test_solver(20)
        state = solver.deal_initial_state()
        tree = build_local_tree(
            state, action_model=solver.action_model, rules=solver.rules, max_depth=2
        )
        ranges = infer_ranges(state, solver)
        with pytest.raises(ValueError, match="expected"):
            solve_subgame(
                tree,
                hero=state.current_player,
                hero_range=ranges.p0,
                opponent_range=ranges.p1,
                rules=solver.rules,
                budget_ms=5,
                max_iterations=2,
                root_prior=np.zeros((3, 3)),
                root_prior_weight=1.0,
            )
