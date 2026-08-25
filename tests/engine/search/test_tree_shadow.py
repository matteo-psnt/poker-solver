"""The resolver's on-tree shadow: blueprint lookups survive an off-menu size."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.actions import Action, ActionType
from src.core.game.state import Street
from src.engine.search.resolver import HUResolver
from src.engine.search.tree_shadow import TreeShadow
from tests.test_helpers import build_trained_test_solver


def _flop_node(solver):
    """A real flop node: pass passively until the board is dealt.

    Not "call, then sample chance": preflop no longer closes on the SB call --
    the BB has its option -- so that recipe silently returned a PREFLOP state
    and every test here quietly stopped testing the flop.
    """
    np.random.seed(11)
    state = solver.deal_initial_state()
    for _ in range(12):
        if solver.is_chance_node(state):
            state = solver.sample_chance_outcome(state)
            if state.street is Street.FLOP and state.board:
                return state
            continue
        legal = solver.rules.get_legal_actions(state, solver.action_model)
        passive = next(
            (a for a in legal if a.type in (ActionType.CHECK, ActionType.CALL)), legal[0]
        )
        state = state.apply_action(passive, solver.rules)
    raise AssertionError("never reached a dealt flop")


def _resolver(solver):
    return HUResolver(
        blueprint=solver,
        action_model=solver.action_model,
        rules=solver.rules,
        config=solver.config.resolver.model_copy(
            update={"max_iterations": 8, "root_prior_weight": 50.0}
        ),
        rng=np.random.default_rng(11),
    )


def _off_menu_bet(solver, state) -> Action:
    sizes = sorted(
        a.amount
        for a in solver.rules.get_legal_actions(state, solver.action_model)
        if a.type == ActionType.BET
    )
    assert sizes, "need a BET on the menu to go off it"
    return Action(ActionType.BET, int(sizes[0] * 0.37) + 1)


class TestOffMenuSizeDoesNotEraseTheBlueprint:
    """MEASURED failure this pins.

    Infoset keys embed the full-hand normalized betting sequence, so one
    off-menu size makes every later lookup on the hand miss;
    `blueprint_action_distribution` returns None and the caller yields a uniform
    row. At a flop node the blueprint played
    `[0.142 0.075 0.401 0.237 0.145]`; after an opponent bet of 25 against a
    menu of `[66, 132, 250]` the same lookup returned exactly
    `[0.25 0.25 0.25 0.25]`.

    Off-tree LBR scored blueprint+resolver at +2568 mbb/hand FOR THE EXPLOITER
    against -254 for the bare blueprint (paired diff +2822, t=10.5): the bare
    blueprint survives because its evaluator hands it a translated state, which
    the resolver never had.
    """

    @pytest.fixture(scope="class")
    def solver(self):
        return build_trained_test_solver(400)

    def test_the_row_stays_informative_after_an_off_menu_bet(self, solver):
        state = _flop_node(solver)
        off = _off_menu_bet(solver, state)
        after = state.apply_action(off, solver.rules)
        menu = list(solver.rules.get_legal_actions(after, solver.action_model))
        uniform = np.full(len(menu), 1.0 / len(menu))

        told = _resolver(solver)
        told.observe(state, off)
        assert not np.allclose(told._blueprint_strategy(after, menu, use_average=True), uniform)

    def test_a_resolver_never_told_still_degrades(self, solver):
        """The control: without the observation there is no shadow to help."""
        state = _flop_node(solver)
        off = _off_menu_bet(solver, state)
        after = state.apply_action(off, solver.rules)
        menu = list(solver.rules.get_legal_actions(after, solver.action_model))
        uniform = np.full(len(menu), 1.0 / len(menu))

        untold = _resolver(solver)
        assert np.allclose(untold._blueprint_strategy(after, menu, use_average=True), uniform), (
            "if this stops being uniform the failure no longer reproduces and the "
            "test above is passing for the wrong reason"
        )

    def test_an_on_menu_hand_never_diverges(self, solver):
        """No shadow, no cost: an on-menu hand must behave as it always did."""
        state = _flop_node(solver)
        on = next(
            a
            for a in solver.rules.get_legal_actions(state, solver.action_model)
            if a.type == ActionType.BET
        )
        res = _resolver(solver)
        res.observe(state, on)
        assert not res._shadow.diverged
        assert res._shadow.state_for(state) is state


class TestTheShadowMenuIsJudgedAgainstShadowChips:
    """MEASURED failure this pins.

    ``_blueprint_strategy`` took its menu, bucket and infoset from the SHADOW
    and then handed the REAL state to the validity filter. Once the shadow has
    diverged the two carry different pots and stacks, and the filter's all-in
    test is an exact ``amount == current_stack`` -- so shadow actions were
    dropped for being unaffordable in a game they were never priced in. Over
    120 randomized hands 40% of shadow-backed lookups had at least one action
    fail, and rows like ``{fold 0.02, jam 0.98}`` collapsed to a pure fold that
    was then blended into what the resolver actually played.

    Pinned as the CONTRACT -- menu, infoset and filter all come from one state
    -- rather than by reproducing a chip split: a single fixture hand does not
    reliably diverge the stacks, so a value assertion here passes either way.
    """

    @pytest.fixture(scope="class")
    def solver(self):
        return build_trained_test_solver(400)

    def _state_the_filter_saw(self, monkeypatch, res, state, menu, *, lookup=None):
        """The state ``_blueprint_strategy`` hands the validity filter.

        ``_lookup_state`` is pinned to ``lookup`` when given, because it builds
        a fresh equal-but-distinct state per call -- so identity is the only
        way to tell "the shadow's state" from "an identical-looking real one".
        """
        seen = {}
        import src.engine.search.resolver as resolver_module

        original = resolver_module.blueprint_action_distribution

        def capture(infoset, filter_state, rules, actions, *, use_average):
            seen["state"] = filter_state
            return original(infoset, filter_state, rules, actions, use_average=use_average)

        monkeypatch.setattr(resolver_module, "blueprint_action_distribution", capture)
        if lookup is not None:
            monkeypatch.setattr(res, "_lookup_state", lambda *_args, **_kw: lookup)
        res._blueprint_strategy(state, menu, use_average=True)
        return seen["state"]

    def test_a_shadow_backed_row_is_filtered_against_the_shadow(self, solver, monkeypatch):
        state = _flop_node(solver)
        off = _off_menu_bet(solver, state)
        after = state.apply_action(off, solver.rules)
        menu = list(solver.rules.get_legal_actions(after, solver.action_model))

        res = _resolver(solver)
        res.observe(state, off)
        lookup = res._lookup_state(after, menu)
        assert lookup is not after, "the shadow must be in use or this pins nothing"

        saw = self._state_the_filter_saw(monkeypatch, res, after, menu, lookup=lookup)
        assert saw is lookup, (
            "the menu, bucket and infoset come from the shadow, so the validity "
            "filter must too -- against the real state it drops shadow actions "
            "for being unaffordable in a game they were never priced in"
        )

    def test_an_on_tree_row_is_unchanged(self, solver, monkeypatch):
        """On-tree ``_lookup_state`` returns the state itself, so the fix is a
        no-op there and existing on-tree numbers are not a new lineage."""
        state = _flop_node(solver)
        menu = list(solver.rules.get_legal_actions(state, solver.action_model))
        res = _resolver(solver)
        assert res._lookup_state(state, menu) is state
        saw = self._state_the_filter_saw(monkeypatch, res, state, menu)
        assert saw is state


class TestTreeShadowMechanics:
    @pytest.fixture(scope="class")
    def solver(self):
        return build_trained_test_solver(20)

    def test_it_aliases_the_real_state_until_something_goes_off_menu(self, solver):
        shadow = TreeShadow(solver.rules, solver.action_model)
        state = _flop_node(solver)
        shadow.start(state)
        assert shadow.state_for(state) is state
        assert not shadow.diverged

    def test_a_proxy_it_cannot_place_marks_broken_rather_than_guessing(self, solver):
        """Breaking restores the old behaviour; it must never invent a state."""
        shadow = TreeShadow(solver.rules, solver.action_model)
        state = _flop_node(solver)
        shadow.start(state)
        shadow._broken = True
        assert shadow.state_for(state) is state


class TestEveryDriverAdvancesTheShadow:
    """The shadow is useless on a path that never advances it.

    MEASURED: wiring it only into `HUResolver.observe` left it dead in LBR,
    because `ResolvedOpponent` tracks ranges itself and calls
    `solve_strategy_matrix` directly. The off-tree re-run came back +2567.9
    mbb/hand BIT-IDENTICAL to the pre-fix number -- the strategy had not moved.
    A per-driver test is the only thing that catches that; a resolver-level one
    passes while the measurement stays broken.
    """

    @pytest.fixture(scope="class")
    def solver(self):
        return build_trained_test_solver(400)

    def test_the_lbr_deployed_opponent_advances_it(self, solver):
        from src.pipeline.evaluation.estimators.lbr.opponent_model import ResolvedOpponent

        state = _flop_node(solver)
        off = _off_menu_bet(solver, state)
        model = ResolvedOpponent(
            solver,
            solver.config.resolver.model_copy(
                update={"max_iterations": 8, "root_prior_weight": 50.0}
            ),
            rng=np.random.default_rng(11),
        )
        model.reset(state, state.current_player)
        model.observe(state, off)
        assert model._resolver._shadow.diverged, (
            "ResolvedOpponent tracks ranges itself, so it must advance the resolver's "
            "shadow explicitly -- otherwise off-menu sizes silently erase the blueprint"
        )

    def test_the_agent_interface_advances_it(self, solver):
        """`resolver_match` drives through BlueprintAgent, a different path again."""
        from src.engine.search.agent import BlueprintAgent

        state = _flop_node(solver)
        off = _off_menu_bet(solver, state)
        agent = BlueprintAgent(
            solver,
            use_resolver=True,
            resolver_config=solver.config.resolver.model_copy(
                update={"max_iterations": 8, "root_prior_weight": 50.0}
            ),
            rng=np.random.default_rng(11),
        )
        agent.observe(state, off)
        assert agent.resolver is not None
        assert agent.resolver._shadow.diverged


class TestTheShadowIsHandScoped:
    """A reused resolver must not carry one hand's shadow into the next.

    MEASURED risk: `ResolvedOpponent` builds ONE `HUResolver` in `__init__` and
    keeps it for the whole evaluation, resetting only ranges per hand. The
    shadow starts lazily on the first observed action, so without an explicit
    per-hand restart it holds hand N-1's betting during hand N, breaks on the
    mismatch, and -- since `broken` is sticky -- stays broken for every
    remaining hand. Across 4,000 hands that is the fix doing nothing while every
    unit test still passes.
    """

    @pytest.fixture(scope="class")
    def solver(self):
        return build_trained_test_solver(400)

    def _model(self, solver):
        from src.pipeline.evaluation.estimators.lbr.opponent_model import ResolvedOpponent

        return ResolvedOpponent(
            solver,
            solver.config.resolver.model_copy(
                update={"max_iterations": 8, "root_prior_weight": 50.0}
            ),
            rng=np.random.default_rng(11),
        )

    def test_a_new_hand_starts_from_a_clean_shadow(self, solver):
        state = _flop_node(solver)
        model = self._model(solver)
        model.reset(state, state.current_player)
        model.observe(state, _off_menu_bet(solver, state))
        assert model._resolver._shadow.diverged

        fresh = solver.deal_initial_state()
        model.reset(fresh, fresh.current_player)
        assert not model._resolver._shadow.diverged, "hand N-1's divergence leaked into hand N"
        assert not model._resolver._shadow.broken
        assert model._resolver._shadow.state_for(fresh) is fresh

    def test_a_broken_shadow_does_not_persist_across_hands(self, solver):
        """`broken` is sticky WITHIN a hand by design; it must not outlive one."""
        state = _flop_node(solver)
        model = self._model(solver)
        model.reset(state, state.current_player)
        model._resolver._shadow._broken = True

        fresh = solver.deal_initial_state()
        model.reset(fresh, fresh.current_player)
        assert not model._resolver._shadow.broken
