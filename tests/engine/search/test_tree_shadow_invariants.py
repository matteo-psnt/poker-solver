"""Randomized soak over the shadow: four invariants, checked after every step.

Two defects reached the pool because every hand-written shadow test observes an
action and looks the state up again WITHIN one street, and none of them asks
about a hypothetical combo. Both cost an 80-minute evaluation:

* the board went stale at lookup time (the resolver acts FIRST on a new street),
  and the blueprint was handed a FLOP with no cards on it --
  ``KeyError: Board () (canonical id 0) not found for FLOP``;
* the lookup returned the shadow verbatim, so all 1,326 per-combo rows were
  answered for the one hand actually held.

Removing either fix trips this file within a handful of hands.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.actions import Action, ActionType, call
from src.engine.search.range_inference import replace_actor_hole_cards
from src.engine.search.resolver import ALL_COMBOS, HUResolver
from tests.test_helpers import build_trained_test_solver

HANDS = 200
MAX_STEPS = 60


@pytest.fixture(scope="module")
def solver():
    return build_trained_test_solver(200)


def _resolver(solver) -> HUResolver:
    return HUResolver(
        blueprint=solver,
        action_model=solver.action_model,
        rules=solver.rules,
        config=solver.config.resolver.model_copy(
            update={"max_iterations": 4, "root_prior_weight": 50.0}
        ),
        rng=np.random.default_rng(3),
    )


def _off_menu(rng, legal: list[Action]) -> Action | None:
    """A size the abstraction does not offer, or None to stay on the menu."""
    sized = [a for a in legal if a.type in (ActionType.BET, ActionType.RAISE)]
    if not sized or rng.random() < 0.5:
        return None
    base = sized[rng.integers(len(sized))]
    return Action(base.type, int(base.amount * float(rng.uniform(0.31, 0.79))) + 1)


def _check(solver, res, state, tag, seen) -> None:
    shadow = res._shadow
    if shadow.started and not shadow.broken and shadow.diverged:
        assert shadow._shadow.street is state.street, f"{tag}: street drift"

    lookup_state = shadow.state_for(state)
    assert lookup_state.board == state.board, f"{tag}: stale board"

    if shadow.broken or state.is_terminal or solver.is_chance_node(state):
        return
    actions = list(solver.rules.get_legal_actions(state, solver.action_model))
    if not actions:
        return

    actor = state.current_player
    lookup = res._lookup_state(state, actions)
    res._policy_source.bucket_for(lookup, actor)  # must not raise

    combo = ALL_COMBOS[seen % len(ALL_COMBOS)]
    if combo[0] in state.board or combo[1] in state.board:
        return
    hypo = replace_actor_hole_cards(lookup, actor=actor, combo=combo)
    out = res._lookup_state(hypo, list(solver.rules.get_legal_actions(hypo, solver.action_model)))
    if out is not hypo:
        assert tuple(out.hole_cards[actor]) == tuple(combo), f"{tag}: combo dropped"


class TestTheShadowHoldsItsInvariantsUnderRandomPlay:
    def test_soak(self, solver):
        diverged = usable = 0
        for hand in range(HANDS):
            rng = np.random.default_rng(hand)
            np.random.seed(hand)
            res = _resolver(solver)
            state = solver.deal_initial_state()
            res.start_hand(state)
            for step in range(MAX_STEPS):
                if state.is_terminal:
                    break
                if solver.is_chance_node(state):
                    state = solver.sample_chance_outcome(state)
                    _check(solver, res, state, f"h{hand}s{step}chance", hand + step)
                    continue
                legal = list(solver.rules.get_legal_actions(state, solver.action_model))
                if not legal:
                    break
                action = _off_menu(rng, legal) or legal[rng.integers(len(legal))]
                _check(solver, res, state, f"h{hand}s{step}pre", hand + step)
                res.observe_public(state, action)
                try:
                    state = state.apply_action(action, solver.rules)
                except (ValueError, AssertionError):
                    break
                _check(solver, res, state, f"h{hand}s{step}post", hand + step)
            if res._shadow.diverged:
                diverged += 1
                usable += not res._shadow.broken

        # The soak is worthless if nothing ever went off the menu.
        assert diverged > HANDS // 10, f"only {diverged} hands diverged"
        # Measured ~83%: the shadow is not merely safe, it usually SURVIVES,
        # which is what bounds how much sharpness the fix can win back.
        assert usable > diverged // 2, f"only {usable}/{diverged} stayed usable"


class TestTheRangeUpdateSurvivesAnOffMenuSize:
    """Off-tree the blueprint has no infoset at all, so every combo fell through
    to the same uniform likelihood and the posterior stopped discriminating for
    the rest of the hand. The shadow gives the lookup a history that exists.

    Asserted as "the infoset resolves", not "the posterior moves": on the toy
    abstraction every combo shares a bucket, so a posterior assertion passes for
    the wrong reason and would stay green if the fix were reverted.
    """

    def test_the_likelihood_has_an_infoset_to_read(self, solver):
        np.random.seed(11)
        state = solver.deal_initial_state().apply_action(call(), solver.rules)
        for _ in range(3):
            if solver.is_chance_node(state):
                state = solver.sample_chance_outcome(state)

        res = _resolver(solver)
        res.start_hand(state)
        legal = list(solver.rules.get_legal_actions(state, solver.action_model))
        sizes = sorted(a.amount for a in legal if a.type is ActionType.BET)
        off = Action(ActionType.BET, int(sizes[0] * 0.37) + 1)

        res.observe(state, off)
        state = state.apply_action(off, solver.rules)
        assert res._shadow.diverged
        assert not res._shadow.broken

        source = res._policy_source
        actor = state.current_player
        with pytest.raises(KeyError):
            source.infoset_at(state, source.bucket_for(state, actor))

        lookup = res.range_lookup_state(state)
        assert lookup is not state, "the shadow was not consulted"
        source.infoset_at(lookup, source.bucket_for(lookup, actor))
