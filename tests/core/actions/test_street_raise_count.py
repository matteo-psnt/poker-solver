"""`ActionModel` counts raises by scanning BACKWARD. Does that ever disagree?

`GameRules._get_actions_on_current_street` replaced its own backward heuristic
with a forward replay, and says why: a backward scan "mismatched the
boundary-straddling pair when two consecutive streets both ended check-check".
`ActionModel._count_raises_on_current_street` still scans backward, and it picks
the legal bet/raise sizes -- so a miscount there changes the action abstraction,
and with it the betting tree.

Measured: it does NOT disagree, because the two uses differ. `rules` asks "does
the action I am applying RIGHT NOW close the street?", where the last action's
status is undecided and a straddling pair can be mismatched. `ActionModel` asks
about a settled state between actions, where the nearest closer going backward
is unambiguously this street's start.

This pins that. It is a differential test, not an example test: examples would
not have found the boundary case, and the coverage assertion below is what makes
a green result mean something.
"""

from __future__ import annotations

import random

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.actions import ActionType
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState
from tests.test_helpers import make_test_config

AGGRESSIVE = (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
CLOSERS_FOR_COVERAGE = 2


def _forward_replay(history) -> list:
    """Actions on the current street, by replaying the WHOLE history forward.

    Deliberately not ``rules._get_actions_on_current_street``: that exempts the
    final action from being a closer because it is called mid-``apply_action``.
    Here every action is settled, so every closer counts.
    """
    on_street: list = []
    for action in history:
        if action.type == ActionType.CALL or (
            action.type == ActionType.CHECK and on_street and on_street[-1].type == ActionType.CHECK
        ):
            on_street = []
        else:
            on_street.append(action)
    return on_street


def _closes_seen(history) -> int:
    """How many check-check street closes are behind this state."""
    closes, run = 0, []
    for action in history:
        if action.type == ActionType.CALL:
            run = []
        elif action.type == ActionType.CHECK and run and run[-1].type == ActionType.CHECK:
            closes += 1
            run = []
        else:
            run.append(action)
    return closes


def _deal_to_street(state: GameState, pool) -> GameState:
    need = state.street.board_card_count - len(state.board)
    if need <= 0:
        return state
    return state.replace(board=tuple(state.board) + tuple(pool[:need]))


@pytest.mark.timeout(60)
def test_backward_raise_scan_agrees_with_forward_replay():
    config = make_test_config(seed=1, small_blind=1, big_blind=2, starting_stack=200)
    rules, model = GameRules(1, 2), ActionModel(config)
    rng = random.Random(0)
    deck = Card.get_full_deck()

    compared = 0
    after_two_closes = 0
    for trial in range(4000):
        cards = rng.sample(deck, 9)
        state = rules.create_initial_state(
            starting_stack=200,
            hole_cards=((cards[0], cards[1]), (cards[2], cards[3])),
            button=trial % 2,
        )
        pool = cards[4:]
        while not state.is_terminal:
            state = _deal_to_street(state, pool)
            legal = rules.get_legal_actions(state, action_model=model)
            if not legal:
                break

            expected = sum(
                1 for a in _forward_replay(state.betting_history) if a.type in AGGRESSIVE
            )
            assert model._count_raises_on_current_street(state) == expected, (
                f"street={state.street.name} "
                f"history={[f'{a.type.name}{a.amount or 0}' for a in state.betting_history]}"
            )
            compared += 1
            if _closes_seen(state.betting_history) >= CLOSERS_FOR_COVERAGE:
                after_two_closes += 1

            # Biased HARD toward passive play: a uniform pick over many bet sizes
            # against one check almost never produces consecutive check-check
            # streets, and those are the whole point of this test.
            passive = [a for a in legal if a.type in (ActionType.CHECK, ActionType.CALL)]
            state = rules.apply_action(
                state,
                rng.choice(passive) if passive and rng.random() < 0.85 else rng.choice(legal),
            )

    assert compared > 10_000, compared
    # Without this the test could pass by never generating the case at issue.
    # An unbiased walk produced 5 of these in 12,007 points; the bias gives ~4,800.
    assert after_two_closes > 1_000, (
        f"only {after_two_closes} states followed >= {CLOSERS_FOR_COVERAGE} check-check "
        "closes; the boundary case is not being exercised"
    )
