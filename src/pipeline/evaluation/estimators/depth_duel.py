"""Two blueprints, cut for any depths, ranked at ONE table depth on duplicate deals.

The depth question cannot be settled by scoring each rung natively: mbb/hand is
denominated in its own game, and a shallower game has less money in play. Each
deal is played twice, seats swapped, off the deck `blueprint_match` draws.

Both seats draw from ONE strategy stream, reseeded per game, so identical players
mirror exactly: a self-duel is all-zero pairs, and a pair carries signal only
where the two players chose differently -- the variance design `blueprint_match`
uses, carried over so the two estimators read alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.engine.solver.mccfr.chance import is_chance_node
from src.pipeline.evaluation.estimators.resolver_match import (
    complete_board,
    deal_for,
    deal_from_stack,
)
from src.pipeline.evaluation.statistics import summarize_samples
from src.pipeline.evaluation.units import pair_mean_mbb

if TYPE_CHECKING:
    from src.core.game.actions import Action
    from src.core.game.rules import GameRules
    from src.core.game.state import Card, GameState
    from src.pipeline.evaluation.estimators.cross_depth import CrossDepthPlayer

# A hand cannot outlast the tree; the bound turns a rules bug into an error
# rather than a hang inside a 20,000-deal task.
MAX_ACTIONS_PER_HAND = 512


class Seat(Protocol):
    """A player, or a ladder of them: anything that answers a table state."""

    def act(self, state: GameState) -> Action: ...


@dataclass(frozen=True)
class DepthDuelResult:
    """``a_mbb_per_hand`` is A's edge over B in the table's blinds."""

    a_mbb_per_hand: float
    se_mbb: float
    confidence_95_mbb: tuple[float, float]
    p_value: float
    num_deals: int
    num_hands: int
    pair_samples_mbb: list[float]


def play_hand(
    players: tuple[Seat, Seat],
    *,
    rules: GameRules,
    hole_cards: tuple[tuple[Card, Card], tuple[Card, Card]],
    board_stack: list[Card],
    button: int,
    starting_stack: int,
) -> float:
    """One hand at ``starting_stack`` a side. Returns seat 0's net in chips.

    There is no table action model: each player computes the legal set from its
    own, which is sound while the rungs share one whose sizes are RELATIVE (pot
    fractions), as every fielded rung does.
    """
    state = rules.create_initial_state(
        starting_stack=starting_stack, hole_cards=hole_cards, button=button
    )
    for _ in range(MAX_ACTIONS_PER_HAND):
        while not state.is_terminal and is_chance_node(state):
            state = deal_from_stack(state, board_stack)
        if state.is_terminal:
            break
        state = state.apply_action(players[state.current_player].act(state), rules)
    else:
        raise RuntimeError(f"a hand ran past {MAX_ACTIONS_PER_HAND} actions")
    if not state.ended_by_fold and len(state.board) < 5:
        # An all-in before the river leaves the runout to the caller.
        state = complete_board(state, board_stack)
    return float(state.get_payoff(0, rules))


def play_depth_duel(
    a: CrossDepthPlayer,
    b: CrossDepthPlayer,
    *,
    rules: GameRules,
    starting_stack: int,
    num_deals: int,
    seed: int = 1,
) -> DepthDuelResult:
    """A's chip edge over B at ``starting_stack``; ``rules`` carry both players' blinds."""
    samples: list[float] = []
    for deal in range(num_deals):
        hole_cards, board_stack, button = deal_for(seed, deal)
        payoffs = []
        for a_seat in (0, 1):
            stream = np.random.default_rng(np.random.SeedSequence([seed, deal]))
            a.rng = b.rng = stream
            seats = (a, b) if a_seat == 0 else (b, a)
            seat0 = play_hand(
                seats,
                rules=rules,
                hole_cards=hole_cards,
                board_stack=board_stack,
                button=button,
                starting_stack=starting_stack,
            )
            payoffs.append(seat0 if a_seat == 0 else -seat0)
        samples.append(pair_mean_mbb(payoffs[0], payoffs[1], rules.big_blind))
    summary = summarize_samples(samples)
    return DepthDuelResult(
        a_mbb_per_hand=summary["mean"],
        se_mbb=summary["se"],
        confidence_95_mbb=(summary["ci_lower"], summary["ci_upper"]),
        p_value=summary["p_value"],
        num_deals=num_deals,
        num_hands=2 * num_deals,
        pair_samples_mbb=samples,
    )
