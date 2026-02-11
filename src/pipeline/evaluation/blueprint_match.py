"""Blueprint-vs-blueprint duplicate-deal match.

Plays two trained blueprints against each other on seat-swapped pairs off a
fixed deck (same variance design as :mod:`resolver_match`): every deal is
played twice with identical cards and the blueprints on opposite seats, and the
per-deal sample is blueprint A's net over the pair. Card luck cancels, leaving
the head-to-head skill difference measurable in ~2k deals.

This answers a different question than LBR: not "how far from equilibrium is
each blueprint" but "who wins chips off whom when they actually play". Two
blueprints can be LBR-comparable yet lopsided head-to-head (or vice versa);
use both signals.

Each blueprint samples from its own average strategy with its own card
abstraction — cross-abstraction matches are legitimate as long as the game
configs (blinds, stacks) agree, which is validated by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Card
from src.engine.solver.protocols import Blueprint
from src.pipeline.evaluation.resolver_match import _complete_board, _deal_from_stack
from src.pipeline.evaluation.statistics import summarize_samples
from src.shared.units import pair_mean_mbb


@dataclass(frozen=True)
class BlueprintMatchResult:
    """Outcome of a duplicate-deal blueprint-vs-blueprint match."""

    a_mbb_per_hand: float
    se_mbb: float
    confidence_95_mbb: tuple[float, float]
    p_value: float
    num_deals: int
    num_hands: int
    pair_samples_mbb: list[float]


def play_blueprint_match(
    solver_a: Blueprint,
    solver_b: Blueprint,
    *,
    num_deals: int = 2000,
    seed: int = 1,
) -> BlueprintMatchResult:
    """Play duplicate deals of A vs B and report A's chip edge.

    Positive ``a_mbb_per_hand`` means blueprint A wins chips off blueprint B.
    Strategy sampling reseeds the global NumPy RNG (which the blueprints draw
    from) identically for both games of a pair, so a match is reproducible for
    a fixed seed and a pair only diverges where the blueprints disagree.
    """
    rules = solver_a.rules
    big_blind = solver_a.config.game.big_blind
    starting_stack = solver_a.config.game.starting_stack

    pair_samples_mbb: list[float] = []

    for deal in range(num_deals):
        rng = np.random.default_rng(np.random.SeedSequence([seed, deal]))
        order = [int(i) for i in rng.permutation(52)]
        hole_cards = (
            (FULL_DECK[order[0]], FULL_DECK[order[1]]),
            (FULL_DECK[order[2]], FULL_DECK[order[3]]),
        )
        board_stack = [FULL_DECK[i] for i in order[4:9]]
        button = deal % 2

        # Both seat-swapped games replay the SAME sampling stream: while the two
        # blueprints agree, the games mirror each other exactly, so a self-match
        # is provably all-zero pairs and an A-vs-B pair only diverges where the
        # blueprints actually disagree — duplicate-style variance reduction.
        game_seed = int(np.random.SeedSequence([seed, deal]).generate_state(1)[0])
        seat_payoffs: list[float] = []
        for a_seat in (0, 1):
            np.random.seed(game_seed)
            seat_payoffs.append(
                _play_game(
                    solver_a,
                    solver_b,
                    rules,
                    hole_cards=hole_cards,
                    board_stack=board_stack,
                    button=button,
                    starting_stack=starting_stack,
                    a_seat=a_seat,
                )
            )

        payoff_seat0, payoff_seat1 = seat_payoffs
        pair_samples_mbb.append(pair_mean_mbb(payoff_seat0, payoff_seat1, big_blind))

    summary = summarize_samples(pair_samples_mbb)
    return BlueprintMatchResult(
        a_mbb_per_hand=summary["mean"],
        se_mbb=summary["se"],
        confidence_95_mbb=(summary["ci_lower"], summary["ci_upper"]),
        p_value=summary["p_value"],
        num_deals=num_deals,
        num_hands=2 * num_deals,
        pair_samples_mbb=pair_samples_mbb,
    )


def _play_game(
    solver_a: Blueprint,
    solver_b: Blueprint,
    rules: GameRules,
    *,
    hole_cards: tuple[tuple[Card, Card], tuple[Card, Card]],
    board_stack: list[Card],
    button: int,
    starting_stack: int,
    a_seat: int,
) -> float:
    """One game off a fixed deck; returns blueprint A's payoff."""
    state = rules.create_initial_state(
        starting_stack=starting_stack,
        hole_cards=hole_cards,
        button=button,
    )

    while not state.is_terminal:
        if solver_a.is_chance_node(state):
            state = _deal_from_stack(state, board_stack)
            continue
        actor = solver_a if state.current_player == a_seat else solver_b
        action = actor.sample_action_from_strategy(state, use_average=True)
        state = state.apply_action(action, rules)

    if not state.ended_by_fold and len(state.board) < 5:
        state = _complete_board(state, board_stack)

    return float(state.get_payoff(a_seat, rules))
