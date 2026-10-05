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
from typing import TYPE_CHECKING

import numpy as np

from src.engine.search.agent import BlueprintAgent
from src.pipeline.evaluation.estimators.resolver_match import (
    MatchTelemetry,
    complete_board,
    deal_for,
    deal_from_stack,
)
from src.pipeline.evaluation.statistics import summarize_samples
from src.pipeline.evaluation.units import pair_mean_mbb

if TYPE_CHECKING:
    from src.core.game.rules import GameRules
    from src.core.game.state import Card
    from src.engine.solver.policy.source import ScorableBlueprint


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
    # Populated only when side A plays the DEPLOYED system. The resolver's cost is
    # a stack-off rate and its symptom is an undecided root row, and neither is
    # visible in a chip edge -- see `resolver_match` for the same instruments.
    telemetry: MatchTelemetry | None = None


def play_blueprint_match(
    solver_a: ScorableBlueprint,
    solver_b: ScorableBlueprint,
    *,
    num_deals: int = 2000,
    seed: int = 1,
    resolver_a: bool = False,
    resolver_b: bool = False,
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
    telemetry = MatchTelemetry()

    for deal in range(num_deals):
        hole_cards, board_stack, button = deal_for(seed, deal)

        # Both seat-swapped games replay the SAME sampling stream: while the two
        # blueprints agree, the games mirror each other exactly, so a self-match
        # is provably all-zero pairs and an A-vs-B pair only diverges where the
        # blueprints actually disagree — duplicate-style variance reduction.
        game_seed = int(np.random.SeedSequence([seed, deal]).generate_state(1)[0])
        seat_payoffs: list[float] = []
        for a_seat in (0, 1):
            np.random.seed(game_seed)
            payoff, game = _play_game(
                solver_a,
                solver_b,
                rules,
                hole_cards=hole_cards,
                board_stack=board_stack,
                button=button,
                starting_stack=starting_stack,
                a_seat=a_seat,
                resolver_a=resolver_a,
                resolver_b=resolver_b,
                # Pinned per (deal, seat) like resolver_match's, so a deployed
                # side A is reproducible rather than reading the ambient stream.
                resolver_rng=np.random.default_rng(np.random.SeedSequence([seed, deal, a_seat]))
                if resolver_a or resolver_b
                else None,
            )
            seat_payoffs.append(payoff)
            telemetry.absorb(game)

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
        telemetry=telemetry if resolver_a else None,
    )


def _play_game(
    solver_a: ScorableBlueprint,
    solver_b: ScorableBlueprint,
    rules: GameRules,
    *,
    hole_cards: tuple[tuple[Card, Card], tuple[Card, Card]],
    board_stack: list[Card],
    button: int,
    starting_stack: int,
    a_seat: int,
    resolver_a: bool = False,
    resolver_b: bool = False,
    resolver_rng: np.random.Generator | None = None,
) -> tuple[float, MatchTelemetry]:
    """One game off a fixed deck; returns blueprint A's payoff and its counts.

    ``resolver_a`` makes side A the DEPLOYED system -- blueprint routed through the
    runtime resolver -- rather than the bare table. That is the only way to measure
    the resolver against a genuinely off-menu opponent: `resolver_match` plays a
    blueprint against ITSELF, so every line it walks is one its own tree has a node
    for, and the resolver's claimed benefit is entirely off-tree.
    """
    telemetry = MatchTelemetry()
    # A fresh agent per game, matching its documented contract (one resolver per
    # agent lifetime, a new agent per hand).
    agent = BlueprintAgent(solver_a, use_resolver=True, rng=resolver_rng) if resolver_a else None
    # A deployed side B is what lets two DIFFERENT action trees meet at all: a bare
    # side raises KeyError on any size its own tree lacks.
    agent_b = BlueprintAgent(solver_b, use_resolver=True, rng=resolver_rng) if resolver_b else None
    state = rules.create_initial_state(
        starting_stack=starting_stack,
        hole_cards=hole_cards,
        button=button,
    )

    while not state.is_terminal:
        if solver_a.is_chance_node(state):
            state = deal_from_stack(state, board_stack)
            continue
        actor_seat = state.current_player
        is_a = actor_seat == a_seat
        if is_a and agent is not None:
            telemetry.decisions += 1
            action = agent.act(state)
        elif not is_a and agent_b is not None:
            telemetry.blueprint_decisions += 1
            action = agent_b.act(state)
        else:
            actor = solver_a if is_a else solver_b
            if is_a:
                telemetry.decisions += 1
            else:
                telemetry.blueprint_decisions += 1
            action = actor.sample_action_from_strategy(state, use_average=True)
        # Every realized action, both seats: the resolver's range inference is a
        # history replay, and B's sizes are the off-menu ones the shadow exists for.
        for observer in (agent, agent_b):
            if observer is not None:
                observer.observe(state, action)
        state = state.apply_action(action, rules)
        telemetry.note_action(action, state.stacks[actor_seat], is_resolver=is_a)

    if not state.ended_by_fold and len(state.board) < 5:
        state = complete_board(state, board_stack)

    if agent is not None and agent.resolver is not None:
        telemetry.fallbacks = agent.resolver.fallback_count
        telemetry.resolver_entropy_sum = agent.resolver.resolver_entropy_sum
        telemetry.played_entropy_sum = agent.resolver.played_entropy_sum
        telemetry.entropy_decisions = agent.resolver.entropy_decisions

    return float(state.get_payoff(a_seat, rules)), telemetry
