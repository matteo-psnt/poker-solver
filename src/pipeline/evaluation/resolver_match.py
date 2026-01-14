"""Resolver gate: blueprint+resolver vs bare blueprint, on duplicate deals.

Answers one question cheaply: does routing decisions through the runtime
subgame resolver (:class:`~src.engine.search.resolver.HUResolver`) beat playing
the raw blueprint? This is the deployment-relevant comparison — the resolver is
how the blueprint is actually played (``resolver.enabled`` defaults ``True``) —
and it gates any investment in resolver-in-eval integration.

Variance design (duplicate poker):
    Every deal is played twice with the *same fixed deck order* and the resolver
    controlling opposite seats. Board cards come off fixed deck positions, so
    whenever the two games reach the same street they see the same cards. The
    per-deal sample is the resolver seat's net over the pair, which cancels the
    deal's card luck — the dominant noise in head-to-head play — leaving mostly
    the skill difference.

Resolver lifecycle: a fresh :class:`HUResolver` per game (memoryless across
hands, continual within a hand). ``MCCFRSolver.act`` caches one resolver for the
process lifetime, which would leak ``_ranges`` across hands — deliberately not
used here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from src.core.game.actions import ActionType
from src.core.game.state import Card, GameState, Street
from src.engine.search.resolver import HUResolver

_DECK: list[Card] = Card.get_full_deck()


@dataclass(frozen=True)
class ResolverMatchResult:
    """Outcome of a duplicate-deal resolver-vs-blueprint match."""

    resolver_mbb_per_hand: float
    se_mbb: float
    confidence_95_mbb: tuple[float, float]
    p_value: float
    num_deals: int
    num_hands: int
    resolver_decisions: int
    resolver_fallbacks: int
    pair_samples_mbb: list[float]


def play_resolver_match(
    solver,
    *,
    num_deals: int = 1000,
    time_budget_ms: int = 100,
    seed: int = 1,
) -> ResolverMatchResult:
    """Play duplicate deals of resolver-vs-blueprint and report the resolver edge.

    Positive ``resolver_mbb_per_hand`` means the resolver seat wins chips off the
    bare blueprint. ``resolver_fallbacks`` counts decisions where the resolver
    raised internally and fell back to the blueprint strategy (a high count means
    the number measures the fallback, not the resolver).
    """
    rules = solver.rules
    big_blind = solver.config.game.big_blind
    starting_stack = solver.config.game.starting_stack

    pair_samples_mbb: list[float] = []
    decisions = 0
    fallbacks = 0

    for deal in range(num_deals):
        rng = np.random.default_rng(np.random.SeedSequence([seed, deal]))
        order = [int(i) for i in rng.permutation(52)]
        hole_cards = (
            (_DECK[order[0]], _DECK[order[1]]),
            (_DECK[order[2]], _DECK[order[3]]),
        )
        board_stack = [_DECK[i] for i in order[4:9]]  # flop, flop, flop, turn, river
        button = deal % 2

        pair_net = 0.0
        for resolver_seat in (0, 1):
            payoff, game_decisions, game_fallbacks = _play_game(
                solver,
                rules,
                hole_cards=hole_cards,
                board_stack=board_stack,
                button=button,
                starting_stack=starting_stack,
                resolver_seat=resolver_seat,
                time_budget_ms=time_budget_ms,
            )
            pair_net += payoff
            decisions += game_decisions
            fallbacks += game_fallbacks

        pair_samples_mbb.append(pair_net / (2.0 * big_blind) * 1000.0)

    samples = np.asarray(pair_samples_mbb, dtype=np.float64)
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(len(samples))) if len(samples) >= 2 else 0.0
    if se > 0:
        p_value = float(scipy_stats.ttest_1samp(samples, 0.0).pvalue)
    else:
        p_value = 1.0 if mean == 0.0 else 0.0

    return ResolverMatchResult(
        resolver_mbb_per_hand=mean,
        se_mbb=se,
        confidence_95_mbb=(mean - 1.96 * se, mean + 1.96 * se),
        p_value=p_value,
        num_deals=num_deals,
        num_hands=2 * num_deals,
        resolver_decisions=decisions,
        resolver_fallbacks=fallbacks,
        pair_samples_mbb=pair_samples_mbb,
    )


def _play_game(
    solver,
    rules,
    *,
    hole_cards,
    board_stack: list[Card],
    button: int,
    starting_stack: int,
    resolver_seat: int,
    time_budget_ms: int,
) -> tuple[float, int, int]:
    """One game off a fixed deck; returns (resolver-seat payoff, decisions, fallbacks)."""
    resolver = HUResolver(
        blueprint=solver,
        action_model=solver.action_model,
        rules=rules,
        config=solver.config.resolver,
    )
    state = rules.create_initial_state(
        starting_stack=starting_stack,
        hole_cards=hole_cards,
        button=button,
    )

    decisions = 0
    fallbacks = 0
    while not state.is_terminal:
        if solver.is_chance_node(state):
            state = _deal_from_stack(state, board_stack)
            continue
        if state.current_player == resolver_seat:
            decisions += 1
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                action = resolver.act(state, time_budget_ms=time_budget_ms)
            fallbacks += sum(issubclass(w.category, RuntimeWarning) for w in caught)
        else:
            action = solver.sample_action_from_strategy(state, use_average=True)
        state = state.apply_action(action, rules)

    is_showdown = bool(state.betting_history) and state.betting_history[-1].type != ActionType.FOLD
    if is_showdown and len(state.board) < 5:
        state = _complete_board(state, board_stack)

    return float(state.get_payoff(resolver_seat, rules)), decisions, fallbacks


def _deal_from_stack(state: GameState, board_stack: list[Card]) -> GameState:
    """Deal the street's cards from fixed deck positions (duplicate-poker dealing)."""
    board_size = len(state.board)
    new_board = list(state.board)
    if state.street == Street.FLOP and board_size == 0:
        new_board.extend(board_stack[:3])
    elif state.street == Street.TURN and board_size == 3:
        new_board.append(board_stack[3])
    elif state.street == Street.RIVER and board_size == 4:
        new_board.append(board_stack[4])
    else:
        return state

    return GameState(
        street=state.street,
        pot=state.pot,
        stacks=state.stacks,
        board=tuple(new_board),
        hole_cards=state.hole_cards,
        betting_history=state.betting_history,
        button_position=state.button_position,
        current_player=1 - state.button_position,
        is_terminal=False,
        to_call=0,
        last_aggressor=None,
        blind_to_call=state.blind_to_call,
    )


def _complete_board(state: GameState, board_stack: list[Card]) -> GameState:
    """Complete an all-in board from the same fixed deck positions."""
    return GameState(
        street=Street.RIVER,
        pot=state.pot,
        stacks=state.stacks,
        board=tuple(board_stack[:5]),
        hole_cards=state.hole_cards,
        betting_history=state.betting_history,
        button_position=state.button_position,
        current_player=state.current_player,
        is_terminal=True,
        to_call=0,
        last_aggressor=state.last_aggressor,
        blind_to_call=state.blind_to_call,
    )
