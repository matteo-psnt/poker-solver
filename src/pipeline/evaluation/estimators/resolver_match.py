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

Resolver lifecycle: a fresh :class:`BlueprintAgent` per game, matching its
documented contract (one resolver per agent lifetime, a new agent per hand),
with the per-game rng pinned so runout sampling is reproducible.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import FULL_DECK, Card, GameState, Street
from src.engine.search.agent import BlueprintAgent
from src.engine.solver.mccfr.chance import begin_street
from src.pipeline.evaluation.statistics import summarize_samples
from src.pipeline.evaluation.units import pair_mean_mbb
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.game.rules import GameRules
    from src.engine.solver.policy.source import ScorableBlueprint


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
    solver: ScorableBlueprint,
    *,
    num_deals: int = 1000,
    time_budget_ms: int = 100,
    seed: int = 1,
    workers: int = 1,
    blueprint_factory: Callable[[], ScorableBlueprint] | None = None,
    allin_runouts: int = 1,
) -> ResolverMatchResult:
    """Play duplicate deals of resolver-vs-blueprint and report the resolver edge.

    Positive ``resolver_mbb_per_hand`` means the resolver seat wins chips off the
    bare blueprint. ``resolver_fallbacks`` counts decisions where the resolver
    raised internally and fell back to the blueprint strategy (a high count means
    the number measures the fallback, not the resolver).

    ``workers`` above 1 needs a ``blueprint_factory``: the solver holds a
    non-picklable member, so a subprocess rebuilds it rather than receiving it
    (the seam parallel LBR and the public-tree BR already use). Deals are split
    into one CONTIGUOUS chunk per worker so the rebuild is paid once per
    process, not once per deal, and the chunks are reassembled in order -- every
    deal's result is a pure function of ``(seed, deal)``, so the parallel path
    returns bit-identical numbers to the serial one.

    Measured why: 100 deals took 62 minutes single-threaded on a 16-vCPU node,
    and se_mbb=1078 there puts ~50 mbb resolution at ~46,500 deals. Serially
    that is ~480 node-hours, which is not an experiment anyone runs.
    """
    rules = solver.rules
    big_blind = solver.config.game.big_blind
    starting_stack = solver.config.game.starting_stack

    if workers > 1 and blueprint_factory is not None and num_deals > 1:
        bounds = _chunk_bounds(num_deals, workers)
        with ProcessPoolExecutor(max_workers=len(bounds)) as pool:
            # `map` yields IN ORDER, which is what keeps the aggregate identical
            # to the serial path rather than merely equivalent.
            parts = list(
                pool.map(
                    _deals_worker,
                    repeat(blueprint_factory),
                    repeat(seed),
                    repeat(time_budget_ms),
                    [start for start, _ in bounds],
                    [stop for _, stop in bounds],
                    repeat(allin_runouts),
                )
            )
        pair_samples_mbb = [sample for part in parts for sample in part[0]]
        decisions = sum(part[1] for part in parts)
        fallbacks = sum(part[2] for part in parts)
    else:
        pair_samples_mbb, decisions, fallbacks = _play_deals(
            solver,
            rules,
            big_blind=big_blind,
            starting_stack=starting_stack,
            seed=seed,
            time_budget_ms=time_budget_ms,
            start=0,
            stop=num_deals,
            allin_runouts=allin_runouts,
        )

    summary = summarize_samples(pair_samples_mbb)
    return ResolverMatchResult(
        resolver_mbb_per_hand=summary["mean"],
        se_mbb=summary["se"],
        confidence_95_mbb=(summary["ci_lower"], summary["ci_upper"]),
        p_value=summary["p_value"],
        num_deals=num_deals,
        num_hands=2 * num_deals,
        resolver_decisions=decisions,
        resolver_fallbacks=fallbacks,
        pair_samples_mbb=pair_samples_mbb,
    )


def _chunk_bounds(num_deals: int, workers: int) -> list[tuple[int, int]]:
    """Contiguous [start, stop) deal ranges, one per worker, largest-first remainder."""
    count = min(workers, num_deals)
    size, extra = divmod(num_deals, count)
    bounds = []
    start = 0
    for index in range(count):
        stop = start + size + (1 if index < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def _deals_worker(
    factory: Callable[[], ScorableBlueprint],
    seed: int,
    time_budget_ms: int,
    start: int,
    stop: int,
    allin_runouts: int,
) -> tuple[list[float], int, int]:
    """One contiguous block of deals, in its own process.

    Rebuilds the blueprint rather than receiving it; returns the raw parts so
    the parent aggregates exactly as the serial path does, instead of this
    duplicating that arithmetic and letting the two drift.
    """
    # Spawned: logging config does not inherit, and factory() rebuilds the
    # blueprint, which logs.
    configure_logging()
    solver = factory()
    return _play_deals(
        solver,
        solver.rules,
        big_blind=solver.config.game.big_blind,
        starting_stack=solver.config.game.starting_stack,
        seed=seed,
        time_budget_ms=time_budget_ms,
        start=start,
        stop=stop,
        allin_runouts=allin_runouts,
    )


def _play_deals(
    solver: ScorableBlueprint,
    rules: GameRules,
    *,
    big_blind: int,
    starting_stack: int,
    seed: int,
    time_budget_ms: int,
    start: int,
    stop: int,
    allin_runouts: int = 1,
) -> tuple[list[float], int, int]:
    """Deals ``[start, stop)``, shared by the serial and parallel paths.

    ONE implementation on purpose: two copies of this drifting apart would leave
    the paths silently unpaired -- still returning numbers, just not the same
    ones -- which is the failure `deal_for` is commented against.
    """
    pair_samples_mbb: list[float] = []
    decisions = 0
    fallbacks = 0

    for deal in range(start, stop):
        hole_cards, board_stack, button = deal_for(seed, deal)

        seat_payoffs: list[float] = []
        for resolver_seat in (0, 1):
            # The bare-blueprint opponent samples its mixed strategy from the global
            # legacy RNG (sample_action_from_strategy -> np.random.choice), which
            # play_resolver_match never seeds -- so without this the advertised
            # ``seed`` did not actually determine the result (the opponent consumed
            # whatever ambient global-RNG state the process was in). Seed it per game
            # for reproducibility, mirroring blueprint_match's per-game np.random.seed.
            np.random.seed(
                int(np.random.SeedSequence([seed, deal, resolver_seat]).generate_state(1)[0])
            )
            payoff, game_decisions, game_fallbacks = _play_game(
                solver,
                rules,
                hole_cards=hole_cards,
                board_stack=board_stack,
                button=button,
                starting_stack=starting_stack,
                resolver_seat=resolver_seat,
                time_budget_ms=time_budget_ms,
                resolver_rng=np.random.default_rng(
                    np.random.SeedSequence([seed, deal, resolver_seat])
                ),
                allin_runouts=allin_runouts,
                # Seeded per DEAL, deliberately without `resolver_seat`: the two
                # halves of a duplicate pair must see the SAME completions, or
                # the averaging introduces exactly the between-seat card luck the
                # pairing exists to cancel.
                runout_rng=np.random.default_rng(np.random.SeedSequence([seed, deal])),
            )
            seat_payoffs.append(payoff)
            decisions += game_decisions
            fallbacks += game_fallbacks

        payoff_seat0, payoff_seat1 = seat_payoffs
        pair_samples_mbb.append(pair_mean_mbb(payoff_seat0, payoff_seat1, big_blind))

    return pair_samples_mbb, decisions, fallbacks


def _play_game(
    solver: ScorableBlueprint,
    rules: GameRules,
    *,
    hole_cards,
    board_stack: list[Card],
    button: int,
    starting_stack: int,
    resolver_seat: int,
    time_budget_ms: int,
    resolver_rng: np.random.Generator | None = None,
    allin_runouts: int = 1,
    runout_rng: np.random.Generator | None = None,
) -> tuple[float, int, int]:
    """One game off a fixed deck; returns (resolver-seat payoff, decisions, fallbacks)."""
    agent = BlueprintAgent(solver, use_resolver=True, rng=resolver_rng)
    state = rules.create_initial_state(
        starting_stack=starting_stack,
        hole_cards=hole_cards,
        button=button,
    )

    decisions = 0
    while not state.is_terminal:
        if solver.is_chance_node(state):
            state = deal_from_stack(state, board_stack)
            continue
        if state.current_player == resolver_seat:
            decisions += 1
            action = agent.act(state, time_budget_ms=time_budget_ms)
        else:
            action = solver.sample_action_from_strategy(state, use_average=True)
        # History-replay range inference: the resolver observes every realized
        # action (both seats), so its next solve sees Bayes-updated ranges
        # instead of uniform ones.
        agent.observe(state, action)
        state = state.apply_action(action, rules)

    assert agent.resolver is not None  # use_resolver=True above
    if not state.ended_by_fold and len(state.board) < 5:
        payoff = _allin_payoff(
            state, rules, board_stack, resolver_seat, runouts=allin_runouts, rng=runout_rng
        )
    else:
        payoff = float(state.get_payoff(resolver_seat, rules))
    return payoff, decisions, agent.resolver.fallback_count


def deal_for(
    seed: int, deal: int
) -> tuple[tuple[tuple[Card, Card], tuple[Card, Card]], list[Card], int]:
    """The hole cards, board stack and button for one deal index.

    Shared by both match scorers deliberately. Paired comparison is only valid
    while they draw the SAME deals for a given ``(seed, deal)``: two copies of
    this drifting apart would leave the scorers silently unpaired -- still
    returning numbers, just no longer numbers that may be compared -- rather
    than visibly broken.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, deal]))
    order = [int(i) for i in rng.permutation(52)]
    hole_cards = (
        (FULL_DECK[order[0]], FULL_DECK[order[1]]),
        (FULL_DECK[order[2]], FULL_DECK[order[3]]),
    )
    board_stack = [FULL_DECK[i] for i in order[4:9]]  # flop, flop, flop, turn, river
    return hole_cards, board_stack, deal % 2


def deal_from_stack(state: GameState, board_stack: list[Card]) -> GameState:
    """Deal the street's cards from fixed deck positions (duplicate-poker dealing)."""
    board_size = len(state.board)
    if state.street == Street.FLOP and board_size == 0:
        cards = board_stack[:3]
    elif state.street == Street.TURN and board_size == 3:
        cards = board_stack[3:4]
    elif state.street == Street.RIVER and board_size == 4:
        cards = board_stack[4:5]
    else:
        return state
    return begin_street(state, cards)


# A flop all-in leaves C(45,2) = 990 completions and a turn all-in 44; both are
# cheap enough to enumerate exactly. Above this, sample.
_MAX_EXACT_RUNOUTS = 1200


def _remaining_deck(state: GameState) -> list[Card]:
    """Cards neither player holds and the board has not shown."""
    known = {card.mask for hand in state.hole_cards for card in hand}
    known |= {card.mask for card in state.board}
    return [card for card in FULL_DECK if card.mask not in known]


def _allin_payoff(
    state: GameState,
    rules: GameRules,
    board_stack: list[Card],
    resolver_seat: int,
    *,
    runouts: int,
    rng: np.random.Generator | None,
) -> float:
    """Payoff of an all-in hand, averaged over completions of the board.

    The dealt board is ONE sample of a quantity whose expectation is what the
    match is trying to measure, and all-in pots are the biggest pots -- so this
    single draw contributes far more than its share of the per-deal spread. The
    measured spread after duplicate-deal pairing was 9.2 BB, which needs ~32,000
    deals to resolve 100 mbb; averaging here attacks that directly, at the SAME
    expectation.

    EXACT WHERE IT CAN BE. A turn all-in has 44 completions and a flop all-in
    990 -- both cheap to enumerate, and enumeration is not an approximation but
    the exact expectation, contributing ZERO variance. Only the deep cases (a
    preflop all-in leaves C(48,5) = 1.7M) fall back to sampling, which is what
    LBR does at every one of its all-in terminals.

    ``runouts <= 1`` reproduces the single dealt board exactly, which is the
    shipped behaviour and the default.
    """
    if runouts <= 1:
        return float(complete_board(state, board_stack).get_payoff(resolver_seat, rules))

    deck = _remaining_deck(state)
    needed = 5 - len(state.board)
    if needed <= 0:
        return float(state.get_payoff(resolver_seat, rules))

    exact = list(itertools.combinations(deck, needed)) if needed <= 2 else []
    if exact and len(exact) <= _MAX_EXACT_RUNOUTS:
        completions: list[tuple[Card, ...]] = exact
    else:
        # Only the sampling branch needs a generator, and a caller asking for
        # >1 runout without one would be asking for an irreproducible number.
        if rng is None:
            raise ValueError("allin_runouts > 1 needs a runout_rng to stay reproducible")
        completions = [
            tuple(deck[i] for i in rng.choice(len(deck), size=needed, replace=False))
            for _ in range(runouts)
        ]

    total = 0.0
    for completion in completions:
        finished = state.replace(
            street=Street.RIVER,
            board=tuple(state.board) + completion,
            is_terminal=True,
            to_call=0,
        )
        total += float(finished.get_payoff(resolver_seat, rules))
    return total / len(completions)


def complete_board(state: GameState, board_stack: list[Card]) -> GameState:
    """Complete an all-in board from the same fixed deck positions."""
    return state.replace(
        street=Street.RIVER,
        board=tuple(board_stack[:5]),
        is_terminal=True,
        to_call=0,
    )
