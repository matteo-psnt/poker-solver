"""Deterministic exact best response over a sampled public betting tree.

Every existing HUNL evaluator in this package is Monte Carlo over deals (LBR,
blueprint match, rollout), so between-eval noise of hundreds of mbb swamps the
effects being measured. This module computes an *exact* best response — full
lookahead, per-combo, range-vs-range with exact card removal — against the
blueprint's average strategy, on a deterministic restriction of the game:

- The betting tree is the blueprint's own action abstraction (on-tree BR; the
  off-tree dimension remains LBR's job).
- Chance is restricted to a fixed, seed-deterministic board sample: a weighted
  subset of canonical flops and, per board, fixed turn/river card subsets.
  Branch weights are public (blocker-blind) probabilities; a deal incompatible
  with a sampled branch contributes zero mass on that branch (the "annulled"
  game). Dial the sample up and the value converges to the classic
  full-enumeration abstraction BR of Johanson et al. (2011).

The output is a point value with zero evaluation variance: the same checkpoint
always scores identically, and two checkpoints scored under the same
``PublicBRConfig`` are exactly paired — any difference is pure signal. The
absolute number is the exploitability of the restricted game, not full HUNL
(it deflates as the board sample thins); cross-checkpoint comparison is the
intended use.

Measure and normalization: hands are dealt uniformly over the 1326 x 1225
ordered disjoint pairs. The walk carries the opponent's unnormalized reach
vector down and the responder's per-combo counterfactual values up; chance
weights fold into the reach vector, and the root value divides by 1326 x 1225.
The responder sees real combos (every combo is its own information set, so the
per-combo max is information-set consistent and strictly stronger than any
bucket-constrained responder). Blueprint policy queries go through
``blueprint_action_distribution`` with the same uniform-over-legal fallback the
deployed blueprint plays, so the responder exploits exactly the strategy that
would be fielded.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import FULL_DECK, Card, GameState, Street
from src.engine.search.range_inference import ALL_COMBOS, NUM_COMBOS, blocked_combos
from src.engine.search.subgame_cfr import RunoutEvaluator, nonblocking_mass
from src.engine.solver.infoset_encoder import get_spr_bucket
from src.engine.solver.infoset_index import preflop_hand_index
from src.engine.solver.policy_lookup import blueprint_action_distribution
from src.engine.solver.policy_source import ScorableBlueprint
from src.pipeline.abstraction.postflop.board_enumeration import CanonicalBoardEnumerator
from src.shared.log import configure_logging

logger = logging.getLogger(__name__)

_NUM_OPP_DEALS = math.comb(50, 2)
_CARD_INDEX: dict[int, int] = {card.mask: i for i, card in enumerate(FULL_DECK)}
_TURN_STREAM = 1
_RIVER_STREAM = 2


def _build_preflop_class_of_combo() -> np.ndarray:
    """Combo -> 169-class id, in the SOLVER's canonical preflop ordering.

    Previously this assigned ids in discovery order over ``ALL_COMBOS``, which
    was self-consistent but was a SECOND ordering of the 169 classes. The policy
    rows below are built per id and then gathered through this map, so the two
    must agree exactly; two independently-derived orderings agreeing was an
    accident waiting to stop happening. Deferring to ``infoset_index`` makes the
    solver's ordering the only one.
    """
    return np.array([preflop_hand_index(combo) for combo in ALL_COMBOS], dtype=np.int64)


_PREFLOP_CLASS_OF_COMBO = _build_preflop_class_of_combo()


@dataclass(frozen=True)
class PublicBRConfig:
    """Board-sampling knobs; together with the checkpoint they pin the result.

    num_flops: canonical flops drawn (with replacement, weighted by raw
        suit-isomorphism count; duplicates merge into one branch with summed
        weight). >= 1755 means exact full flop enumeration.
    num_turns / num_rivers: fixed card subsets drawn per board node (>= 49/48
        means exact enumeration for that street).
    board_seed: seeds every board draw; identical seeds give identical board
        samples and therefore exactly paired evaluations.
    num_workers: processes over the four (responder seat, button) walks, which
        are independent. 1 keeps everything in-process. Above 1 requires a
        ``blueprint_factory``, since the solver is not picklable. Does NOT
        change the result: each walk is deterministic and the aggregate is a
        mean over the same four numbers.

    Node count scales ~linearly in num_flops * num_turns * num_rivers, but
    wall-clock is strongly sublinear: per-context policy tables and showdown
    evaluators amortize across boards. Measured on a 6.8M-infoset production
    checkpoint (full 4-walk evaluation, single-core): 2 board paths -> 1.3M
    nodes / ~7 min; 16 paths -> 9.7M nodes / ~13.5 min. Prefer the Modal
    entrypoint for production checkpoints.
    """

    num_flops: int = 8
    num_turns: int = 2
    num_rivers: int = 2
    board_seed: int = 7
    num_workers: int = 1


@dataclass(frozen=True)
class SeatResult:
    """Best-response value for one (responder seat, button) configuration."""

    br_seat: int
    button: int
    value_mbb: float
    missing_policy_mass: float


@dataclass(frozen=True)
class PublicBRResult:
    """Exact BR values on the restricted game, in mbb per hand.

    exploitability_mbb: mean of the four (seat, button) BR values; >= 0 up to
        float error, 0 iff the blueprint is a restricted-game equilibrium.
    missing_policy_mass: reach-weighted fraction of opponent decisions that
        fell back to uniform (untrained infoset) — a large value means the BR
        is partly exploiting the fallback, not the trained strategy.
    """

    exploitability_mbb: float
    seat_results: tuple[SeatResult, ...]
    missing_policy_mass: float
    nodes_visited: int
    num_flops: int
    elapsed_s: float
    config: PublicBRConfig


class _BoardPlan:
    """Deterministic sampled chance: fixed flop set, fixed per-board turn/river sets."""

    def __init__(self, config: PublicBRConfig):
        self._config = config
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        infos = list(enumerator.iterate())
        counts = np.array([info.raw_count for info in infos], dtype=np.float64)
        probs = counts / counts.sum()
        if config.num_flops >= len(infos):
            self.flops = [
                (info.representative, float(p)) for info, p in zip(infos, probs, strict=True)
            ]
        else:
            rng = np.random.default_rng(np.random.SeedSequence([config.board_seed]))
            draws = rng.choice(len(infos), size=config.num_flops, replace=True, p=probs)
            unique, tallies = np.unique(draws, return_counts=True)
            self.flops = [
                (infos[int(i)].representative, float(n) / config.num_flops)
                for i, n in zip(unique, tallies, strict=True)
            ]

    def deal_options(self, board: tuple[Card, ...]) -> list[tuple[tuple[Card, ...], float]]:
        """Weighted card branches extending ``board`` by one deal (public weights)."""
        if len(board) == 0:
            return self.flops
        if len(board) == 3:
            cards = self._street_cards(board, _TURN_STREAM, self._config.num_turns)
        elif len(board) == 4:
            cards = self._street_cards(board, _RIVER_STREAM, self._config.num_rivers)
        else:
            raise ValueError(f"No deal extends a board of {len(board)} cards")
        weight = 1.0 / len(cards)
        return [((card,), weight) for card in cards]

    def _street_cards(self, board: tuple[Card, ...], stream: int, count: int) -> list[Card]:
        board_mask = 0
        for card in board:
            board_mask |= card.mask
        candidates = [card for card in FULL_DECK if not (card.mask & board_mask)]
        if count >= len(candidates):
            return candidates
        board_key = sorted(_CARD_INDEX[card.mask] for card in board)
        seed = np.random.SeedSequence([self._config.board_seed, stream, *board_key])
        rng = np.random.default_rng(seed)
        picks = np.sort(rng.choice(len(candidates), size=count, replace=False))
        return [candidates[int(i)] for i in picks]


class PublicTreeBestResponse:
    """Exact best response of one seat against the blueprint on the sampled tree."""

    def __init__(
        self,
        blueprint: ScorableBlueprint,
        config: PublicBRConfig,
        *,
        starting_stack: int,
        blueprint_factory: Callable[[], ScorableBlueprint] | None = None,
    ):
        self._policy_source = blueprint.policy_source
        self._factory = blueprint_factory
        self._rules = blueprint.rules
        self._action_model = blueprint.action_model
        self._abstraction = blueprint.card_abstraction
        self._config = config
        self._plan = _BoardPlan(config)
        self._starting_stack = starting_stack
        self._dummy_holes = ((FULL_DECK[0], FULL_DECK[1]), (FULL_DECK[2], FULL_DECK[3]))
        self._bucket_cache: dict[tuple, np.ndarray] = {}
        self._showdown_cache: dict[tuple, RunoutEvaluator] = {}
        self._policy_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._br_seat = 0
        self._nodes = 0
        self._decision_mass = 0.0
        self._missing_mass = 0.0

    def evaluate(self) -> PublicBRResult:
        """Run all four (responder seat, button) walks and aggregate."""
        start = time.perf_counter()
        big_blind = self._rules.big_blind
        seat_results = []
        total_nodes = 0
        total_decision = 0.0
        total_missing = 0.0
        walks = [(s, b) for s in (0, 1) for b in (0, 1)]
        if self._config.num_workers > 1 and self._factory is not None:
            # Independent walks: same tree, disjoint responder/button, no shared
            # mutable state. Ordered results, so seat_results stays in the same
            # order the serial path produces.
            with ProcessPoolExecutor(max_workers=min(self._config.num_workers, len(walks))) as pool:
                parts = list(
                    pool.map(
                        _walk_worker,
                        repeat(self._factory),
                        repeat(self._config),
                        repeat(self._starting_stack),
                        [s for s, _ in walks],
                        [b for _, b in walks],
                    )
                )
        else:
            parts = [self.run_walk(br_seat, button) for br_seat, button in walks]

        for (br_seat, button), (chips, nodes, decision, missing) in zip(walks, parts, strict=True):
            fraction = missing / decision if decision else 0.0
            seat_results.append(
                SeatResult(
                    br_seat=br_seat,
                    button=button,
                    value_mbb=chips / big_blind * 1000.0,
                    missing_policy_mass=fraction,
                )
            )
            total_nodes += nodes
            total_decision += decision
            total_missing += missing
        exploitability = float(np.mean([r.value_mbb for r in seat_results]))
        return PublicBRResult(
            exploitability_mbb=exploitability,
            seat_results=tuple(seat_results),
            missing_policy_mass=total_missing / total_decision if total_decision else 0.0,
            nodes_visited=total_nodes,
            num_flops=len(self._plan.flops),
            elapsed_s=time.perf_counter() - start,
            config=self._config,
        )

    def responder_values(self, br_seat: int, button: int, opp_reach: np.ndarray) -> np.ndarray:
        """Responder's per-combo counterfactual chip values against ``opp_reach``.

        ``values[h]`` sums utility over every opponent combo weighted by
        ``opp_reach`` and every sampled board branch; dividing by the
        compatible opponent mass turns it into a per-hand expectation. Exposed
        for validation against the scalar reference game and for range-level
        analysis; :meth:`evaluate` is the standard entry point.
        """
        self._br_seat = br_seat
        self._nodes = 0
        self._decision_mass = 0.0
        self._missing_mass = 0.0
        root = self._rules.create_initial_state(
            starting_stack=self._starting_stack, hole_cards=self._dummy_holes, button=button
        )
        return self._walk(root, opp_reach.astype(np.float64))

    def _run(self, br_seat: int, button: int) -> float:
        values = self.responder_values(br_seat, button, np.ones(NUM_COMBOS, dtype=np.float64))
        return float(values.sum()) / (NUM_COMBOS * _NUM_OPP_DEALS)

    def run_walk(self, br_seat: int, button: int) -> tuple[float, int, float, float]:
        """One walk's chips plus the telemetry the aggregate needs.

        Public because a worker process runs exactly this and nothing else, and
        because returning the raw parts keeps the value/mass arithmetic in ONE
        place -- the serial and parallel paths then cannot drift.
        """
        chips = self._run(br_seat, button)
        return chips, self._nodes, self._decision_mass, self._missing_mass

    def _walk(self, state: GameState, opp_reach: np.ndarray) -> np.ndarray:
        """Responder's per-combo counterfactual values under best play below ``state``."""
        self._nodes += 1
        if not opp_reach.any():
            return np.zeros(NUM_COMBOS, dtype=np.float64)
        if state.is_terminal:
            return self._terminal_values(state, opp_reach)
        if len(state.board) < state.street.board_card_count:
            return self._deal_values(state, opp_reach)
        legal = self._rules.get_legal_actions(state, self._action_model)
        if state.current_player == self._br_seat:
            children = [
                self._walk(self._rules.apply_action(state, action), opp_reach) for action in legal
            ]
            return np.maximum.reduce(children)
        return self._opponent_values(state, legal, opp_reach)

    def _terminal_values(self, state: GameState, opp_reach: np.ndarray) -> np.ndarray:
        if state.ended_by_fold:
            payoff = self._rules.get_payoff(state, self._br_seat)
            return payoff * nonblocking_mass(opp_reach)
        invested = self._rules.invested_chips(state)[self._br_seat]
        return self._runout_values(state.board, float(state.pot), invested, opp_reach)

    def _runout_values(
        self, board: tuple[Card, ...], pot: float, invested: float, opp_reach: np.ndarray
    ) -> np.ndarray:
        """Showdown values, completing an all-in board over the sampled deals."""
        if len(board) == 5:
            win, tie, alive = self._showdown_evaluator(board).masses(opp_reach)
            return win * pot + tie * (pot / 2.0) - invested * alive
        values = np.zeros(NUM_COMBOS, dtype=np.float64)
        for cards, weight in self._plan.deal_options(board):
            block = blocked_combos(cards)
            child_reach = np.where(block, 0.0, opp_reach) * weight
            if not child_reach.any():
                continue
            child = self._runout_values(board + cards, pot, invested, child_reach)
            child[block] = 0.0
            values += child
        return values

    def _deal_values(self, state: GameState, opp_reach: np.ndarray) -> np.ndarray:
        first_to_act = 1 - state.button_position
        values = np.zeros(NUM_COMBOS, dtype=np.float64)
        for cards, weight in self._plan.deal_options(state.board):
            block = blocked_combos(cards)
            child_reach = np.where(block, 0.0, opp_reach) * weight
            if not child_reach.any():
                continue
            child_state = state.replace(
                board=(*state.board, *cards),
                current_player=first_to_act,
                is_terminal=False,
                to_call=0,
                last_aggressor=None,
            )
            child = self._walk(child_state, child_reach)
            child[block] = 0.0
            values += child
        return values

    def _opponent_values(
        self, state: GameState, legal: tuple[Action, ...], opp_reach: np.ndarray
    ) -> np.ndarray:
        sigma, missing = self._policy_matrix(state, legal)
        self._decision_mass += float(opp_reach.sum())
        self._missing_mass += float(opp_reach[missing].sum())
        values = np.zeros(NUM_COMBOS, dtype=np.float64)
        for a_idx, action in enumerate(legal):
            child_reach = opp_reach * sigma[:, a_idx]
            if not child_reach.any():
                continue
            values += self._walk(self._rules.apply_action(state, action), child_reach)
        return values

    def _policy_matrix(
        self, state: GameState, legal: tuple[Action, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-combo blueprint distribution over ``legal`` plus uniform-fallback mask.

        The per-bucket rows depend only on the betting context, never the board,
        so they are built densely over all buckets once per context and every
        node with that context (one per sampled board) reduces to a vectorized
        gather through the board's bucket vector.
        """
        sequence = state.normalized_betting_sequence()
        spr = min(state.stacks) / state.pot if state.pot > 0 else 0
        spr_bucket = get_spr_bucket(spr)
        context_key = (
            state.current_player,
            state.street,
            sequence,
            spr_bucket,
            legal,
            state.pot,
            state.stacks,
            state.to_call,
        )
        cached = self._policy_cache.get(context_key)
        if cached is None:
            num_buckets = self._policy_source.num_buckets(state.street)
            rows = np.empty((num_buckets, len(legal)), dtype=np.float64)
            row_missing = np.empty(num_buckets, dtype=bool)
            for bucket in range(num_buckets):
                rows[bucket], row_missing[bucket] = self._policy_row(state, legal, bucket)
            cached = (rows, row_missing)
            self._policy_cache[context_key] = cached
        rows, row_missing = cached
        bucket_vec = self._bucket_vector(state.board, state.street)
        alive = bucket_vec >= 0
        sigma = np.zeros((NUM_COMBOS, len(legal)), dtype=np.float64)
        sigma[alive] = rows[bucket_vec[alive]]
        missing = np.zeros(NUM_COMBOS, dtype=bool)
        missing[alive] = row_missing[bucket_vec[alive]]
        return sigma, missing

    def _policy_row(
        self,
        state: GameState,
        legal: tuple[Action, ...],
        bucket: int,
    ) -> tuple[np.ndarray, bool]:
        distribution = blueprint_action_distribution(
            self._policy_source.infoset_at(state, bucket),
            state,
            self._rules,
            legal,
            use_average=True,
        )
        if distribution is None:
            return np.full(len(legal), 1.0 / len(legal)), True
        return np.array([distribution.get(a, 0.0) for a in legal]), False

    def _bucket_vector(self, board: tuple[Card, ...], street: Street) -> np.ndarray:
        """Combo -> bucket id over ALL_COMBOS (-1 where blocked by the board)."""
        if street == Street.PREFLOP:
            return _PREFLOP_CLASS_OF_COMBO
        cache_key = (street, tuple(sorted(card.mask for card in board)))
        cached = self._bucket_cache.get(cache_key)
        if cached is not None:
            return cached
        vector = np.full(NUM_COMBOS, -1, dtype=np.int64)
        blocked = blocked_combos(board)
        for i in np.nonzero(~blocked)[0]:
            vector[i] = self._abstraction.get_bucket(ALL_COMBOS[int(i)], board, street)
        self._bucket_cache[cache_key] = vector
        return vector

    def _showdown_evaluator(self, board: tuple[Card, ...]) -> RunoutEvaluator:
        cache_key = tuple(sorted(card.mask for card in board))
        cached = self._showdown_cache.get(cache_key)
        if cached is None:
            cached = RunoutEvaluator(board)
            self._showdown_cache[cache_key] = cached
        return cached


def _walk_worker(
    factory: Callable[[], ScorableBlueprint],
    config: PublicBRConfig,
    starting_stack: int,
    br_seat: int,
    button: int,
) -> tuple[float, int, float, float]:
    """One (responder seat, button) walk, in its own process.

    Rebuilds the blueprint rather than receiving it: the solver holds a
    non-picklable member, which is the same reason parallel LBR takes a factory.
    Returns the raw parts so the parent aggregates exactly as the serial path
    does -- deriving the seat value here would duplicate that arithmetic in two
    places and let the two drift.
    """
    # Spawned: logging config does not inherit, and factory() rebuilds the
    # blueprint, which logs.
    configure_logging()
    engine = PublicTreeBestResponse(factory(), config, starting_stack=starting_stack)
    return engine.run_walk(br_seat, button)


def compute_public_tree_br(
    blueprint: ScorableBlueprint,
    config: PublicBRConfig,
    *,
    starting_stack: int,
    blueprint_factory: Callable[[], ScorableBlueprint] | None = None,
) -> PublicBRResult:
    """Exact best response against ``blueprint`` on the sampled public tree.

    ``blueprint_factory`` enables ``config.num_workers > 1``: workers rebuild the
    blueprint rather than receiving it, since the solver is not picklable.
    """
    engine = PublicTreeBestResponse(
        blueprint, config, starting_stack=starting_stack, blueprint_factory=blueprint_factory
    )
    result = engine.evaluate()
    logger.info(
        "public-tree BR: %.1f mbb over %d flops (%d nodes, %.1fs, %.1f%% uniform-fallback mass)",
        result.exploitability_mbb,
        result.num_flops,
        result.nodes_visited,
        result.elapsed_s,
        100.0 * result.missing_policy_mass,
    )
    return result
