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
  game). Dial the sample up and the value converges to the exploitability of
  the full-enumeration annulled game -- NOT of the real game: every deal blocks
  four cards, so a void refunds the whole hand with probability 0.217 at the
  flop, 4/49 at the turn and 4/48 at the river, and a postflop terminal is
  worth 0.66-0.78 of a preflop fold. ``conditional_chance`` divides each
  street's weights by that compatible fraction, which is exact at full
  enumeration (the classic abstraction BR of Johanson et al. 2011) and
  unbiased over deals at any sample.

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
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.game.state import FULL_DECK, Card, GameState, Street
from src.engine.search.range_inference import ALL_COMBOS, NUM_COMBOS, blocked_combos
from src.engine.search.subgame_cfr import RunoutEvaluator, nonblocking_mass
from src.engine.solver.infoset.encoder import get_spr_bucket
from src.engine.solver.infoset.index import preflop_hand_index
from src.engine.solver.mccfr.chance import begin_street
from src.engine.solver.policy.lookup import blueprint_policy_table
from src.pipeline.abstraction.postflop.board_enumeration import CanonicalBoardEnumerator
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.game.actions import Action
    from src.engine.solver.policy.source import ScorableBlueprint

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
    conditional_chance: weight each street by 1 / (fraction of its draws a
        four-card deal leaves compatible), so a void is no longer a refund. A
        different game from the annulled one: a separate comparison tier.
    num_workers: processes over the flop subtrees (one job per preflop line x
        sampled flop x walk), a fork-join BELOW the preflop best-response max,
        where the walk is a plain weighted sum. Above 1 requires a
        ``blueprint_factory``, since the solver is not picklable. Does NOT
        change the result: every subtree is deterministic and the join sums in
        the serial order, so the number is bit-identical at any worker count.

    Node count scales ~linearly in num_flops * num_turns * num_rivers, and
    wall-clock with it now that the per-context policy tables are one slice
    each: 4/2/2 on a 30M production checkpoint is 9.7M nodes in 82 s over four
    walks on a pool node (08-22; was 239 s). Run production checkpoints on the
    pool.
    """

    num_flops: int = 8
    num_turns: int = 2
    num_rivers: int = 2
    board_seed: int = 7
    conditional_chance: bool = False
    num_workers: int = 1
    in_abstraction: bool = False
    """The responder picks ONE action per (public node, bucket), maximising the
    bucket-summed counterfactual value: the abstract game's own best response,
    the figure a converged abstract-game solver drives to zero. Own-reach is not
    in the weighting (imperfect recall makes the reach-aware optimum circular),
    so it is a valid constrained strategy's value and a lower bound on the true
    in-abstraction BR -- the standard CFR-BR convention."""
    policy_threshold: float = 0.0
    """Eval-time transform of the strategy under measurement: blueprint actions
    below this probability are zeroed and the row renormalised (Ganzfried &
    Sandholm 2012). Applied to trained rows only; fallback rows stay uniform."""
    purify: bool = False
    """The blueprint plays its argmax. Overrides ``policy_threshold``."""
    decompose: bool = False
    """Attribute the responder's gain to the public nodes it comes from. Does
    not change the number; costs one extra reach vector per walk."""


@dataclass(frozen=True)
class SeatResult:
    """Best-response value for one (responder seat, button) configuration.

    ``value_mbb`` is the BR's value against the blueprint; ``self_play_mbb`` the
    blueprint's own value in that seat; their difference ``gain_mbb`` is what
    the decomposition attributes node by node. Zero-sum: the two seats' self-play
    values at one button cancel, so the mean of the four BR values is also the
    mean of the four gains.
    """

    br_seat: int
    button: int
    value_mbb: float
    missing_policy_mass: float
    self_play_mbb: float
    gain_mbb: float


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
    decomposition: dict[str, Any] | None = None


@dataclass
class _NodeGain:
    """Per-walk accumulator for one public node (street, betting sequence).

    Arrays are indexed by the node's legal-action position. ``mass`` is the
    self-play reach mass (own reach x compatible opponent mass), the weight
    behind both mixes; ``gain`` is the sum of this node's deviation terms.
    """

    tokens: tuple[str, ...]
    types: tuple[str, ...]
    line: str
    gain: float = 0.0
    mass: float = 0.0
    blueprint_mix: np.ndarray = field(default_factory=lambda: np.zeros(0))
    br_mix: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gain_by_br_action: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def absorb(self, other: _NodeGain) -> None:
        self.gain += other.gain
        self.mass += other.mass
        self.blueprint_mix += other.blueprint_mix
        self.br_mix += other.br_mix
        self.gain_by_br_action += other.gain_by_br_action


@dataclass
class _Tally:
    """Everything a walk counts besides its values, summed from zero per subtree.

    A flop job carries one back and the join absorbs it in visit order, so a
    worker's part and an in-process part are the same arithmetic and the
    masses round identically at any worker count. ``terms`` is the
    decomposition's running sum; ``gains`` its per-node entries.
    """

    nodes: int = 0
    decision_mass: float = 0.0
    missing_mass: float = 0.0
    selfplay_mass: float = 0.0
    selfplay_missing: float = 0.0
    terms: float = 0.0
    gains: dict[tuple[Street, str], _NodeGain] = field(default_factory=dict)

    def absorb(self, other: _Tally) -> None:
        self.nodes += other.nodes
        self.decision_mass += other.decision_mass
        self.missing_mass += other.missing_mass
        self.selfplay_mass += other.selfplay_mass
        self.selfplay_missing += other.selfplay_missing
        self.terms += other.terms
        for key, entry in other.gains.items():
            mine = self.gains.get(key)
            if mine is None:
                self.gains[key] = entry
            else:
                mine.absorb(entry)


@dataclass(frozen=True)
class WalkResult:
    """One (responder seat, button) walk's raw parts, aggregated by the parent.

    Values are per dealt hand (already divided by the 1326 x 1225 measure) in
    chips. ``terms_chips`` is the decomposition's sum, carried separately so the
    identity ``chips - self_play_chips == terms_chips`` is CHECKED, never assumed.
    """

    chips: float
    self_play_chips: float
    terms_chips: float
    nodes: int
    decision_mass: float
    missing_mass: float
    selfplay_mass: float
    selfplay_missing: float
    branches: int
    gains: dict[tuple[Street, str], _NodeGain] | None


@dataclass(frozen=True)
class _FlopPart:
    """One flop subtree's values and tally, summed from zero inside the subtree."""

    best: np.ndarray
    self_play: np.ndarray
    tally: _Tally


@dataclass(frozen=True)
class _FlopJob:
    """A flop subtree waiting for a worker: the dealt state and the reaches into it.

    ``key`` is (walk, top-level deal node, flop index) in VISIT order, which is
    deterministic, so the joining pass finds its part without any other
    bookkeeping. ``line`` is the preflop sequence the subtree hangs off.
    """

    key: tuple[int, int, int]
    br_seat: int
    state: GameState
    reach: np.ndarray
    own_reach: np.ndarray
    line: str


def _conditional_factor(remaining: int, drawn: int) -> float:
    """1 / P(a ``drawn``-card draw from ``remaining`` misses the four hole cards)."""
    return math.comb(remaining, drawn) / math.comb(remaining - 4, drawn)


class _BoardPlan:
    """Deterministic sampled chance: fixed flop set, fixed per-board turn/river sets."""

    def __init__(self, config: PublicBRConfig):
        self._config = config
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        infos = list(enumerator.iterate())
        counts = np.array([info.raw_count for info in infos], dtype=np.float64)
        probs = counts / counts.sum()
        scale = _conditional_factor(52, 3) if config.conditional_chance else 1.0
        if config.num_flops >= len(infos):
            self.flops = [
                (info.representative, float(p) * scale)
                for info, p in zip(infos, probs, strict=True)
            ]
        else:
            rng = np.random.default_rng(np.random.SeedSequence([config.board_seed]))
            draws = rng.choice(len(infos), size=config.num_flops, replace=True, p=probs)
            unique, tallies = np.unique(draws, return_counts=True)
            self.flops = [
                (infos[int(i)].representative, float(n) / config.num_flops * scale)
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
        if self._config.conditional_chance:
            weight *= _conditional_factor(52 - len(board), 1)
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


TOP_NODES = 30
TOP_LINES = 30


class PublicTreeBestResponse:
    """Exact best response of one seat against the blueprint on the sampled tree.

    Every walk carries two reach vectors down -- the opponent's under the
    blueprint (with chance folded in) and the responder's OWN under the
    blueprint -- and two value vectors up: the best response's and the
    blueprint's self-play. At a responder node the gain decomposition records
    ``own_reach * (B(n) - sum_a sigma_a B_a)``: prefix reach under the blueprint,
    suffix values under the best response. Those terms telescope exactly to
    ``BR - self-play`` at the root and are each >= 0 for the per-combo
    responder, so aggregating them by street or node is attribution, not
    cancellation.
    """

    def __init__(
        self,
        blueprint: ScorableBlueprint,
        config: PublicBRConfig,
        *,
        starting_stack: int,
        blueprint_factory: Callable[[], ScorableBlueprint] | None = None,
        on_branch: Callable[[int, int], None] | None = None,
    ):
        self._on_branch = on_branch
        self._branches_done = 0
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
        self._tally = _Tally()
        # Fork-join state: a list while the preflop pass RECORDS flop subtrees,
        # a dict while it JOINS the parts workers returned, None on the serial
        # path. `_walk` and `_deal` index the jobs in visit order.
        self._fringe: list[_FlopJob] | None = None
        self._joined: dict[tuple[int, int, int], _FlopPart] | None = None
        self._walk_index = 0
        self._deal = 0
        # Last, because it needs the rules, the action model and the board plan.
        self._branches_per_walk = self._count_flop_deals()

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
            parts = self._fork_join(walks, self._factory)
        else:
            parts = [self.run_walk(br_seat, button) for br_seat, button in walks]

        for (br_seat, button), part in zip(walks, parts, strict=True):
            fraction = part.missing_mass / part.decision_mass if part.decision_mass else 0.0
            seat_results.append(
                SeatResult(
                    br_seat=br_seat,
                    button=button,
                    value_mbb=part.chips / big_blind * 1000.0,
                    missing_policy_mass=fraction,
                    self_play_mbb=part.self_play_chips / big_blind * 1000.0,
                    gain_mbb=(part.chips - part.self_play_chips) / big_blind * 1000.0,
                )
            )
            total_nodes += part.nodes
            total_decision += part.decision_mass
            total_missing += part.missing_mass
        exploitability = float(np.mean([r.value_mbb for r in seat_results]))
        decomposition = (
            _summarise_decomposition(walks, parts, seat_results, big_blind)
            if self._config.decompose
            else None
        )
        return PublicBRResult(
            exploitability_mbb=exploitability,
            seat_results=tuple(seat_results),
            missing_policy_mass=total_missing / total_decision if total_decision else 0.0,
            nodes_visited=total_nodes,
            num_flops=len(self._plan.flops),
            elapsed_s=time.perf_counter() - start,
            config=self._config,
            decomposition=decomposition,
        )

    def walk_values(
        self, br_seat: int, button: int, opp_reach: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Responder's per-combo (best-response, self-play) chip values vs ``opp_reach``.

        ``values[h]`` sums utility over every opponent combo weighted by
        ``opp_reach`` and every sampled board branch; dividing by the
        compatible opponent mass turns it into a per-hand expectation. The
        second vector is the same sum with the responder following the
        blueprint too. Exposed for validation against the scalar reference
        game; :meth:`evaluate` is the standard entry point.
        """
        self._br_seat = br_seat
        self._tally = _Tally()
        root = self._rules.create_initial_state(
            starting_stack=self._starting_stack, hole_cards=self._dummy_holes, button=button
        )
        own_reach = np.ones(NUM_COMBOS, dtype=np.float64)
        return self._walk(root, opp_reach.astype(np.float64), own_reach, "")

    def responder_values(self, br_seat: int, button: int, opp_reach: np.ndarray) -> np.ndarray:
        """Best-response half of :meth:`walk_values`."""
        return self.walk_values(br_seat, button, opp_reach)[0]

    def run_walk(self, br_seat: int, button: int) -> WalkResult:
        """One walk's values plus the telemetry the aggregate needs.

        Returning the raw parts keeps the value/mass arithmetic in ONE place --
        the serial and fork-join paths then cannot drift.
        """
        self._deal = 0
        before = self._branches_done
        best, self_play = self.walk_values(br_seat, button, np.ones(NUM_COMBOS, dtype=np.float64))
        measure = NUM_COMBOS * _NUM_OPP_DEALS
        tally = self._tally
        return WalkResult(
            chips=float(best.sum()) / measure,
            self_play_chips=float(self_play.sum()) / measure,
            terms_chips=tally.terms / measure,
            nodes=tally.nodes,
            decision_mass=tally.decision_mass,
            missing_mass=tally.missing_mass,
            selfplay_mass=tally.selfplay_mass,
            selfplay_missing=tally.selfplay_missing,
            branches=self._branches_done - before,
            gains=tally.gains if self._config.decompose else None,
        )

    def flop_part(self, job: _FlopJob) -> _FlopPart:
        """One flop subtree from fresh counters -- what a worker process runs."""
        self._br_seat = job.br_seat
        return self._flop_part(job.state, job.reach, job.own_reach, job.line)

    def _flop_part(
        self, state: GameState, reach: np.ndarray, own_reach: np.ndarray, line: str
    ) -> _FlopPart:
        saved, self._tally = self._tally, _Tally()
        best, self_play = self._walk(state, reach, own_reach, line)
        part = _FlopPart(best, self_play, self._tally)
        self._tally = saved
        return part

    def _fork_join(
        self, walks: list[tuple[int, int]], factory: Callable[[], ScorableBlueprint]
    ) -> list[WalkResult]:
        """The four walks with every flop subtree farmed out to a process pool.

        Three passes over the PREFLOP tree, which is cheap: record every flop
        deal that has reach (the downward reach never depends on the values
        coming back up), run all of them on the pool at once -- that is what
        fills sixteen cores when there are only four sampled flops -- then walk
        again joining the parts in the serial order. The bar counts jobs as
        they land, in the same branch unit as the serial path.
        """
        publish, self._on_branch = self._on_branch, None
        try:
            self._fringe = []
            self._branches_done = 0
            for walk, (br_seat, button) in enumerate(walks):
                self._walk_index = walk
                self.run_walk(br_seat, button)
            jobs, self._fringe = self._fringe, None
            # Branches that need no job: no reach, or under an action the
            # blueprint never takes. The bar starts from them.
            done = self._branches_done - len(jobs)
            self._joined = {}
            if jobs:
                workers = _ram_safe_workers(min(self._config.num_workers, len(jobs)), self._config)
                self._joined, peak, anon = _run_jobs(
                    jobs,
                    workers=workers,
                    init=(factory, self._config, self._starting_stack),
                    publish=publish,
                    done=done,
                    total=self.branch_total,
                )
                logger.info(
                    "public-tree BR fork-join: %d flop jobs over %d workers, "
                    "peak worker RSS %.2f GB of which private %.2f GB",
                    len(jobs),
                    workers,
                    peak / 2**30,
                    anon / 2**30,
                )
            self._branches_done = 0
            parts = []
            for walk, (br_seat, button) in enumerate(walks):
                self._walk_index = walk
                parts.append(self.run_walk(br_seat, button))
            self._joined = None
            return parts
        finally:
            self._on_branch = publish

    def _walk(
        self, state: GameState, opp_reach: np.ndarray, own_reach: np.ndarray, line: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """(best-response, self-play) per-combo values below ``state``.

        ``line`` is the preflop betting line a postflop node descends from, for
        attribution only. The two returned arrays may be one object at a
        terminal; callers that modify in place apply the same mask to both.
        """
        self._tally.nodes += 1
        if not opp_reach.any():
            return np.zeros(NUM_COMBOS, dtype=np.float64), np.zeros(NUM_COMBOS, dtype=np.float64)
        if state.is_terminal:
            values = self._terminal_values(state, opp_reach)
            return values, values
        if len(state.board) < state.street.board_card_count:
            return self._deal_values(state, opp_reach, own_reach, line)
        legal = self._rules.get_legal_actions(state, self._action_model)
        if state.current_player == self._br_seat:
            return self._responder_values(state, legal, opp_reach, own_reach, line)
        return self._opponent_values(state, legal, opp_reach, own_reach, line)

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

    def _deal_values(
        self, state: GameState, opp_reach: np.ndarray, own_reach: np.ndarray, line: str
    ) -> tuple[np.ndarray, np.ndarray]:
        best = np.zeros(NUM_COMBOS, dtype=np.float64)
        self_play = np.zeros(NUM_COMBOS, dtype=np.float64)
        # The FLOP deal only -- an empty board. It is the outermost branching in
        # the whole walk, so the counter fires `num_flops` times per walk rather
        # than once per node, and costs nothing measurable. Turn and river deals
        # reach this same method and are deliberately not counted: they are the
        # inner loops, and a counter there WOULD be in the hot path.
        # Counted whether or not anyone is listening. An evaluation with no bar
        # is exactly the one that MEASURES what a walk costs for the next one,
        # and while this was tied to `on_branch` it measured zero and taught the
        # next evaluation nothing.
        top_level = not state.board
        deal = self._deal
        if top_level:
            self._deal += 1
            line = state.normalized_betting_sequence()
        for flop, (cards, weight) in enumerate(self._plan.deal_options(state.board)):
            block = blocked_combos(cards)
            child_reach = np.where(block, 0.0, opp_reach) * weight
            # An `if` rather than the `continue` this replaced: a branch with no
            # reach is still a branch DONE, and skipping the count would leave the
            # bar permanently short of its own total.
            if child_reach.any():
                child_state = begin_street(state, cards)
                if top_level:
                    key = (self._walk_index, deal, flop)
                    if self._fringe is not None:
                        self._fringe.append(
                            _FlopJob(key, self._br_seat, child_state, child_reach, own_reach, line)
                        )
                        part = None
                    elif self._joined is not None:
                        part = self._joined[key]
                    else:
                        part = self._flop_part(child_state, child_reach, own_reach, line)
                    if part is not None:
                        self._tally.absorb(part.tally)
                        child_best, child_self = part.best, part.self_play
                        child_best[block] = 0.0
                        child_self[block] = 0.0
                        best += child_best
                        self_play += child_self
                else:
                    child_best, child_self = self._walk(child_state, child_reach, own_reach, line)
                    child_best[block] = 0.0
                    child_self[block] = 0.0
                    best += child_best
                    self_play += child_self
            if top_level:
                self._branches_done += 1
                # Silent until the denominator is known -- measured by a walk of
                # this tree, here or in an earlier evaluation. Publishing against
                # a total of zero would be a bar with no denominator, and against
                # a growing one a bar that moves backwards.
                if self._on_branch is not None and self._branches_per_walk:
                    self._on_branch(self._branches_done, self.branch_total)
        return best, self_play

    def _flop_deals_below(self, state: GameState) -> int:
        """Top-level flop branches in the tree below ``state``, walked structurally.

        NOT `4 * num_flops`. The flop deal is reached once per preflop betting
        line that survives to a flop, so the count per walk is a property of the
        betting tree -- but it is a property this can READ, by recursing over
        the same legal actions the walk does with no cards, no combos and no
        policy. Counted UP FRONT, which is the only way the bar has a
        denominator before a walk has finished: under `--workers` the four walks
        that could measure it all finish together, so a bar that waits for one
        is a bar that never draws.

        Cheap because the board is empty throughout: it recurses over the
        PREFLOP betting tree only and stops at the deal.
        """
        if state.board or state.is_terminal:
            return 0
        if len(state.board) < state.street.board_card_count:
            return len(self._plan.deal_options(()))
        return sum(
            self._flop_deals_below(self._rules.apply_action(state, action))
            for action in self._rules.get_legal_actions(state, self._action_model)
        )

    def _count_flop_deals(self) -> int:
        """What one walk costs, from the root. Identical across the four walks:
        they traverse the same public tree and differ only in whose values are
        being maximised."""
        root = self._rules.create_initial_state(
            starting_stack=self._starting_stack, hole_cards=self._dummy_holes, button=0
        )
        return self._flop_deals_below(root)

    @property
    def branch_total(self) -> int:
        """Top-level flop branches across the whole evaluation."""
        return 4 * self._branches_per_walk

    def _opponent_values(
        self,
        state: GameState,
        legal: tuple[Action, ...],
        opp_reach: np.ndarray,
        own_reach: np.ndarray,
        line: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma, missing = self._policy_matrix(state, legal)
        tally = self._tally
        tally.decision_mass += float(opp_reach.sum())
        tally.missing_mass += float(opp_reach[missing].sum())
        if self._config.decompose:
            # The self-play view of the same fallback: what fraction of the
            # opponent decisions the BLUEPRINT actually brings the responder to
            # land on a uniform row. The all-branches figure above counts
            # subtrees the responder explores and the blueprint never enters.
            alive = self._alive(state)
            tally.selfplay_mass += float((own_reach * nonblocking_mass(opp_reach))[alive].sum())
            if missing.any():
                fallback = own_reach * nonblocking_mass(np.where(missing, opp_reach, 0.0))
                tally.selfplay_missing += float(fallback[alive].sum())
        best = np.zeros(NUM_COMBOS, dtype=np.float64)
        self_play = np.zeros(NUM_COMBOS, dtype=np.float64)
        for a_idx, action in enumerate(legal):
            child = self._rules.apply_action(state, action)
            child_reach = opp_reach * sigma[:, a_idx]
            if not child_reach.any():
                # An action the blueprint never takes with ANY combo. The flop
                # branches under it are work that will not happen, so they are
                # DONE -- otherwise the denominator, which counts the tree and
                # cannot know the policy, is one the bar can never reach.
                self._skip_branches(self._flop_deals_below(child))
                continue
            child_best, child_self = self._walk(child, child_reach, own_reach, line)
            best += child_best
            self_play += child_self
        return best, self_play

    def _responder_values(
        self,
        state: GameState,
        legal: tuple[Action, ...],
        opp_reach: np.ndarray,
        own_reach: np.ndarray,
        line: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Max over actions for the responder; the blueprint's mix for self-play.

        Every action is walked whatever the blueprint's own probability: the
        best response does not depend on the responder's reach, only its
        attribution does.
        """
        sigma, _ = self._policy_matrix(state, legal)
        best_children = np.empty((NUM_COMBOS, len(legal)), dtype=np.float64)
        self_children = np.empty((NUM_COMBOS, len(legal)), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            child = self._rules.apply_action(state, action)
            child_best, child_self = self._walk(child, opp_reach, own_reach * sigma[:, a_idx], line)
            best_children[:, a_idx] = child_best
            self_children[:, a_idx] = child_self
        if self._config.in_abstraction:
            choice = self._bucket_choice(state, best_children)
        else:
            choice = best_children.argmax(axis=1)
        best = np.take_along_axis(best_children, choice[:, None], axis=1)[:, 0]
        self_play = np.einsum("ij,ij->i", sigma, self_children)
        if self._config.decompose:
            self._record_gain(
                state, legal, line, opp_reach, own_reach, sigma, best_children, choice, best
            )
        return best, self_play

    def _bucket_choice(self, state: GameState, best_children: np.ndarray) -> np.ndarray:
        """One action per bucket: argmax of the bucket-summed counterfactual values.

        The values already aggregate every sampled runout below this node, and
        a bucket here is a function of the cards face up at this node, so the
        maximisation is joint over exactly the futures the responder cannot
        yet distinguish -- being told your bucket is not being told the river.
        Blocked combos (bucket -1) are excluded from the sums; their values are
        zeroed at the deal above on the way up regardless.
        """
        buckets = self._responder_buckets(state.board, state.street)
        alive = buckets >= 0
        live = buckets[alive]
        count = int(live.max()) + 1
        totals = np.stack(
            [
                np.bincount(live, weights=best_children[alive, a], minlength=count)
                for a in range(best_children.shape[1])
            ],
            axis=1,
        )
        choice = np.zeros(NUM_COMBOS, dtype=np.int64)
        choice[alive] = totals.argmax(axis=1)[live]
        return choice

    def _responder_buckets(self, board: tuple[Card, ...], street: Street) -> np.ndarray:
        """What the constrained responder is allowed to tell apart: its own bucket.

        A seam distinct from :meth:`_bucket_vector` so a test can give the
        responder a coarser partition while the OPPONENT's policy lookup keeps
        the real one.
        """
        return self._bucket_vector(board, street)

    def _record_gain(
        self,
        state: GameState,
        legal: tuple[Action, ...],
        line: str,
        opp_reach: np.ndarray,
        own_reach: np.ndarray,
        sigma: np.ndarray,
        best_children: np.ndarray,
        choice: np.ndarray,
        best: np.ndarray,
    ) -> None:
        """One node's deviation terms, masked to combos alive on this board.

        A combo blocked by the board still carries a (meaningless) value inside
        the subtree -- it is zeroed at the deal on the way up -- so its term
        must not enter the sum here, where nothing above will cancel it.
        """
        alive = self._alive(state)
        terms = own_reach * (best - np.einsum("ij,ij->i", sigma, best_children))
        terms[~alive] = 0.0
        weight = own_reach * nonblocking_mass(opp_reach)
        weight[~alive] = 0.0
        key = (state.street, state.normalized_betting_sequence())
        gains = self._tally.gains
        entry = gains.get(key)
        if entry is None:
            width = len(legal)
            entry = gains[key] = _NodeGain(
                tokens=tuple(action.normalize(state.pot) for action in legal),
                types=tuple(action.type.name for action in legal),
                line=key[1] if state.street == Street.PREFLOP else line,
                blueprint_mix=np.zeros(width),
                br_mix=np.zeros(width),
                gain_by_br_action=np.zeros(width),
            )
        total = float(terms.sum())
        self._tally.terms += total
        entry.gain += total
        entry.mass += float(weight.sum())
        entry.blueprint_mix += weight @ sigma
        entry.br_mix += np.bincount(choice, weights=weight, minlength=len(legal))
        entry.gain_by_br_action += np.bincount(choice, weights=terms, minlength=len(legal))

    def _alive(self, state: GameState) -> np.ndarray:
        """Combos not sharing a card with the board at ``state``."""
        return self._bucket_vector(state.board, state.street) >= 0

    def _skip_branches(self, branches: int) -> None:
        if branches <= 0:
            return
        self._branches_done += branches
        if self._on_branch is not None:
            self._on_branch(self._branches_done, self.branch_total)

    def _policy_matrix(
        self, state: GameState, legal: tuple[Action, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-combo blueprint distribution over ``legal`` plus uniform-fallback mask.

        The per-bucket rows depend only on the betting context, never the board,
        so they are built densely over all buckets once per context and every
        node with that context (one per sampled board) reduces to a vectorized
        gather through the board's bucket vector. The eval-time transform
        (threshold / purify) is applied to the dense rows, so it costs nothing
        per node.
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
            rows, row_missing = blueprint_policy_table(
                self._policy_source.rows_at(state), state, self._rules, legal
            )
            cached = (transform_policy_rows(rows, row_missing, self._config), row_missing)
            self._policy_cache[context_key] = cached
        rows, row_missing = cached
        bucket_vec = self._bucket_vector(state.board, state.street)
        alive = bucket_vec >= 0
        sigma = np.zeros((NUM_COMBOS, len(legal)), dtype=np.float64)
        sigma[alive] = rows[bucket_vec[alive]]
        missing = np.zeros(NUM_COMBOS, dtype=bool)
        missing[alive] = row_missing[bucket_vec[alive]]
        return sigma, missing

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


def transform_policy_rows(
    rows: np.ndarray, missing: np.ndarray, config: PublicBRConfig
) -> np.ndarray:
    """The strategy under measurement after the eval-time transform, row by row.

    Trained rows only: a fallback row is uniform because nothing was learned
    there, and purifying it would field "always the first action" (fold) in
    place of the blueprint's real behaviour. A thresholded row that zeroes out
    entirely (every action below the cut) falls back to its argmax rather than
    to NaN.
    """
    if not config.purify and config.policy_threshold <= 0.0:
        return rows
    out = rows.copy()
    present = ~missing
    if config.purify:
        one_hot = np.zeros_like(out[present])
        one_hot[np.arange(one_hot.shape[0]), out[present].argmax(axis=1)] = 1.0
        out[present] = one_hot
        return out
    cut = out[present]
    cut[cut < config.policy_threshold] = 0.0
    totals = cut.sum(axis=1)
    empty = totals <= 0.0
    if empty.any():
        cut[empty] = 0.0
        cut[np.nonzero(empty)[0], rows[present][empty].argmax(axis=1)] = 1.0
        totals[empty] = 1.0
    out[present] = cut / totals[:, None]
    return out


def _summarise_decomposition(
    walks: list[tuple[int, int]],
    parts: list[WalkResult],
    seats: list[SeatResult],
    big_blind: float,
) -> dict[str, Any]:
    """The gain attribution, in mbb per hand, as it is recorded.

    Aggregates are MEANS over the four walks, so ``by_street`` sums to
    ``exploitability_mbb``; a node or line entry carries its walk-level
    ``gain_mbb`` (what that responder takes there) and ``headline_mbb``, its
    quarter share of the reported number. ``identity`` is the check that the
    terms sum to ``gain_mbb`` walk by walk -- the one number that says the
    attribution is exact rather than approximate.
    """
    per_hand = 1000.0 / (NUM_COMBOS * _NUM_OPP_DEALS * big_blind)
    share = 1.0 / len(walks)
    streets = [str(street) for street in Street]
    by_street = dict.fromkeys(streets, 0.0)
    by_line: dict[str, dict[str, Any]] = {}
    by_type: dict[str, dict[str, float]] = {street: {} for street in streets}
    nodes: list[dict[str, Any]] = []
    per_walk: list[dict[str, Any]] = []
    selfplay_mass = sum(part.selfplay_mass for part in parts)
    selfplay_missing = sum(part.selfplay_missing for part in parts)
    for (br_seat, button), part, seat in zip(walks, parts, seats, strict=True):
        walk_streets = dict.fromkeys(streets, 0.0)
        for (street, sequence), entry in (part.gains or {}).items():
            name = str(street)
            gain = entry.gain * per_hand
            walk_streets[name] += gain
            line = by_line.setdefault(
                entry.line,
                {"line": entry.line, "headline_mbb": 0.0, "by_street": dict.fromkeys(streets, 0.0)},
            )
            line["headline_mbb"] += gain * share
            line["by_street"][name] += gain * share
            for kind, amount in zip(entry.types, entry.gain_by_br_action, strict=True):
                by_type[name][kind] = (
                    by_type[name].get(kind, 0.0) + float(amount) * per_hand * share
                )
            mass = entry.mass
            nodes.append(
                {
                    "br_seat": br_seat,
                    "button": button,
                    "street": name,
                    "sequence": sequence,
                    "line": entry.line,
                    "gain_mbb": gain,
                    "headline_mbb": gain * share,
                    "reach": mass / (NUM_COMBOS * _NUM_OPP_DEALS),
                    "blueprint": {
                        token: float(p / mass) if mass else 0.0
                        for token, p in zip(entry.tokens, entry.blueprint_mix, strict=True)
                    },
                    "best_response": {
                        token: float(p / mass) if mass else 0.0
                        for token, p in zip(entry.tokens, entry.br_mix, strict=True)
                    },
                    "gain_by_br_action_mbb": {
                        token: float(g) * per_hand
                        for token, g in zip(entry.tokens, entry.gain_by_br_action, strict=True)
                    },
                }
            )
        for name in streets:
            by_street[name] += walk_streets[name] * share
        terms_mbb = part.terms_chips / big_blind * 1000.0
        per_walk.append(
            {
                "br_seat": br_seat,
                "button": button,
                "br_value_mbb": seat.value_mbb,
                "self_play_mbb": seat.self_play_mbb,
                "gain_mbb": seat.gain_mbb,
                "terms_mbb": terms_mbb,
                "identity_gap_mbb": seat.gain_mbb - terms_mbb,
                "by_street": walk_streets,
            }
        )
    nodes.sort(key=lambda node: -node["gain_mbb"])
    lines = sorted(by_line.values(), key=lambda item: -item["headline_mbb"])

    def mean_over(predicate: Callable[[tuple[int, int]], bool]) -> float:
        picked = [seat.gain_mbb for walk, seat in zip(walks, seats, strict=True) if predicate(walk)]
        return float(np.mean(picked)) if picked else 0.0

    return {
        "form": (
            "sum over responder nodes of own_reach(blueprint) * "
            "[value(best response) - sum_a sigma(a) value_a(best response)]"
        ),
        "identity": {
            "max_abs_gap_mbb": max(abs(walk["identity_gap_mbb"]) for walk in per_walk),
            "per_walk": per_walk,
        },
        "by_street": by_street,
        "by_seat": {str(seat): mean_over(lambda w, s=seat: w[0] == s) for seat in (0, 1)},
        "by_button": {str(b): mean_over(lambda w, b=b: w[1] == b) for b in (0, 1)},
        "by_position": {
            "button": mean_over(lambda w: w[0] == w[1]),
            "big_blind": mean_over(lambda w: w[0] != w[1]),
        },
        "by_br_action_type": by_type,
        "by_preflop_line": lines[:TOP_LINES],
        "top_nodes": nodes[:TOP_NODES],
        "nodes_with_gain": sum(1 for node in nodes if node["gain_mbb"] > 0.0),
        "responder_nodes": len(nodes),
        "selfplay_missing_policy_mass": selfplay_missing / selfplay_mass if selfplay_mass else 0.0,
    }


_ENGINE: PublicTreeBestResponse | None = None


def _init_worker(
    factory: Callable[[], ScorableBlueprint], config: PublicBRConfig, starting_stack: int
) -> None:
    """Build the engine ONCE per process: the blueprint load is seconds, a job
    can be milliseconds, and the per-context policy tables amortise across
    every job the process runs. Rebuilt through the factory rather than
    received, since the solver is not picklable. Spawned, so logging config
    does not inherit."""
    global _ENGINE
    configure_logging()
    _ENGINE = PublicTreeBestResponse(factory(), config, starting_stack=starting_stack)


def _flop_worker(job: _FlopJob) -> tuple[_FlopPart, int, int]:
    assert _ENGINE is not None, "worker used before _init_worker"
    part = _ENGINE.flop_part(job)
    return part, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, _rss_anon_kb()


def _proc_kb(path: str, field: str) -> int:
    """One ``field: N kB`` line of a /proc file, Linux only; 0 elsewhere."""
    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except OSError:
        return 0
    return next((int(line.split()[1]) for line in lines if line.startswith(field)), 0)


def _rss_anon_kb() -> int:
    """Private (anonymous) resident memory of this process.

    ``ru_maxrss`` also counts file-backed pages every worker shares through
    the page cache, so it overstates what sixteen workers cost the node. The
    checkpoint arrays a scoring blueprint holds are ``np.zeros`` (process
    local), so they ARE here, and so is the card abstraction.
    """
    return _proc_kb("/proc/self/status", "RssAnon:")


# A worker's peak over its steady footprint. `load_checkpoint` reads each
# zarr array into a private copy before writing it into the storage's array,
# and the walk's own tables grow on top: measured 4.13 GB peak against a
# 1.67 GB parent on the production tree and 5.4 GB against 2.25 GB on the
# 45.5M-row limp-fix tree, so 2.4.
_LOAD_PEAK = 2.4
# Per sampled board a worker caches a bucket vector and, on the river, a
# RunoutEvaluator: ~a dozen int64 arrays over the ~1081 alive combos.
_PER_BOARD_BYTES = 120 * 1024
# Left to the OS and the page cache the checkpoint is read through.
_HEADROOM_BYTES = 2 * 2**30


def _ram_safe_workers(
    requested: int,
    config: PublicBRConfig,
    *,
    footprint: int | None = None,
    available: int | None = None,
) -> int:
    """``requested`` lowered to what free RAM holds; ``--workers`` is a ceiling.

    Every worker rebuilds the blueprint, so this process's private RSS -- it
    holds the same blueprint and has walked only the preflop tree -- is the
    steady cost of one worker, scaled for the load peak and the board caches
    the tier will fill. Measured here rather than tabulated so a bigger tree
    sizes itself: 16 workers on a 32 GB node died of the OOM killer at 4/8/8
    on the production tree and at 4/4/4 on one 1.4x its size.
    """
    footprint = _rss_anon_kb() * 1024 if footprint is None else footprint
    available = (
        _proc_kb("/proc/meminfo", "MemAvailable:") * 1024 if available is None else available
    )
    if footprint <= 0 or available <= 0:
        return requested
    flops = min(config.num_flops, 1755)
    turns = min(config.num_turns, 49)
    rivers = min(config.num_rivers, 48)
    boards = flops * (turns + turns * rivers)
    per_worker = footprint * _LOAD_PEAK + boards * _PER_BOARD_BYTES
    fits = max(1, int((available - _HEADROOM_BYTES) // per_worker))
    workers = min(requested, fits)
    logger.info(
        "public-tree BR fork-join: %d workers of %d requested -- %.2f GB private per worker "
        "(x%.1f load peak, +%.2f GB caches for %d boards) against %.2f GB available",
        workers,
        requested,
        footprint / 2**30,
        _LOAD_PEAK,
        boards * _PER_BOARD_BYTES / 2**30,
        boards,
        available / 2**30,
    )
    return workers


def _run_jobs(
    jobs: list[_FlopJob],
    *,
    workers: int,
    init: tuple[Callable[[], ScorableBlueprint], PublicBRConfig, int],
    publish: Callable[[int, int], None] | None,
    done: int,
    total: int,
) -> tuple[dict[tuple[int, int, int], _FlopPart], int, int]:
    """Every flop job on a pool, parts keyed for the join; peak worker RSS and
    private RSS in bytes.

    Submitted all at once and collected as they land, so the bar moves from the
    first finished subtree. ``ru_maxrss`` is kilobytes on Linux and bytes on
    macOS; the pool runs on Linux and the figures size the NEXT tier's worker
    count, which is the only reason they are collected.
    """
    parts: dict[tuple[int, int, int], _FlopPart] = {}
    peak = anon = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=init) as pool:
        futures = {pool.submit(_flop_worker, job): job for job in jobs}
        for future in as_completed(futures):
            part, rss, rss_anon = future.result()
            parts[futures[future].key] = part
            peak = max(peak, rss)
            anon = max(anon, rss_anon)
            done += 1
            if publish is not None:
                publish(done, total)
    return parts, peak * 1024, anon * 1024


def compute_public_tree_br(
    blueprint: ScorableBlueprint,
    config: PublicBRConfig,
    *,
    starting_stack: int,
    blueprint_factory: Callable[[], ScorableBlueprint] | None = None,
    on_branch: Callable[[int, int], None] | None = None,
) -> PublicBRResult:
    """Exact best response against ``blueprint`` on the sampled public tree.

    ``blueprint_factory`` enables ``config.num_workers > 1``: workers rebuild the
    blueprint rather than receiving it, since the solver is not picklable.

    ``on_branch`` is called with (done, total) as each top-level flop branch
    finishes -- the only thing here that is both cheap to count and numerous. See
    :meth:`PublicTreeBestResponse.evaluate`.
    """
    engine = PublicTreeBestResponse(
        blueprint,
        config,
        starting_stack=starting_stack,
        blueprint_factory=blueprint_factory,
        on_branch=on_branch,
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
