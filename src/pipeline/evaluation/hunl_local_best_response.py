"""Local Best Response (LBR) for the trained HUNL blueprint.

Exact best response is intractable for full HUNL, so we measure a trustworthy
*lower bound* on exploitability with Local Best Response (Lisý & Bowling 2017).
The LBR player plays a concrete strategy against the frozen blueprint: at each of
its decisions it picks, over a menu of actions, the one maximizing a cheap
*myopic* value that assumes it then checks/calls to showdown. Because that is a
realizable strategy, its **realized** value against the blueprint is a lower
bound on the blueprint's true exploitability — the same property validated
exactly on Kuhn/Leduc in :mod:`local_best_response`.

The LBR player re-decides at *every* node against the full, Bayes-updated
opponent range (not just its first action, as the one-ply rollout in
:mod:`exploitability` does), which already makes it a far more informative
lower bound.

Off-tree bets (opt-in, ``include_off_tree``)
--------------------------------------------
LBR's menu can also include **off-tree bet and raise sizes** the blueprint never
trained on; a real opponent using unseen sizes is what exposes the action
abstraction. Infoset keys encode the *full-hand* betting sequence (see
:func:`~src.core.game.state.GameState.normalized_betting_sequence` — history is
cumulative across streets), so a naively-played off-tree amount would normalize
to an unseen pot fraction (``b0.63``) and poison every later opponent lookup of
the hand (miss → uniform-random play → invalid upward bias). To keep the number
rigorous the evaluator carries a persistent on-tree **shadow state**
(:mod:`~src.pipeline.evaluation.shadow_state`) alongside the real one: each
off-tree exploiter size is committed to an on-tree proxy on the shadow (sampled
once from the pseudo-harmonic translation weights), every opponent lookup keys
off the shadow, and the opponent's shadow decision is realized in the real game
by a pot-fraction-preserving map-back. The real state stays authoritative for
chips, legality, terminality, and payoffs.

Sizes with no structure-preserving shadow proxy are dropped from the menu,
which only restricts the exploiter — the bound loosens but never breaks. With
the shadow in place a uniform fallback can only mean a genuinely untrained
infoset, exactly as in on-tree evals. ``include_off_tree=False`` (the default,
kept for comparability with recorded baselines) restricts the menu to on-tree
actions; the shadow then never diverges and the result is bitwise-identical to
the pre-shadow evaluator.

Opponent model / what this number means
----------------------------------------
Exploitability is defined against a *complete* strategy, but training only
defines the blueprint on-tree. This evaluator completes it with the standard
**translation policy** applied to the whole history via the carried shadow:
the blueprint responds as if every off-tree size had been its committed
on-tree proxy. Sampling the proxy makes the completion *behavioral* — a
mixture over deterministic translations — which is still a well-defined
strategy, so the realized value against it is a valid lower bound on that
completed strategy's exploitability. The blueprint is normally deployed
through the runtime resolver (``resolver.enabled`` defaults to ``True``),
whose off-tree response is a re-solve rather than a translation; measure that
system directly with ``opponent="deployed"``, where the resolver receives the
real states natively and the shadow only feeds the exploiter's scorer.

Correctness note: the myopic pot-arithmetic value is a *scoring function* used
only to choose the LBR action. The returned mbb/g figure is the **realized**
payoff of playing those chosen actions out against the blueprint to true
terminals (including the blueprint's own future bets). Approximations in the
scorer only loosen the bound; they never invalidate it.

References:
- Lisý & Bowling, "Equilibrium Approximation Quality of Current No-Limit Poker
  Bots" (2017).
"""

from __future__ import annotations

import multiprocessing
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from tqdm import tqdm

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType
from src.core.game.evaluator import get_evaluator
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Card, GameState, Street
from src.engine.search.range_inference import ALL_COMBOS, COMBO_MASKS, NUM_COMBOS
from src.engine.solver.protocols import Blueprint
from src.pipeline.evaluation.lookahead_scorer import BlueprintDistMemo, LookaheadScorer
from src.pipeline.evaluation.opponent_model import (
    BlueprintOpponent,
    OpponentModel,
    ResolvedOpponent,
    known_mask,
)
from src.pipeline.evaluation.shadow_state import MenuCandidate, ShadowTracker
from src.shared.config import ResolverConfig
from src.shared.numeric import NORMALIZE_EPS
from src.shared.units import chips_to_bb, chips_to_mbb

_DECK_MASKS: np.ndarray = np.array([card.mask for card in FULL_DECK], dtype=np.int64)

# Off-tree bet sizes (as a fraction of the pot) the LBR player may bet when it is
# first to put money in on a street. These are deliberately not the blueprint's
# trained sizes; overbets in particular probe the action abstraction.
DEFAULT_OFF_TREE_POT_FRACTIONS: tuple[float, ...] = (0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0)


@dataclass
class LBRConfig:
    """Settings for the HUNL LBR evaluator."""

    num_hands: int = 1000
    equity_runouts: int = 12
    off_tree_pot_fractions: tuple[float, ...] = DEFAULT_OFF_TREE_POT_FRACTIONS
    # Add off-tree bet/raise sizes to the exploiter's menu. Rigorous: opponent
    # lookups go through a persistent on-tree shadow state (see module docs), so
    # off-tree amounts never leak into infoset keys. Off by default only for
    # comparability with recorded baselines — it changes the measured completion,
    # so never mix on/off numbers in one comparison.
    include_off_tree: bool = False
    seed: int | None = None
    # Parallel workers for hand evaluation. LBR is embarrassingly parallel over hands;
    # each hand is seeded independently so the result is identical for any worker count.
    num_workers: int = 1
    # Board runouts averaged at all-in showdown terminals (board incomplete). All-in
    # pots are the largest payoffs in the game, so valuing them on a single sampled
    # runout was the dominant remaining variance source; averaging is a pure
    # Rao-Blackwellization (same expectation, lower variance). When exactly one card
    # is missing the runout is enumerated instead.
    allin_runouts: int = 50
    # WHICH strategy the LBR player exploits on the realized path. "blueprint" is
    # the raw table (historical numbers); "deployed" routes the opponent's actual
    # decisions through the runtime subgame resolver — the system that really
    # plays. The exploiter's myopic scorer stays blueprint-backed either way
    # (selection-only: approximations there loosen the bound, never invalidate it).
    opponent: str = "blueprint"
    # Resolver settings for opponent="deployed". Must set max_iterations (wall-
    # clock budgets make the measured strategy machine-dependent). None with
    # opponent="deployed" raises at engine construction.
    resolver: ResolverConfig | None = None
    # HOW the exploiter selects its actions. "myopic" is the classic one-step
    # check/call-to-showdown arithmetic; "lookahead" scores candidates by a
    # depth-limited best-response walk against the blueprint policy (see
    # :mod:`lookahead_scorer`). Selection-only: any scorer keeps the bound valid.
    scorer: str = "myopic"
    # Opponent-response levels the lookahead expands (depth - 1 exploiter
    # re-decisions; depth 1 ~= branch-resolved myopic). Ignored under "myopic".
    lookahead_depth: int = 2
    # Myopic prefilter width: lookahead-rescore only the top-k myopic candidates
    # (the myopic argmax is always included). <= 0 rescores the whole menu.
    lookahead_top_k: int = 3


@dataclass(frozen=True)
class HandOutcome:
    """Per-position outcome of one deal: realized value plus variance-attribution tags.

    ``terminal`` is ``"fold"`` (hand ended on a fold), ``"allin"`` (showdown reached
    with the board incomplete — an early all-in), or ``"showdown"`` (river showdown).
    When the hand ends in a branched final decision (see ``play_hand``), the label
    and ``pot`` come from the highest-variance branch with positive probability
    (allin > showdown > fold), since that branch dominates the payoff spread.
    ``pot`` is in chips. Both exist so eval variance can be decomposed by terminal
    type offline without re-running hands.
    """

    value: float
    terminal: str
    pot: int


# Severity order for attributing a deal (or branched terminal) to its dominant
# kind: an all-in dominates the payoff spread, a fold contributes almost none.
# Single source of truth — variance decomposition grouping uses it too.
TERMINAL_SEVERITY = {"fold": 0, "showdown": 1, "allin": 2}


def dominant_terminal(*kinds: str) -> str:
    """Highest-variance terminal type among ``kinds`` (allin > showdown > fold)."""
    return max(kinds, key=TERMINAL_SEVERITY.__getitem__)


@dataclass
class LBRResult:
    """LBR exploitability estimate with statistical uncertainty.

    ``base_seed`` is the seed that actually anchored per-hand seeding (recorded even
    when the config seed was ``None``), and ``hand_outcomes`` holds the per-deal
    ``(p0, p1)`` outcome pair. Together they enable paired common-random-numbers
    comparisons across checkpoints: two evals run with the same ``base_seed`` see
    identical deals, so the difference of per-hand samples has far lower variance
    than the two independent estimates.
    """

    exploitability_mbb: float
    exploitability_bb: float
    lbr_utility_p0: float
    lbr_utility_p1: float
    std_error_mbb: float
    confidence_95_mbb: tuple[float, float]
    num_hands: int
    base_seed: int
    hand_outcomes: list[tuple[HandOutcome, HandOutcome]]


class _HUNLLocalBestResponse:
    """Plays LBR against a frozen blueprint and measures realized value.

    One instance owns a shared hand evaluator and per-call caches; it is not
    thread-safe. Callers should use :func:`compute_lbr_exploitability`.
    """

    def __init__(self, blueprint: Blueprint, config: LBRConfig, rng: np.random.Generator):
        self.blueprint = blueprint
        self.config = config
        self.rng = rng
        self.rules: GameRules = blueprint.rules
        self.action_model: ActionModel = blueprint.action_model
        self.card_abstraction = blueprint.card_abstraction
        self.storage = blueprint.storage
        self.evaluator = get_evaluator()
        # On-tree shadow state for rigorous off-tree play (see module docs).
        self.shadow = ShadowTracker(self.rules, self.action_model)
        if config.scorer not in ("myopic", "lookahead"):
            raise ValueError(f"Unknown LBRConfig.scorer: {config.scorer!r}")
        # The scorer ALWAYS uses the blueprint model (selection-only, see module
        # docs); the realized path uses whichever strategy is under measurement.
        # The cross-call memo exists only under the lookahead scorer, keeping the
        # myopic path byte-identical.
        dist_memo = BlueprintDistMemo() if config.scorer == "lookahead" else None
        self._blueprint_model = BlueprintOpponent(blueprint, dist_memo=dist_memo)
        self._lookahead: LookaheadScorer | None = None
        if config.scorer == "lookahead":
            self._lookahead = LookaheadScorer(
                blueprint_model=self._blueprint_model,
                rules=self.rules,
                action_model=self.action_model,
                is_chance_node=blueprint.is_chance_node,
                equity_fn=self._equity,
                depth=config.lookahead_depth,
            )
        self.opponent: OpponentModel
        if config.opponent == "blueprint":
            self.opponent = self._blueprint_model
        elif config.opponent == "deployed":
            if config.resolver is None:
                raise ValueError(
                    'LBRConfig.opponent="deployed" requires LBRConfig.resolver '
                    "(with max_iterations set)."
                )
            # Child generator: the resolver's runout sampling stays reproducible
            # under the eval seed without entangling it with the deal stream.
            self.opponent = ResolvedOpponent(
                blueprint,
                config.resolver,
                rng=np.random.default_rng(int(self.rng.integers(2**63))),
            )
        else:
            raise ValueError(f"Unknown LBRConfig.opponent: {config.opponent!r}")

    # -- Realized play -----------------------------------------------------

    def play_hand(self, lbr_player: int, initial_state: GameState) -> HandOutcome:
        """Play one hand with ``lbr_player`` using LBR; return its realized outcome.

        The opponent is carried as a **range** (``belief``), never a sampled
        hand: its actions are drawn from the range-aggregate distribution and
        Bayes-update the range, and terminals are valued analytically against the
        surviving range. When *every* legal opponent action ends the hand (facing
        an all-in, or closing the river), the action itself is integrated out
        instead of sampled — the fold-vs-call coinflip on jams is the largest
        payoff spread in the game, so branching it removes the dominant remaining
        opponent variance at the cost of valuing each branch once. Only public
        chance (the board) and the LBR player's own hand are sampled.
        """
        opp = 1 - lbr_player
        state = initial_state
        belief = self._initial_belief(initial_state, opp)
        self.opponent.reset(initial_state, opp)
        shadow = self.shadow
        shadow.start(initial_state)

        while not state.is_terminal:
            if self.blueprint.is_chance_node(state):
                state = self.blueprint.sample_chance_outcome(state)
                shadow.mirror_chance(state)
                continue

            shadow.assert_sync(state)
            if state.current_player == lbr_player:
                action, shadow_action = self._choose_lbr_action(state, shadow, lbr_player, belief)
            else:
                if self.opponent.wants_translated_state:
                    legal, vecs = self.opponent.action_matrix(shadow.state, opp)
                    pairs = [(a, shadow.map_back(state, a)) for a in legal]
                else:
                    legal, vecs = self.opponent.action_matrix(state, opp)
                    pairs = [(a, a) for a in legal]
                branched = self._branch_terminal_decision(
                    state, lbr_player, opp, pairs, vecs, belief
                )
                if branched is not None:
                    return branched
                choice, belief = self._sample_opponent(legal, vecs, belief)
                shadow_action, action = pairs[choice]
                if not self.opponent.wants_translated_state:
                    # Deployed mode acts on the real state; mirror its realized
                    # action back onto the shadow (or give up shadowing — the
                    # shadow only feeds the scorer there, see ShadowTracker).
                    mirrored = shadow.counterpart(state, action)
                    if mirrored is None:
                        shadow.mark_broken(state)
                        shadow_action = action
                    else:
                        shadow_action = mirrored

            self.opponent.observe(state, action)
            next_state = state.apply_action(action, self.rules)
            if not next_state.is_terminal:
                shadow.commit(action, next_state, shadow_action)
            state = next_state

        value = self._terminal_value(state, lbr_player, opp, belief)
        return HandOutcome(value=value, terminal=self._terminal_kind(state), pot=state.pot)

    def _branch_terminal_decision(
        self,
        state: GameState,
        lbr_player: int,
        opp: int,
        pairs: list[tuple[Action, Action]],
        vecs: dict[Action, np.ndarray],
        belief: np.ndarray,
    ) -> HandOutcome | None:
        """Integrate out an opponent decision whose every action ends the hand.

        ``pairs`` holds ``(decision_action, real_action)`` per legal action of
        the opponent's decision state (identical when the decision state is the
        real state): probabilities and posteriors come from the decision action's
        per-combo vector, terminality and values from applying the real action.
        Returns the probability-weighted mix of per-branch terminal values (each
        branch valued against its own Bayes-posterior range), or ``None`` when any
        action continues the hand — then the caller samples as usual. Replacing
        the sample with its exact conditional expectation is a Rao-Blackwellization:
        the estimator's expectation (and the LBR bound) is unchanged.
        """
        if not pairs:
            return None
        # BET/RAISE always reopen the betting, so such menus can never be
        # all-terminal — skip without paying any apply_action. (ALL_IN cannot be
        # pre-filtered: calling a jam is itself encoded as ALL_IN and IS terminal.)
        if any(decision.type in (ActionType.BET, ActionType.RAISE) for decision, _ in pairs):
            return None
        next_states = []
        for _, real_action in pairs:
            next_state = state.apply_action(real_action, self.rules)
            if not next_state.is_terminal:
                return None
            next_states.append(next_state)

        probs = np.array([float(np.dot(belief, vecs[decision])) for decision, _ in pairs])
        total = probs.sum()
        if total <= NORMALIZE_EPS:
            return None  # degenerate belief: let the caller's uniform fallback handle it

        value = 0.0
        terminal, pot = "fold", 0
        for weight, (decision, _), next_state in zip(probs / total, pairs, next_states):
            if weight <= 0.0:
                continue
            posterior = belief * vecs[decision]
            mass = posterior.sum()
            branch_belief = posterior / mass if mass > NORMALIZE_EPS else belief
            value += weight * self._terminal_value(next_state, lbr_player, opp, branch_belief)
            kind = self._terminal_kind(next_state)
            if TERMINAL_SEVERITY[kind] >= TERMINAL_SEVERITY[terminal]:
                terminal, pot = kind, next_state.pot
        return HandOutcome(value=value, terminal=terminal, pot=pot)

    def _terminal_kind(self, state: GameState) -> str:
        """Variance-attribution label of a terminal state."""
        if not self._is_showdown(state):
            return "fold"
        if len(state.board) < 5:
            return "allin"
        return "showdown"

    def _initial_belief(self, state: GameState, opp: int) -> np.ndarray:
        """Uniform opponent range vector masked by the LBR player's known cards."""
        known = known_mask(state, opp)
        weights = np.where((COMBO_MASKS & known) == 0, 1.0, 0.0)
        total = weights.sum()
        if total <= NORMALIZE_EPS:
            return np.full(NUM_COMBOS, 1.0 / NUM_COMBOS)
        return weights / total

    def _terminal_value(
        self, state: GameState, lbr_player: int, opp: int, belief: np.ndarray
    ) -> float:
        """Realized LBR payoff at a terminal, valued against the surviving range.

        A fold ends the hand independently of the cards, so the pot-based payoff
        is exact. A river showdown is valued as the belief-weighted payoff over
        every opponent combo still in range. An early all-in (board incomplete)
        additionally averages that value over board runouts — integrating out both
        the opponent hand and (approximately) the runout rather than sampling them.
        """
        if not self._is_showdown(state):
            return float(state.get_payoff(lbr_player, self.rules))
        if len(state.board) == 5:
            return self._showdown_value(state, lbr_player, opp, belief)
        return self._allin_showdown_value(state, lbr_player, opp, belief)

    def _showdown_value(
        self, state: GameState, lbr_player: int, opp: int, belief: np.ndarray
    ) -> float:
        """Belief-weighted payoff over the surviving range on a complete board.

        Payoffs are pot arithmetic on hand-rank comparisons (win: pot - invested,
        tie: pot/2 - invested, lose: -invested — exactly ``get_payoff``'s showdown
        cases), so no per-combo GameState construction is needed.
        """
        known = known_mask(state, opp)
        weights = np.where((COMBO_MASKS & known) == 0, belief, 0.0)
        total = weights.sum()
        if total <= NORMALIZE_EPS:
            return float(state.get_payoff(lbr_player, self.rules))

        pot = float(state.pot)
        invested = self.rules.invested_chips(state)[lbr_player]
        win_payoff = pot - invested
        tie_payoff = pot / 2.0 - invested
        lose_payoff = -invested
        lbr_rank = self.evaluator.evaluate(state.hole_cards[lbr_player], state.board)

        ev = 0.0
        for idx in np.nonzero(weights)[0]:
            opp_rank = self.evaluator.evaluate(ALL_COMBOS[idx], state.board)
            if lbr_rank < opp_rank:
                payoff = win_payoff
            elif lbr_rank == opp_rank:
                payoff = tie_payoff
            else:
                payoff = lose_payoff
            ev += float(weights[idx]) * payoff
        return ev / float(total)

    def _allin_showdown_value(
        self, state: GameState, lbr_player: int, opp: int, belief: np.ndarray
    ) -> float:
        """All-in showdown value averaged over board runouts.

        Runouts draw from the cards the LBR player cannot see (its holes + board);
        the opponent's *dealt* hand is deliberately not excluded — it is a fiction
        this evaluator integrates out, so letting runouts cover those cards matches
        the range convention (combos colliding with a runout drop out per runout,
        exactly as on a real river). One missing card is enumerated exactly; more
        are sampled ``allin_runouts`` times from the per-hand deterministic RNG.
        """
        known = known_mask(state, opp)
        missing = 5 - len(state.board)
        unseen = [card for card in FULL_DECK if not (card.mask & known)]
        runouts: list[tuple[Card, ...]]
        if missing == 1:
            runouts = [(card,) for card in unseen]
        else:
            count = max(1, self.config.allin_runouts)
            runouts = []
            for _ in range(count):
                picks = self.rng.choice(len(unseen), size=missing, replace=False)
                runouts.append(tuple(unseen[int(i)] for i in picks))

        total = 0.0
        for extra in runouts:
            runout_state = self._with_runout(state, extra)
            total += self._showdown_value(runout_state, lbr_player, opp, belief)
        return total / len(runouts)

    @staticmethod
    def _with_runout(state: GameState, extra: tuple[Card, ...]) -> GameState:
        """Terminal copy of ``state`` with the board completed by ``extra``."""
        return state.replace(
            street=Street.RIVER,
            board=state.board + extra,
            is_terminal=True,
            to_call=0,
            validate=False,
        )

    # -- LBR action choice -------------------------------------------------

    def _choose_lbr_action(
        self, state: GameState, shadow: ShadowTracker, lbr_player: int, belief: np.ndarray
    ) -> tuple[Action, Action]:
        """Pick the argmax menu candidate; return its (real, shadow) actions.

        Under the lookahead scorer, the myopic scores act as a prefilter: only
        the top ``lookahead_top_k`` candidates (which always include the myopic
        argmax) are rescored by the lookahead walk. The committed shadow proxy
        is sampled once, only for the chosen action.
        """
        opp = 1 - lbr_player
        opp_weights = belief
        lbr_hand = state.hole_cards[lbr_player]
        candidates = self._action_menu(state, shadow)
        myopic = [
            self._score_action(state, shadow.state, opp, lbr_hand, opp_weights, candidate)
            for candidate in candidates
        ]
        best = candidates[0]
        best_value = float("-inf")
        if self._lookahead is None:
            for candidate, value in zip(candidates, myopic):
                if value > best_value:
                    best_value = value
                    best = candidate
        else:
            top_k = self.config.lookahead_top_k
            k = len(candidates) if top_k <= 0 else min(top_k, len(candidates))
            order = sorted(range(len(candidates)), key=lambda i: (-myopic[i], i))
            for idx in order[:k]:
                value = self._lookahead.score(
                    state, shadow.state, opp, lbr_hand, opp_weights, candidates[idx]
                )
                if value > best_value:
                    best_value = value
                    best = candidates[idx]
        dist = best.shadow_dist
        if len(dist) == 1:
            return best.real_action, dist[0][0]
        weights = np.array([weight for _, weight in dist])
        total = weights.sum()
        # A degenerate off-tree translation distribution (all-zero weights) would
        # make ``weights / total`` NaN and crash ``rng.choice``; fall back to uniform.
        probs = weights / total if total > 0 else np.full(len(dist), 1.0 / len(dist))
        choice = int(self.rng.choice(len(dist), p=probs))
        return best.real_action, dist[choice][0]

    def _action_menu(self, state: GameState, shadow: ShadowTracker) -> list[MenuCandidate]:
        """The exploiter's menu: mirrored on-tree actions plus gated off-tree sizes.

        Every candidate carries its shadow realization; candidates with no
        structure-preserving shadow proxy are gated out (restricting the
        exploiter only loosens the lower bound). Off-tree BETs are offered when
        leading, off-tree RAISEs when facing a bet (which covers preflop, where
        the first decision always faces the blind gap). Sizes that would be
        de-facto jams (reaching the acting stack) or exceed what the opponent
        can call are skipped — the on-tree ALL_IN already represents them.
        """
        legal = list(self.rules.get_legal_actions(state, action_model=self.action_model))
        menu: list[MenuCandidate] = []
        seen = {(a.type, a.amount) for a in legal}
        for action in legal:
            mirror = shadow.counterpart(state, action)
            if mirror is None:
                continue
            menu.append(MenuCandidate(action, ((mirror, 1.0),)))

        if not self.config.include_off_tree or shadow.broken:
            return menu

        current_stack = state.stacks[state.current_player]
        opp_stack = state.stacks[1 - state.current_player]
        leading = state.to_call == 0
        base = state.pot if leading else state.pot + state.to_call
        action_type = ActionType.BET if leading else ActionType.RAISE
        for frac in self.config.off_tree_pot_fractions:
            amount = round(frac * base)
            committed = amount if leading else state.to_call + amount
            if amount <= 0 or committed >= current_stack or amount > opp_stack:
                continue
            action = Action(action_type, amount)
            if (action.type, action.amount) in seen:
                continue
            if not self.rules.is_action_valid(state, action):
                continue
            dist = shadow.off_tree_dist(state, action)
            if dist is None:
                continue
            seen.add((action.type, action.amount))
            menu.append(MenuCandidate(action, dist))
        return menu

    def _score_action(
        self,
        state: GameState,
        shadow_state: GameState,
        opp: int,
        lbr_hand: tuple[Card, Card],
        opp_weights: np.ndarray,
        candidate: MenuCandidate,
    ) -> float:
        """Myopic (check/call-to-showdown) value of a candidate; selection only.

        Chip arithmetic uses the real action on the real state; the opponent's
        fold response is read on the shadow (where blueprint lookups are
        defined). Approximations here loosen the bound, never invalidate it.
        """
        action = candidate.real_action
        pot = float(state.pot)
        if action.type == ActionType.FOLD:
            return 0.0

        if action.type in (ActionType.CHECK, ActionType.CALL):
            wp = self._equity(lbr_hand, state.board, opp_weights)
            to_call = float(state.to_call)
            # Check: no chips added. Call: risk `to_call` to contest the pot.
            return wp * pot - (1.0 - wp) * to_call

        # Aggressive action (bet/raise/all-in): fold equity + called equity.
        size = self._chips_committed(state, action)
        fold_probs = self._opp_fold_probs(shadow_state, opp, candidate.shadow_dist)
        fold_equity = float(np.dot(opp_weights, fold_probs))
        weight_sum = float(opp_weights.sum())
        fp = fold_equity / weight_sum if weight_sum > NORMALIZE_EPS else 0.0

        continue_weights = opp_weights * (1.0 - fold_probs)
        wp = self._equity(lbr_hand, state.board, continue_weights)
        called_value = wp * (pot + size) - (1.0 - wp) * size
        return fp * pot + (1.0 - fp) * called_value

    def _chips_committed(self, state: GameState, action: Action) -> float:
        """Chips the LBR player puts in beyond any amount already matched."""
        if action.type == ActionType.BET:
            return float(action.amount)
        if action.type == ActionType.RAISE:
            return float(state.to_call + action.amount)
        if action.type == ActionType.ALL_IN:
            return float(action.amount)
        return 0.0

    def _sample_opponent(
        self, legal: list[Action], vecs: dict[Action, np.ndarray], belief: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """Sample an opponent action index from the range aggregate and Bayes-update."""
        aggregate = np.array([float(np.dot(belief, vecs[action])) for action in legal])
        total = aggregate.sum()
        if total <= NORMALIZE_EPS:
            return int(self.rng.integers(0, len(legal))), belief
        aggregate /= total
        choice = int(self.rng.choice(len(legal), p=aggregate))
        posterior = belief * vecs[legal[choice]]
        mass = posterior.sum()
        return choice, (posterior / mass if mass > NORMALIZE_EPS else belief)

    def _opp_fold_probs(
        self,
        shadow_state: GameState,
        opp: int,
        shadow_dist: tuple[tuple[Action, float], ...],
    ) -> np.ndarray:
        """Per-combo probability the opponent folds to the candidate (scorer input).

        Mixes the blueprint's fold response over the candidate's shadow proxies
        (the states blueprint lookups are defined on). Deliberately
        blueprint-backed even under opponent="deployed": this feeds the
        exploiter's selection scorer only, where approximation loosens the
        lower bound but never invalidates it — and resolver-backing it would
        multiply eval cost ~10x (one solve per candidate action per decision).
        """
        fold_probs = np.zeros(NUM_COMBOS)
        for proxy, weight in shadow_dist:
            resp_state = shadow_state.apply_action(proxy, self.rules)
            legal, vecs = self._blueprint_model.action_matrix(resp_state, opp)
            for response in legal:
                if response.type == ActionType.FOLD:
                    fold_probs += weight * vecs[response]
        return fold_probs

    # -- Equity ------------------------------------------------------------

    def _equity(
        self, lbr_hand: tuple[Card, Card], board: tuple[Card, ...], opp_weights: np.ndarray
    ) -> float:
        """Weighted win probability of ``lbr_hand`` vs the opponent range.

        Board cards are rolled out to the river when incomplete. Scorer-only, so
        Monte-Carlo runout noise is acceptable — it only loosens the bound.
        """
        known = 0
        for card in lbr_hand:
            known |= card.mask
        for card in board:
            known |= card.mask
        active = np.where((opp_weights > NORMALIZE_EPS) & ((COMBO_MASKS & known) == 0))[0]
        if active.size == 0:
            return 0.5

        if len(board) >= 5:
            return self._showdown_equity(lbr_hand, board, opp_weights, active)

        runouts = max(1, self.config.equity_runouts)
        total = 0.0
        for _ in range(runouts):
            full_board = self._complete_board(board, known)
            total += self._showdown_equity(lbr_hand, full_board, opp_weights, active)
        return total / runouts

    def _showdown_equity(
        self,
        lbr_hand: tuple[Card, Card],
        board: tuple[Card, ...],
        opp_weights: np.ndarray,
        active: np.ndarray,
    ) -> float:
        board_mask = 0
        for card in board:
            board_mask |= card.mask
        lbr_rank = self.evaluator.evaluate(lbr_hand, board)
        acc = 0.0
        weight = 0.0
        for idx in active:
            if COMBO_MASKS[idx] & board_mask:
                continue
            w = opp_weights[idx]
            opp_rank = self.evaluator.evaluate(ALL_COMBOS[idx], board)
            if lbr_rank < opp_rank:
                acc += w
            elif lbr_rank == opp_rank:
                acc += 0.5 * w
            weight += w
        return acc / weight if weight > NORMALIZE_EPS else 0.5

    def _complete_board(self, board: tuple[Card, ...], known: int) -> tuple[Card, ...]:
        needed = 5 - len(board)
        drawn: list[Card] = []
        used = known
        while len(drawn) < needed:
            idx = int(self.rng.integers(0, 52))
            mask = int(_DECK_MASKS[idx])
            if used & mask:
                continue
            used |= mask
            drawn.append(FULL_DECK[idx])
        return tuple(board) + tuple(drawn)

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _is_showdown(state: GameState) -> bool:
        return bool(state.betting_history) and not state.ended_by_fold


def compute_lbr_exploitability(
    blueprint: Blueprint,
    config: LBRConfig | None = None,
    *,
    blueprint_factory: Callable[[], Blueprint] | None = None,
) -> LBRResult:
    """Estimate the blueprint's exploitability via Local Best Response.

    Plays ``config.num_hands`` deals; on each, the LBR player exploits from both
    positions (paired sampling on the same deal to cut variance) and the sample
    exploitability is ``(u_p0 + u_p1) / 2``. This mirrors the convention in
    :func:`~src.pipeline.evaluation.exploitability.compute_exploitability`,
    which assumes button-symmetrized on-policy value is ~0, so the two figures
    are directly comparable — LBR should never report *less*.

    To *compare* two blueprints, run both evals with the same explicit seed and
    feed the per-hand samples (``result.hand_outcomes``) to
    :func:`~src.pipeline.evaluation.statistics.compare_paired_samples` — the
    deals match hand-for-hand, so the paired difference resolves far smaller
    gaps than two independent confidence intervals.
    """
    if config is None:
        config = LBRConfig()
    # A base seed anchors per-hand seeding. Each hand is seeded from (base_seed, hand),
    # so the set of per-hand results — and thus the aggregate — is identical for any
    # worker count (num_workers only changes how the hands are distributed).
    base_seed = (
        config.seed
        if config.seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    starting_stack = blueprint.config.game.starting_stack
    big_blind = blueprint.config.game.big_blind

    num_workers = max(1, config.num_workers)
    if num_workers == 1:
        engine = _HUNLLocalBestResponse(blueprint, config, np.random.default_rng(base_seed))
        pairs = [
            _play_hand_pair(engine, hand, base_seed, starting_stack)
            for hand in tqdm(range(config.num_hands), desc="LBR hands", unit="hand")
        ]
    else:
        if blueprint_factory is None:
            raise ValueError(
                "num_workers > 1 requires blueprint_factory: the solver holds a "
                "non-picklable Cython member, so each worker must build its own from "
                "a picklable spec (e.g. config + checkpoint dir)."
            )
        pairs = _run_hands_parallel(
            blueprint_factory, config, base_seed, starting_stack, num_workers
        )

    utilities_p0 = [o0.value for o0, _ in pairs]
    utilities_p1 = [o1.value for _, o1 in pairs]
    samples = [(o0.value + o1.value) / 2.0 for o0, o1 in pairs]

    exploitability = float(np.mean(samples)) if samples else float("nan")
    if len(samples) >= 2:
        se = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    else:
        se = 0.0

    exploitability_mbb = chips_to_mbb(exploitability, big_blind)
    se_mbb = chips_to_mbb(se, big_blind)
    # t-multiplier with n-1 df, matching the sibling evaluators' summarize_samples;
    # a fixed z=1.96 understates the interval at the small hand counts LBR often runs.
    t_mult = float(stats.t.ppf(0.975, len(samples) - 1)) if len(samples) >= 2 else 0.0
    return LBRResult(
        exploitability_mbb=exploitability_mbb,
        exploitability_bb=chips_to_bb(exploitability, big_blind),
        lbr_utility_p0=float(np.mean(utilities_p0)) if utilities_p0 else float("nan"),
        lbr_utility_p1=float(np.mean(utilities_p1)) if utilities_p1 else float("nan"),
        std_error_mbb=se_mbb,
        confidence_95_mbb=(
            exploitability_mbb - t_mult * se_mbb,
            exploitability_mbb + t_mult * se_mbb,
        ),
        num_hands=len(samples),
        base_seed=base_seed,
        hand_outcomes=pairs,
    )


def _deal_initial_state(
    engine: _HUNLLocalBestResponse,
    starting_stack: int,
    button: int,
    rng: np.random.Generator,
) -> GameState:
    """Deal a fresh hand (both hole cards) using ``rng`` for reproducibility."""
    order = rng.permutation(52)
    cards = [FULL_DECK[int(i)] for i in order[:4]]
    hole_cards = ((cards[0], cards[1]), (cards[2], cards[3]))
    return engine.rules.create_initial_state(
        starting_stack=starting_stack,
        hole_cards=hole_cards,
        button=button,
    )


def _hand_seed(base_seed: int, hand: int) -> int:
    """Per-hand seed, independent of worker assignment, for reproducible parallel LBR."""
    return int(np.random.SeedSequence([base_seed, hand]).generate_state(1)[0])


# Distinguishes the opponent-model stream from the deal/LBR stream under the
# same (base_seed, hand): the two must not be correlated.
_OPPONENT_STREAM = 1


def _opponent_hand_seed(base_seed: int, hand: int) -> int:
    """Per-hand seed for the opponent model's own randomness.

    Kept on a separate stream from ``_hand_seed`` so the opponent's sampling is
    uncorrelated with the deal it is responding to.
    """
    return int(np.random.SeedSequence([base_seed, hand, _OPPONENT_STREAM]).generate_state(1)[0])


def _play_hand_pair(
    engine: _HUNLLocalBestResponse,
    hand: int,
    base_seed: int,
    starting_stack: int,
) -> tuple[HandOutcome, HandOutcome]:
    """Play one deal from both positions under a per-hand deterministic RNG.

    Reseeds every randomness source consumed during the hand — the engine's own
    ``rng``, the global ``random`` module (the blueprint deals the board via
    ``random.shuffle``), and the opponent model's own generator — from
    ``(base_seed, hand)``, so the result is independent of which worker runs the hand
    or in what order. ``np.random`` is reseeded defensively (no global ``np.random``
    is used in the current LBR path, but this guards new ones).

    The opponent is reseeded identically before each seat: the two games of a pair
    are the same deal with the seats swapped, so giving them common random numbers
    keeps the pair a true paired sample rather than adding opponent noise to the
    difference.
    """
    s_h = _hand_seed(base_seed, hand)
    random.seed(s_h)
    np.random.seed(s_h)
    engine.rng = np.random.default_rng(s_h)
    s_opp = _opponent_hand_seed(base_seed, hand)
    button = hand % 2
    state = _deal_initial_state(engine, starting_stack, button, engine.rng)
    engine.opponent.reseed(s_opp)
    outcome_p0 = engine.play_hand(0, state)
    engine.opponent.reseed(s_opp)
    outcome_p1 = engine.play_hand(1, state)
    return outcome_p0, outcome_p1


# --- Parallel execution -------------------------------------------------------
# Workers use the "spawn" start method, never fork: building the blueprint spins up
# numba/BLAS thread pools, and forking a threaded process deadlocks on inherited locks.
# The solver is NOT picklable (Cython member), so each worker builds its OWN blueprint
# from the picklable factory. Per-hand seeding makes the result identical to serial.
_WORKER_ENGINE: _HUNLLocalBestResponse | None = None
_WORKER_CTX: tuple[int, int] | None = None  # (base_seed, starting_stack)


def _init_worker(
    blueprint_factory: Callable[[], Blueprint],
    config: LBRConfig,
    base_seed: int,
    starting_stack: int,
) -> None:
    global _WORKER_ENGINE, _WORKER_CTX
    # Worker subprocess stdout (e.g. checkpoint-load logs from building the blueprint)
    # cannot be captured by the parent's stdout redirect; route it to stderr so the
    # parent's --json output on stdout stays clean.
    sys.stdout = sys.stderr
    blueprint = blueprint_factory()
    _WORKER_ENGINE = _HUNLLocalBestResponse(blueprint, config, np.random.default_rng(base_seed))
    _WORKER_CTX = (base_seed, starting_stack)


def _worker_play_hand(hand: int) -> tuple[HandOutcome, HandOutcome]:
    assert _WORKER_ENGINE is not None and _WORKER_CTX is not None
    base_seed, starting_stack = _WORKER_CTX
    return _play_hand_pair(_WORKER_ENGINE, hand, base_seed, starting_stack)


def _run_hands_parallel(
    blueprint_factory: Callable[[], Blueprint],
    config: LBRConfig,
    base_seed: int,
    starting_stack: int,
    num_workers: int,
) -> list[tuple[HandOutcome, HandOutcome]]:
    ctx = multiprocessing.get_context("spawn")
    chunksize = max(1, config.num_hands // (num_workers * 4))
    with ctx.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(blueprint_factory, config, base_seed, starting_stack),
    ) as pool:
        # imap preserves input order exactly like map (results stay in canonical hand
        # order, so the mean/std reduction is bitwise-identical), but yields as chunks
        # finish — so tqdm can render live progress on an otherwise silent eval.
        return list(
            tqdm(
                pool.imap(_worker_play_hand, range(config.num_hands), chunksize=chunksize),
                total=config.num_hands,
                desc="LBR hands",
                unit="hand",
            )
        )
