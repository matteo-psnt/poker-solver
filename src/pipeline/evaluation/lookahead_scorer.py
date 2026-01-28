"""Depth-limited best-response lookahead scorer for the HUNL LBR exploiter.

The myopic scorer (one-step check/call-to-showdown pot arithmetic) misranks
candidates: measured on both weak and converged blueprints, widening the menu
made the exploiter *worse* because the argmax kept picking overestimated
actions. This scorer replaces the continuation model with a small expectimax
walk **against the blueprint's actual policy** — opponent nodes branch over the
blueprint's per-combo action distribution (Bayes-updating the belief per
branch), exploiter nodes take the max over their on-tree menu, and leaves are
valued by check-down equity against the posterior range. It is deliberately
NOT a CFR re-solve: an equilibrium solve assumes a counter-adapting opponent
and systematically underestimates how exploitable the fixed blueprint is.

Scoring is selection-only: the reported LBR figure is the realized payoff of
the chosen actions, so any approximation here (depth limit, runout noise, the
chip-scaling rule) loosens the lower bound but never invalidates it.

Chip-space rule
---------------
Valuation never reads chips off the shadow state. The walk carries explicit
``(pot, mine)`` accumulators in REAL-chip units: they start from the real state
at the scored decision, the root candidate adds its real committed chips (so
off-tree candidates stay distinguishable from their on-tree proxies), and every
deeper hypothetical action adds its shadow committed chips scaled by the fixed
ratio ``real_pot / shadow_pot`` taken once at the scored decision (1 unless the
hand diverged earlier; preflop map-back is blind-anchored, so the ratio is a
first-order approximation there — scorer-only, bound-safe). Shadow states
provide only structure: menus, turn order, street/terminal transitions, and
blueprint lookups.

Depth semantics
---------------
``depth`` counts opponent-response levels. Opponent nodes are ALWAYS expanded —
they are the fold-equity source, and cutting at an exploiter bet would value it
as called-and-checked-down with zero fold equity, burying exactly the bluffs
the lookahead exists to find. Exploiter re-decisions consume budget: ``depth``
allows ``depth - 1`` of them, so ``depth=1`` is the branch-resolved analogue of
the myopic scorer and ``depth=2`` (default) adds one exploiter re-decision and
the opponent's response to it. Street boundaries are leaves (no chance fanout,
matching the resolver's local-tree convention); leaf equity rolls the board out.

Determinism: candidates and actions are iterated in deterministic order and all
sampling happens inside the injected ``equity_fn`` (the LBR engine's
``_equity``, which draws from the per-hand engine RNG), so parallel == serial
is preserved by the existing per-hand reseeding.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState
from src.pipeline.evaluation.shadow_state import MenuCandidate
from src.shared.numeric import NORMALIZE_EPS

# The blueprint response distribution at one public opponent node. Keyed
# beyond the infoset key: two public states can share an infoset key (same
# normalized sequence / bucket / SPR bucket) while differing in exact chips,
# which changes the legal menu and hence the distribution.
MemoKey = tuple[object, int, int, tuple[int, int]]

EquityFn = Callable[[tuple[Card, Card], tuple[Card, ...], np.ndarray], float]


class BlueprintDistMemo:
    """Cross-call memo for blueprint response distributions.

    Engine-lifetime and worker-local (each spawn worker builds its own engine).
    Freeze-on-full: once ``max_entries`` is reached no new entries are inserted
    — deterministic under any hit pattern (no eviction-order dependence), and a
    real eval touches far fewer unique keys than the cap. Entries carry the
    ``was_fallback`` flag so the owner's ``queries``/``uniform_fallbacks``
    counters keep their per-logical-query semantics on memo hits.
    """

    def __init__(self, max_entries: int = 2_000_000):
        self.max_entries = max_entries
        self._entries: dict[MemoKey, tuple[dict[Action, float], bool]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(infoset_key: object, state: GameState) -> MemoKey:
        return (infoset_key, state.pot, state.to_call, state.stacks)

    def get(self, key: MemoKey) -> tuple[dict[Action, float], bool] | None:
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def put(self, key: MemoKey, dist: dict[Action, float], was_fallback: bool) -> None:
        self.misses += 1
        if len(self._entries) < self.max_entries:
            self._entries[key] = (dist, was_fallback)

    def __len__(self) -> int:
        return len(self._entries)


def committed_chips(state: GameState, action: Action) -> int:
    """Chips ``state.current_player`` adds to the pot by taking ``action``."""
    if action.type == ActionType.CALL:
        return min(state.to_call, state.stacks[state.current_player])
    if action.type == ActionType.BET:
        return action.amount
    if action.type == ActionType.RAISE:
        return state.to_call + action.amount
    if action.type == ActionType.ALL_IN:
        return action.amount
    return 0  # FOLD / CHECK


class LookaheadScorer:
    """Best-response expectimax over the shadow tree vs the blueprint policy.

    Dependencies are injected callables so unit tests can script the opponent
    and the equity model without a trained blueprint:

    - ``blueprint_model``: exposes ``action_matrix(state, actor) ->
      (legal, {action: per-combo prob vector})`` (the LBR engine's
      always-blueprint-backed model — the scorer stays blueprint-backed even
      under ``opponent="deployed"``, same rationale as the myopic fold-probs).
    - ``equity_fn(lbr_hand, board, opp_weights)``: win probability vs the
      weighted range, rolling out incomplete boards (the engine's ``_equity``).
    - ``is_chance_node(state)``: street-boundary detection (the blueprint's).
    """

    def __init__(
        self,
        *,
        blueprint_model,
        rules: GameRules,
        action_model: ActionModel,
        is_chance_node: Callable[[GameState], bool],
        equity_fn: EquityFn,
        depth: int,
    ):
        if depth < 1:
            raise ValueError(f"lookahead depth must be >= 1, got {depth}")
        self._blueprint_model = blueprint_model
        self._rules = rules
        self._action_model = action_model
        self._is_chance_node = is_chance_node
        self._equity = equity_fn
        self._depth = depth

    def score(
        self,
        real_state: GameState,
        shadow_state: GameState,
        opp: int,
        lbr_hand: tuple[Card, Card],
        belief: np.ndarray,
        candidate: MenuCandidate,
    ) -> float:
        """Lookahead value of ``candidate`` for the exploiter; selection only."""
        action = candidate.real_action
        if action.type == ActionType.FOLD:
            return 0.0  # same convention as the myopic scorer

        ratio = real_state.pot / shadow_state.pot
        root_mine = float(committed_chips(real_state, action))
        root_pot = float(real_state.pot) + root_mine

        value = 0.0
        for proxy, weight in candidate.shadow_dist:
            child = shadow_state.apply_action(proxy, self._rules)
            value += weight * self._walk(
                child,
                belief,
                pot=root_pot,
                mine=root_mine,
                budget=self._depth - 1,
                ratio=ratio,
                opp=opp,
                lbr_hand=lbr_hand,
            )
        return value

    def _walk(
        self,
        node: GameState,
        belief: np.ndarray,
        *,
        pot: float,
        mine: float,
        budget: int,
        ratio: float,
        opp: int,
        lbr_hand: tuple[Card, Card],
    ) -> float:
        if (
            node.is_terminal
            or self._is_chance_node(node)
            or (node.current_player != opp and budget <= 0)
        ):
            return self._leaf_value(node, belief, pot=pot, mine=mine, ratio=ratio, hand=lbr_hand)

        if node.current_player == opp:
            return self._opponent_node(
                node,
                belief,
                pot=pot,
                mine=mine,
                budget=budget,
                ratio=ratio,
                opp=opp,
                lbr_hand=lbr_hand,
            )

        # Exploiter re-decision: best response over the on-tree shadow menu.
        best = float("-inf")
        for action in self._rules.get_legal_actions(node, action_model=self._action_model):
            if action.type == ActionType.FOLD:
                best = max(best, -mine)
                continue
            chips = ratio * committed_chips(node, action)
            best = max(
                best,
                self._walk(
                    node.apply_action(action, self._rules),
                    belief,
                    pot=pot + chips,
                    mine=mine + chips,
                    budget=budget - 1,
                    ratio=ratio,
                    opp=opp,
                    lbr_hand=lbr_hand,
                ),
            )
        return best

    def _opponent_node(
        self,
        node: GameState,
        belief: np.ndarray,
        *,
        pot: float,
        mine: float,
        budget: int,
        ratio: float,
        opp: int,
        lbr_hand: tuple[Card, Card],
    ) -> float:
        legal, vecs = self._blueprint_model.action_matrix(node, opp)
        if not legal:
            return self._leaf_value(node, belief, pot=pot, mine=mine, ratio=ratio, hand=lbr_hand)

        weights = np.array([float(np.dot(belief, vecs[action])) for action in legal])
        total = weights.sum()
        if total <= NORMALIZE_EPS:
            # Degenerate belief: mirror _sample_opponent's uniform fallback so
            # the scorer matches realized-path semantics in dead-range corners.
            weights = np.full(len(legal), 1.0 / len(legal))
            posteriors: list[np.ndarray] = [belief] * len(legal)
        else:
            weights = weights / total
            posteriors = []
            for action in legal:
                posterior = belief * vecs[action]
                mass = posterior.sum()
                posteriors.append(posterior / mass if mass > NORMALIZE_EPS else belief)

        value = 0.0
        for action, weight, posterior in zip(legal, weights, posteriors):
            if weight <= 0.0:
                continue
            if action.type == ActionType.FOLD:
                value += weight * (pot - mine)  # closed form: pot captured
                continue
            chips = ratio * committed_chips(node, action)
            value += weight * self._walk(
                node.apply_action(action, self._rules),
                posterior,
                pot=pot + chips,
                mine=mine,
                budget=budget,
                ratio=ratio,
                opp=opp,
                lbr_hand=lbr_hand,
            )
        return value

    def _leaf_value(
        self,
        node: GameState,
        belief: np.ndarray,
        *,
        pot: float,
        mine: float,
        ratio: float,
        hand: tuple[Card, Card],
    ) -> float:
        """Call-then-check-down value: fold any pending call in, then equity.

        A pending ``to_call`` at a leaf can only belong to the exploiter
        (opponent nodes are always expanded), so the completion charges
        ``mine``. The board rolls out inside ``equity_fn`` when incomplete.
        """
        if node.to_call > 0:
            chips = ratio * min(node.to_call, node.stacks[node.current_player])
            pot += chips
            mine += chips
        equity = self._equity(hand, node.board, belief)
        return equity * pot - mine
