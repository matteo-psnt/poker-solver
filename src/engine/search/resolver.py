"""HU runtime subgame resolver."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import GameState
from src.engine.search.range_inference import (
    ALL_COMBOS,
    NUM_COMBOS,
    PlayerRanges,
    blocked_combos,
    combo_index_for,
    infer_ranges,
    replace_actor_hole_cards,
    update_ranges,
)
from src.engine.search.subgame_cfr import solve_subgame
from src.engine.search.tree_builder import build_local_tree
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.engine.solver.policy_lookup import blueprint_action_distribution
from src.shared.config import ResolverConfig
from src.shared.numeric import NORMALIZE_EPS


@dataclass
class ResolveResult:
    """Resolver output for a single decision."""

    action: Action
    root_actions: list[Action]
    strategy: np.ndarray
    resolver_strategy: np.ndarray
    blueprint_strategy: np.ndarray
    action_values: np.ndarray


class HUResolver:
    """Depth-limited resolver for heads-up runtime decisions."""

    def __init__(
        self,
        *,
        blueprint,
        action_model: ActionModel,
        rules: GameRules,
        config: ResolverConfig,
    ):
        self.blueprint = blueprint
        self.action_model = action_model
        self.rules = rules
        self.config = config
        # History-replayed ranges: written ONLY by observe(). A driver that
        # feeds every realized action through observe() gives the resolver
        # Bayes-updated ranges for BOTH players; without a driver, solves fall
        # back to fresh board-masked-uniform inference per decision (stateless,
        # board-correct — no cross-hand leakage).
        self._ranges: PlayerRanges | None = None
        # Decisions where solve() raised and act() fell back to the blueprint.
        # Harnesses read this directly; the warning below is for humans only.
        self.fallback_count: int = 0

    def observe(self, state: GameState, action: Action) -> None:
        """Bayes-update the acting player's range for a realized ``action``.

        History-replay range inference: the driver calls this for EVERY realized
        action (both seats) as the hand plays out, so the opponent's observed
        actions shape the opponent range the next solve uses — instead of the
        uniform range that made deployed play measurably exploitable. Likelihoods
        come from the blueprint (translation-completed for off-tree sizes), so
        the update is a model-based inference: "the opponent plays roughly like
        the blueprint." Deterministic; no RNG.
        """
        if self._ranges is None:
            self._ranges = infer_ranges(state, self.blueprint)
        self._ranges = update_ranges(state, self._ranges, action, self.blueprint)

    def act(self, state: GameState, *, time_budget_ms: int | None = None) -> Action:
        """Resolve local subgame and sample an action at the root."""
        budget = int(time_budget_ms or self.config.time_budget_ms)
        try:
            result = self.solve(state, time_budget_ms=budget)
            return result.action
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            self.fallback_count += 1
            warnings.warn(
                f"Resolver failed at runtime, falling back to blueprint strategy: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.blueprint.sample_action_from_strategy(state, use_average=True)

    def solve_strategy_matrix(
        self,
        state: GameState,
        *,
        ranges: PlayerRanges | None = None,
        time_budget_ms: int | None = None,
    ) -> tuple[list[Action], np.ndarray]:
        """Pure strategy oracle: the blended root strategy for EVERY combo at once.

        Returns ``(root_actions, matrix)`` where ``matrix[c, a]`` is the
        probability the deployed system (resolver blended with blueprint) takes
        ``root_actions[a]`` holding combo ``c`` (canonical ``ALL_COMBOS`` order).
        Rows of board-blocked combos are uniform placeholders.

        Unlike :meth:`solve` this samples nothing and mutates nothing — repeated
        calls are side-effect free — and it never consults ANY dealt hole cards
        (``solve`` reads the hero's actual combo to pick its row; here every row
        is produced). That makes it usable as an opponent model in range-based
        evaluators (LBR), which need action distributions per opponent combo.

        ``ranges`` is the resolver's view of both players' ranges at this public
        node; ``None`` means fresh inference (uniform today), matching a resolver
        that just entered the hand. Pass ``config.max_iterations`` for
        reproducible output — budget-driven iteration counts vary with wall clock.
        """
        budget = int(time_budget_ms or self.config.time_budget_ms)
        if ranges is None:
            ranges = infer_ranges(state, self.blueprint)
        root_actions, solution = self._solve_root(state, ranges, budget)
        blueprint_matrix = self._blueprint_strategy_matrix(state, root_actions)
        return root_actions, self._blend_matrix(solution.root_strategy, blueprint_matrix)

    def solve(self, state: GameState, *, time_budget_ms: int | None = None) -> ResolveResult:
        """Run a range-vs-range subgame CFR and return the root strategy + action.

        All valuation is range-honest: the opponent's dealt cards in ``state``
        are never consulted. The local tree is solved with per-combo CFR for
        both players (the opponent counter-adapts inside the solve), and the
        played strategy is the average root strategy at the hero's actual combo,
        blended with the blueprint.
        """
        budget = int(time_budget_ms or self.config.time_budget_ms)
        # Ranges come from observe()'s history replay when a driver provides it;
        # otherwise infer fresh for this state WITHOUT persisting — solve() no
        # longer writes _ranges (observe() is the single update path, so a
        # driver observing the applied action would otherwise double-count it).
        ranges = self._ranges if self._ranges is not None else infer_ranges(state, self.blueprint)

        root_actions, solution = self._solve_root(state, ranges, budget)
        hero = state.current_player
        hero_combo = combo_index_for(state.hole_cards[hero])
        # Rows of root_strategy are already normalized distributions.
        resolver_strategy = solution.root_strategy[hero_combo]
        action_values = solution.root_values[hero_combo]

        blueprint_strategy = self._blueprint_strategy(state, root_actions, use_average=True)
        strategy = self._blend_strategies(resolver_strategy, blueprint_strategy)

        idx = int(np.random.choice(len(root_actions), p=strategy))
        chosen_action = root_actions[idx]

        return ResolveResult(
            action=chosen_action,
            root_actions=root_actions,
            strategy=strategy,
            resolver_strategy=resolver_strategy,
            blueprint_strategy=blueprint_strategy,
            action_values=action_values,
        )

    def _solve_root(self, state: GameState, ranges: PlayerRanges, budget_ms: int):
        """Build the local tree and run the subgame CFR; shared by solve paths."""
        hero = state.current_player
        hero_range = ranges.p0 if hero == 0 else ranges.p1
        opponent_range = ranges.p1 if hero == 0 else ranges.p0

        tree = build_local_tree(
            state,
            action_model=self.action_model,
            rules=self.rules,
            max_depth=self.config.max_depth,
        )
        solution = solve_subgame(
            tree,
            hero=hero,
            hero_range=hero_range,
            opponent_range=opponent_range,
            rules=self.rules,
            budget_ms=budget_ms,
            num_runouts=self.config.leaf_rollouts,
            max_iterations=self.config.max_iterations,
        )
        return tree.root.actions, solution

    def _blueprint_strategy_matrix(self, state: GameState, actions: list[Action]) -> np.ndarray:
        """Per-combo blueprint strategy over ``actions`` at ``state``.

        The blueprint's distribution depends on the actor's hole cards only
        through the card-abstraction bucket, so distinct combos sharing a bucket
        share a row — cached by infoset key (a few hundred lookups, not 1326).
        Board-blocked combos keep a uniform placeholder row.
        """
        actor = state.current_player
        blocked = blocked_combos(state.board)
        matrix = np.full((NUM_COMBOS, len(actions)), 1.0 / len(actions), dtype=np.float64)
        cache: dict[object, np.ndarray] = {}
        for idx, combo in enumerate(ALL_COMBOS):
            if blocked[idx]:
                continue
            hypo_state = replace_actor_hole_cards(state, actor=actor, combo=combo)
            key = encode_infoset_key(hypo_state, actor, self.blueprint.card_abstraction)
            row = cache.get(key)
            if row is None:
                row = self._blueprint_strategy(hypo_state, actions, use_average=True)
                cache[key] = row
            matrix[idx] = row
        return matrix

    def _blend_matrix(
        self, resolver_matrix: np.ndarray, blueprint_matrix: np.ndarray
    ) -> np.ndarray:
        """Row-wise :meth:`_blend_strategies`: blend, floor, renormalize."""
        if resolver_matrix.shape != blueprint_matrix.shape:
            raise ValueError("Resolver and blueprint strategy matrices must have the same shape.")

        alpha = float(np.clip(self.config.policy_blend_alpha, 0.0, 1.0))
        mixed = ((1.0 - alpha) * blueprint_matrix) + (alpha * resolver_matrix)
        floor = max(0.0, float(self.config.min_strategy_prob))
        if floor > 0:
            mixed = np.maximum(mixed, floor)

        totals = mixed.sum(axis=1, keepdims=True)
        uniform = np.full(mixed.shape[1], 1.0 / mixed.shape[1])
        return np.where(totals > NORMALIZE_EPS, mixed / np.maximum(totals, NORMALIZE_EPS), uniform)

    def _blueprint_strategy(
        self,
        state: GameState,
        actions: list[Action],
        *,
        use_average: bool,
    ) -> np.ndarray:
        """Blueprint probabilities over ``actions``, which may repeat an action.

        A repeated action's blueprint mass is split evenly across its slots so
        the vector still sums to 1 over the caller's action list.
        """
        if not actions:
            return np.array([], dtype=np.float64)

        infoset_key = encode_infoset_key(
            state, state.current_player, self.blueprint.card_abstraction
        )
        infoset = self.blueprint.storage.get_infoset(infoset_key)
        distribution = blueprint_action_distribution(
            infoset, state, self.rules, actions, use_average=use_average
        )
        if distribution is None:
            return np.full(len(actions), 1.0 / len(actions), dtype=np.float64)

        slot_counts: dict[Action, int] = {}
        for action in actions:
            slot_counts[action] = slot_counts.get(action, 0) + 1
        probs = np.array(
            [distribution.get(action, 0.0) / slot_counts[action] for action in actions],
            dtype=np.float64,
        )

        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= NORMALIZE_EPS:
            return np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
        return probs / total

    def _blend_strategies(
        self,
        resolver_strategy: np.ndarray,
        blueprint_strategy: np.ndarray,
    ) -> np.ndarray:
        """Single-row :meth:`_blend_matrix`."""
        return self._blend_matrix(resolver_strategy[None, :], blueprint_strategy[None, :])[0]
