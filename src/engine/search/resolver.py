"""HU runtime subgame resolver."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import GameState
from src.engine.search.fast_cfr import solve_root_strategy
from src.engine.search.leaf_values import LeafValueConfig, estimate_leaf_values
from src.engine.search.range_inference import (
    ALL_COMBOS,
    COMBO_MASKS,
    PlayerRanges,
    infer_ranges,
    replace_actor_hole_cards,
    update_ranges,
)
from src.engine.search.tree_builder import LocalTreeNode, build_local_tree
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.shared.config import ResolverConfig


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
        self._ranges: PlayerRanges | None = None

    def act(self, state: GameState, *, time_budget_ms: int | None = None) -> Action:
        """Resolve local subgame and sample an action at the root."""
        budget = int(time_budget_ms or self.config.time_budget_ms)
        try:
            result = self.solve(state, time_budget_ms=budget)
            return result.action
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            warnings.warn(
                f"Resolver failed at runtime, falling back to blueprint strategy: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.blueprint.sample_action_from_strategy(state, use_average=True)

    def solve(self, state: GameState, *, time_budget_ms: int | None = None) -> ResolveResult:
        """Run local resolve and return root strategy and sampled action.

        All valuation is range-honest: the opponent's dealt cards in ``state``
        are never consulted — leaf rollouts resample the opponent's hand from
        the tracked posterior range, and opponent decision nodes inside the tree
        use the range-aggregate blueprint distribution.
        """
        budget = int(time_budget_ms or self.config.time_budget_ms)
        self._ranges = infer_ranges(state, self.blueprint) if self._ranges is None else self._ranges

        hero = state.current_player
        opponent_range = self._ranges.p1 if hero == 0 else self._ranges.p0

        tree = build_local_tree(
            state,
            action_model=self.action_model,
            rules=self.rules,
            max_depth=self.config.max_depth,
        )

        root_actions = tree.root.actions
        if not root_actions:
            raise ValueError("Resolver found no legal root actions.")

        leaves = tree.leaves
        leaf_values = estimate_leaf_values(
            leaves,
            blueprint=self.blueprint,
            traversing_player=hero,
            opponent_range=opponent_range,
            config=LeafValueConfig(
                num_rollouts=self.config.leaf_rollouts,
                use_average_strategy=self.config.leaf_use_average_strategy,
            ),
        )
        leaf_value_by_node_id = {id(leaf): leaf_values[i] for i, leaf in enumerate(leaves)}

        action_values = np.array(
            [
                self._node_expected_value(
                    child,
                    leaf_value_by_node_id=leaf_value_by_node_id,
                    use_average=self.config.leaf_use_average_strategy,
                    traversing_player=hero,
                    opponent_range=opponent_range,
                )
                for child in tree.root.children
            ],
            dtype=np.float64,
        )
        resolver_strategy = solve_root_strategy(action_values, budget_ms=budget)
        blueprint_strategy = self._blueprint_strategy(
            state,
            root_actions,
            use_average=self.config.leaf_use_average_strategy,
        )
        strategy = self._blend_strategies(resolver_strategy, blueprint_strategy)

        idx = int(np.random.choice(len(root_actions), p=strategy))
        chosen_action = root_actions[idx]

        self._ranges = update_ranges(
            state,
            self._ranges,
            chosen_action,
            self.blueprint,
        )
        return ResolveResult(
            action=chosen_action,
            root_actions=root_actions,
            strategy=strategy,
            resolver_strategy=resolver_strategy,
            blueprint_strategy=blueprint_strategy,
            action_values=action_values,
        )

    def _node_expected_value(
        self,
        node: LocalTreeNode,
        *,
        leaf_value_by_node_id: dict[int, float],
        use_average: bool,
        traversing_player: int,
        opponent_range: np.ndarray,
    ) -> float:
        if node.is_leaf or not node.children:
            return float(leaf_value_by_node_id.get(id(node), 0.0))

        child_values = np.array(
            [
                self._node_expected_value(
                    child,
                    leaf_value_by_node_id=leaf_value_by_node_id,
                    use_average=use_average,
                    traversing_player=traversing_player,
                    opponent_range=opponent_range,
                )
                for child in node.children
            ],
            dtype=np.float64,
        )
        if node.state.current_player == traversing_player:
            # Hero's own future node: hero knows its cards, direct lookup is honest.
            strategy = self._blueprint_strategy(node.state, node.actions, use_average=use_average)
        else:
            strategy = self._range_aggregate_strategy(
                node.state,
                node.actions,
                use_average=use_average,
                traversing_player=traversing_player,
                opponent_range=opponent_range,
            )
        return float(np.dot(strategy, child_values))

    def _range_aggregate_strategy(
        self,
        state: GameState,
        actions: list[Action],
        *,
        use_average: bool,
        traversing_player: int,
        opponent_range: np.ndarray,
    ) -> np.ndarray:
        """Opponent action distribution aggregated over its posterior range.

        The opponent's dealt cards are hidden information; predicting its play
        from the true-hand infoset would be clairvoyant. Instead, aggregate the
        blueprint distribution over every combo the hero cannot exclude, weighted
        by the posterior range: ``P(a) = Σ_c w(c) · σ(a | c)``. Combos sharing a
        bucket share an infoset, so the lookup is cached per bucket.
        """
        if not actions:
            return np.array([], dtype=np.float64)

        opponent = state.current_player
        known = 0
        for card in state.hole_cards[traversing_player]:
            known |= card.mask
        for card in state.board:
            known |= card.mask

        weights = np.where((COMBO_MASKS & known) == 0, np.maximum(opponent_range, 0.0), 0.0)
        if weights.sum() <= 1e-12:
            weights = np.where((COMBO_MASKS & known) == 0, 1.0, 0.0)

        aggregate = np.zeros(len(actions), dtype=np.float64)
        cache: dict[object, np.ndarray] = {}
        total = 0.0
        for idx in np.nonzero(weights)[0]:
            hypo_state = replace_actor_hole_cards(state, actor=opponent, combo=ALL_COMBOS[idx])
            key = encode_infoset_key(hypo_state, opponent, self.blueprint.card_abstraction)
            dist = cache.get(key)
            if dist is None:
                dist = self._blueprint_strategy(hypo_state, actions, use_average=use_average)
                cache[key] = dist
            weight = float(weights[idx])
            aggregate += weight * dist
            total += weight

        if total <= 1e-12:
            return np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
        aggregate /= aggregate.sum()
        return aggregate

    def _blueprint_strategy(
        self,
        state: GameState,
        actions: list[Action],
        *,
        use_average: bool,
    ) -> np.ndarray:
        if not actions:
            return np.array([], dtype=np.float64)

        infoset_key = encode_infoset_key(
            state, state.current_player, self.blueprint.card_abstraction
        )
        infoset = self.blueprint.storage.get_infoset(infoset_key)
        if infoset is None:
            return np.full(len(actions), 1.0 / len(actions), dtype=np.float64)

        action_to_indices: dict[Action, list[int]] = {}
        for idx, action in enumerate(actions):
            action_to_indices.setdefault(action, []).append(idx)

        valid_indices: list[int] = []
        valid_actions: list[Action] = []
        for idx, action in enumerate(infoset.legal_actions):
            if action in action_to_indices and self.rules.is_action_valid(state, action):
                valid_indices.append(idx)
                valid_actions.append(action)

        if not valid_indices:
            return np.full(len(actions), 1.0 / len(actions), dtype=np.float64)

        filtered = infoset.get_filtered_strategy(
            valid_indices=valid_indices, use_average=use_average
        )
        probs = np.zeros(len(actions), dtype=np.float64)
        for action, prob in zip(valid_actions, filtered):
            target_indices = action_to_indices.get(action, [])
            if not target_indices:
                continue
            share = float(prob) / len(target_indices)
            for idx in target_indices:
                probs[idx] += share

        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 1e-12:
            return np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
        return probs / total

    def _blend_strategies(
        self,
        resolver_strategy: np.ndarray,
        blueprint_strategy: np.ndarray,
    ) -> np.ndarray:
        if len(resolver_strategy) != len(blueprint_strategy):
            raise ValueError("Resolver and blueprint strategy vectors must have the same length.")

        alpha = float(np.clip(self.config.policy_blend_alpha, 0.0, 1.0))
        mixed = ((1.0 - alpha) * blueprint_strategy) + (alpha * resolver_strategy)
        floor = max(0.0, float(self.config.min_strategy_prob))
        if floor > 0:
            mixed = np.maximum(mixed, floor)

        total = mixed.sum()
        if total <= 1e-12:
            return np.full(len(mixed), 1.0 / len(mixed), dtype=np.float64)
        return mixed / total
