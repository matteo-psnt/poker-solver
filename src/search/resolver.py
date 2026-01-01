"""HU runtime subgame resolver."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from src.actions.action_model import ActionModel
from src.game.actions import Action
from src.game.rules import GameRules
from src.game.state import GameState
from src.search.fast_cfr import solve_root_strategy
from src.search.leaf_values import LeafValueConfig, estimate_leaf_values
from src.search.range_inference import PlayerRanges, infer_ranges, update_ranges
from src.search.tree_builder import LocalTreeNode, build_local_tree
from src.solver.infoset_encoder import encode_infoset_key
from src.utils.config import ResolverConfig


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
        """Run local resolve and return root strategy and sampled action."""
        budget = int(time_budget_ms or self.config.time_budget_ms)
        self._ranges = (
            infer_ranges(state, self.blueprint, mode=self.config.range_update_mode)
            if self._ranges is None
            else self._ranges
        )

        tree = build_local_tree(
            state,
            action_model=self.action_model,
            rules=self.rules,
            max_depth=self.config.max_depth,
        )

        root_actions = tree.root.actions
        if not root_actions:
            raise ValueError("Resolver found no legal root actions.")

        if self.config.leaf_value_mode != "blueprint_rollout":
            raise ValueError(
                f"Unsupported resolver leaf_value_mode: {self.config.leaf_value_mode}. "
                "Only 'blueprint_rollout' is supported."
            )

        leaves = tree.leaves
        leaf_values = estimate_leaf_values(
            leaves,
            blueprint=self.blueprint,
            traversing_player=state.current_player,
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
            mode=self.config.range_update_mode,
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
    ) -> float:
        if node.is_leaf or not node.children:
            return float(leaf_value_by_node_id.get(id(node), 0.0))

        child_values = np.array(
            [
                self._node_expected_value(
                    child,
                    leaf_value_by_node_id=leaf_value_by_node_id,
                    use_average=use_average,
                )
                for child in node.children
            ],
            dtype=np.float64,
        )
        strategy = self._blueprint_strategy(node.state, node.actions, use_average=use_average)
        return float(np.dot(strategy, child_values))

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
