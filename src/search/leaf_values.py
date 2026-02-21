"""Leaf-value estimation for depth-limited resolving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.search.tree_builder import LocalTreeNode


@dataclass
class LeafValueConfig:
    """Rollout settings for blueprint leaf estimation."""

    num_rollouts: int = 8
    use_average_strategy: bool = True


def estimate_leaf_values(
    leaves: list[LocalTreeNode],
    *,
    blueprint,
    traversing_player: int,
    config: LeafValueConfig | None = None,
) -> dict[int, float]:
    """
    Estimate values for each leaf using blueprint rollouts.
    """
    if config is None:
        config = LeafValueConfig()

    values: dict[int, float] = {}
    for i, leaf in enumerate(leaves):
        total = 0.0
        for _ in range(config.num_rollouts):
            total += _rollout_with_blueprint(
                blueprint,
                leaf.state,
                traversing_player=traversing_player,
                use_average_strategy=config.use_average_strategy,
            )
        values[i] = total / config.num_rollouts
    return values


def _rollout_with_blueprint(
    blueprint,
    state,
    *,
    traversing_player: int,
    use_average_strategy: bool,
) -> float:
    if state.is_terminal:
        if len(state.board) < 5:
            state = blueprint._deal_remaining_cards(state)
        return float(state.get_payoff(traversing_player, blueprint.rules))

    if blueprint._is_chance_node(state):
        next_state = blueprint._sample_chance_outcome(state)
        return _rollout_with_blueprint(
            blueprint,
            next_state,
            traversing_player=traversing_player,
            use_average_strategy=use_average_strategy,
        )

    action = blueprint.sample_action_from_strategy(state, use_average=use_average_strategy)
    next_state = state.apply_action(action, blueprint.rules)
    return _rollout_with_blueprint(
        blueprint,
        next_state,
        traversing_player=traversing_player,
        use_average_strategy=use_average_strategy,
    )


def expected_value_vector(values: dict[int, float], size: int) -> np.ndarray:
    """Convert sparse leaf value mapping to dense vector."""
    out = np.zeros(size, dtype=np.float64)
    for idx, value in values.items():
        if 0 <= idx < size:
            out[idx] = value
    return out
