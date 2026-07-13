"""Leaf-value estimation for depth-limited resolving.

Honesty contract: the resolver must not condition on the opponent's actual hole
cards — a real player cannot see them. Every rollout therefore *resamples* the
opponent's hand from the caller-supplied posterior range (masked against the
cards the traversing player can see), and the rollout — including the
opponent's blueprint decisions and the showdown payoff — is played with that
sampled hand. Averaging over rollouts integrates the opponent's hand out.

(Before 2026-07-13 rollouts kept the dealt opponent hand from the game state,
which made leaf values clairvoyant and inflated the resolver's measured edge.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.engine.search.range_inference import (
    ALL_COMBOS,
    COMBO_MASKS,
    NUM_COMBOS,
    replace_actor_hole_cards,
)
from src.engine.search.tree_builder import LocalTreeNode

_EPS = 1e-12


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
    opponent_range: np.ndarray,
    config: LeafValueConfig | None = None,
) -> dict[int, float]:
    """Estimate each leaf's value for ``traversing_player`` via blueprint rollouts.

    ``opponent_range`` is a weight vector over
    :data:`~src.engine.search.range_inference.ALL_COMBOS` (the resolver's
    posterior for the opponent). Each rollout draws the opponent's hand from it,
    conditioned on the cards the traversing player can see at the leaf.
    """
    if config is None:
        config = LeafValueConfig()

    opponent = 1 - traversing_player
    values: dict[int, float] = {}
    for i, leaf in enumerate(leaves):
        weights = _visible_masked_range(leaf.state, traversing_player, opponent_range)
        total = 0.0
        for _ in range(config.num_rollouts):
            combo_idx = int(np.random.choice(NUM_COMBOS, p=weights))
            sampled_state = replace_actor_hole_cards(
                leaf.state, actor=opponent, combo=ALL_COMBOS[combo_idx]
            )
            total += _rollout_with_blueprint(
                blueprint,
                sampled_state,
                traversing_player=traversing_player,
                use_average_strategy=config.use_average_strategy,
            )
        values[i] = total / config.num_rollouts
    return values


def _visible_masked_range(state, traversing_player: int, opponent_range: np.ndarray) -> np.ndarray:
    """Opponent range masked by what the traversing player can see, normalized."""
    known = 0
    for card in state.hole_cards[traversing_player]:
        known |= card.mask
    for card in state.board:
        known |= card.mask

    weights = np.where((COMBO_MASKS & known) == 0, np.maximum(opponent_range, 0.0), 0.0)
    total = weights.sum()
    if total <= _EPS:
        weights = np.where((COMBO_MASKS & known) == 0, 1.0, 0.0)
        total = weights.sum()
    return weights / total


def _rollout_with_blueprint(
    blueprint,
    state,
    *,
    traversing_player: int,
    use_average_strategy: bool,
) -> float:
    if state.is_terminal:
        if len(state.board) < 5:
            state = blueprint.deal_remaining_cards(state)
        return float(state.get_payoff(traversing_player, blueprint.rules))

    if blueprint.is_chance_node(state):
        next_state = blueprint.sample_chance_outcome(state)
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
