"""Off-tree action translation utilities."""

from __future__ import annotations

import numpy as np

from src.actions.action_model import ActionModel
from src.game.actions import Action, ActionType
from src.game.state import GameState


def translate_action_distribution(
    state: GameState,
    observed_action: Action,
    action_model: ActionModel,
) -> list[tuple[Action, float]]:
    """
    Map an observed action to a distribution over legal abstract actions.

    - nearest: deterministic nearest mapping
    - probabilistic: pseudo-harmonic interpolation between adjacent sizes
      for off-tree aggressive actions
    """
    legal_actions = action_model.get_legal_actions(state)
    if not legal_actions:
        return [(observed_action, 1.0)]

    if observed_action in legal_actions:
        return [(observed_action, 1.0)]

    mode = action_model.off_tree_mapping
    if mode != "probabilistic":
        return [(action_model.discretize_action(state, observed_action), 1.0)]

    if observed_action.type not in {ActionType.BET, ActionType.RAISE}:
        return [(action_model.discretize_action(state, observed_action), 1.0)]

    target_type = observed_action.type
    candidates = [a for a in legal_actions if a.type == target_type and a.amount > 0]
    if len(candidates) < 2:
        candidates = [a for a in legal_actions if a.is_aggressive() and a.amount > 0]
    if len(candidates) < 2:
        return [(action_model.discretize_action(state, observed_action), 1.0)]

    candidates = sorted(candidates, key=lambda a: a.amount)
    target_amount = observed_action.amount

    if target_amount <= candidates[0].amount:
        return [(candidates[0], 1.0)]
    if target_amount >= candidates[-1].amount:
        return [(candidates[-1], 1.0)]

    lower = candidates[0]
    upper = candidates[-1]
    for idx in range(1, len(candidates)):
        if candidates[idx].amount >= target_amount:
            lower = candidates[idx - 1]
            upper = candidates[idx]
            break

    span = upper.amount - lower.amount
    if span <= 0:
        return [(lower, 1.0)]

    upper_w = (target_amount - lower.amount) / span
    lower_w = 1.0 - upper_w
    lower_w = float(np.clip(lower_w, 0.0, 1.0))
    upper_w = float(np.clip(upper_w, 0.0, 1.0))
    total = lower_w + upper_w
    if total <= 0:
        return [(action_model.discretize_action(state, observed_action), 1.0)]
    lower_w /= total
    upper_w /= total
    return [(lower, lower_w), (upper, upper_w)]


def translate_off_tree_action(
    state: GameState,
    observed_action: Action,
    action_model: ActionModel,
    *,
    rng: np.random.Generator | None = None,
) -> Action:
    """Translate an observed action to one legal abstract action."""
    dist = translate_action_distribution(state, observed_action, action_model)
    if len(dist) == 1:
        return dist[0][0]
    if rng is None:
        rng = np.random.default_rng()
    probs = np.array([p for _, p in dist], dtype=np.float64)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return dist[0][0]
    probs /= probs_sum
    idx = int(rng.choice(len(dist), p=probs))
    return dist[idx][0]
