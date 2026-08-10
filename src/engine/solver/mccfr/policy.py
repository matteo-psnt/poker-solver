"""Blueprint-policy action selection helpers for MCCFR solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import GameState
from src.engine.solver.policy.lookup import blueprint_action_distribution

if TYPE_CHECKING:
    from src.engine.solver.policy.source import ScorableBlueprint


def sample_action_from_strategy(
    self: ScorableBlueprint,
    state: GameState,
    *,
    use_average: bool = True,
) -> Action:
    """Sample an action from the blueprint strategy at the current infoset."""
    legal_actions = self.rules.get_legal_actions(state, action_model=self.action_model)
    if not legal_actions:
        raise ValueError(f"No legal actions at state: {state}")

    # Through the policy source, not storage directly: this is the sampling path
    # the Blueprint protocol exposes, and StaticTreeSolver inherits it. Reaching
    # for a key here would make a tree-addressed blueprint unplayable while every
    # other runtime path worked.
    source = self.policy_source
    infoset = source.infoset_at(state, source.bucket_for(state, state.current_player))
    distribution = blueprint_action_distribution(
        infoset, state, self.rules, legal_actions, use_average=use_average
    )
    if distribution is None:
        return legal_actions[np.random.choice(len(legal_actions))]

    actions = list(distribution)
    probabilities = np.fromiter(distribution.values(), dtype=np.float64, count=len(actions))
    action_idx = int(np.random.choice(len(actions), p=probabilities))
    return actions[action_idx]
