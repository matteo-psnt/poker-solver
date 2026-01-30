"""Blueprint-policy action selection helpers for MCCFR solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import GameState
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.engine.solver.policy_lookup import blueprint_action_distribution

if TYPE_CHECKING:
    from .solver import MCCFRSolver


def sample_action_from_strategy(
    self: MCCFRSolver,
    state: GameState,
    *,
    use_average: bool = True,
) -> Action:
    """Sample an action from the blueprint strategy at the current infoset."""
    legal_actions = self.rules.get_legal_actions(state, action_model=self.action_model)
    if not legal_actions:
        raise ValueError(f"No legal actions at state: {state}")

    infoset_key = encode_infoset_key(state, state.current_player, self.card_abstraction)
    infoset = self.storage.get_infoset(infoset_key)
    distribution = blueprint_action_distribution(
        infoset, state, self.rules, legal_actions, use_average=use_average
    )
    if distribution is None:
        return legal_actions[np.random.choice(len(legal_actions))]

    actions = list(distribution)
    probabilities = np.fromiter(distribution.values(), dtype=np.float64, count=len(actions))
    action_idx = int(np.random.choice(len(actions), p=probabilities))
    return actions[action_idx]
