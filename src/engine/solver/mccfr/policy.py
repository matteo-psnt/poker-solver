"""Blueprint-policy action selection helpers for MCCFR solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import GameState
from src.engine.solver.infoset_encoder import encode_infoset_key

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
    if infoset is None:
        return legal_actions[np.random.choice(len(legal_actions))]

    legal_set = set(legal_actions)
    valid_indices = [i for i, action in enumerate(infoset.legal_actions) if action in legal_set]
    if not valid_indices:
        return legal_actions[np.random.choice(len(legal_actions))]

    filtered_actions = [infoset.legal_actions[i] for i in valid_indices]
    strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=use_average)
    action_idx = int(np.random.choice(len(filtered_actions), p=strategy))
    return filtered_actions[action_idx]


def act(
    self: MCCFRSolver,
    state: GameState,
    *,
    use_resolver: bool | None = None,
    time_budget_ms: int | None = None,
    use_average: bool = True,
) -> Action:
    """Choose an action from blueprint policy or runtime subgame resolver."""
    if use_resolver is None:
        use_resolver = self.config.resolver.enabled

    if use_resolver:
        # Lazy import to avoid circular dependency when training imports solver.
        from src.engine.search.resolver import HUResolver

        resolver = getattr(self, "_resolver", None)
        if resolver is None:
            resolver = HUResolver(
                blueprint=self,
                action_model=self.action_model,
                rules=self.rules,
                config=self.config.resolver,
            )
            self._resolver = resolver
        return resolver.act(state, time_budget_ms=time_budget_ms)

    return sample_action_from_strategy(self, state, use_average=use_average)
