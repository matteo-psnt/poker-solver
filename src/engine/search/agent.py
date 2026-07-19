"""Runtime decision agent: blueprint policy plus optional subgame resolver.

Deployment-time action selection used to live on ``MCCFRSolver.act``, which
made the trainer double as the deployed player and forced a lazy
solver -> search import to reach the resolver. The agent inverts that: it
wraps any :class:`~src.engine.solver.protocols.Blueprint` and owns the
resolver lifecycle, so playing a strategy never requires carrying (or
importing) the training machinery.
"""

from __future__ import annotations

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import GameState
from src.engine.search.resolver import HUResolver
from src.engine.solver.protocols import Blueprint
from src.shared.config import ResolverConfig


class BlueprintAgent:
    """Plays a trained blueprint, optionally through the runtime resolver.

    The resolver decision is made once at construction: ``use_resolver=None``
    defers to ``resolver_config.enabled`` (deployment's switch). One resolver
    instance persists across the agent's lifetime — feed every realized action
    (both seats) through :meth:`observe` so its range inference tracks the
    hand; drop the agent (or build a fresh one) between hands.

    ``rng`` drives the resolver's leaf-runout sampling and is therefore part of
    the played strategy: evaluation harnesses pin it per hand (fresh agent,
    fresh generator) so results are reproducible and worker-count independent.
    """

    def __init__(
        self,
        blueprint: Blueprint,
        *,
        use_resolver: bool | None = None,
        resolver_config: ResolverConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.blueprint = blueprint
        config = resolver_config if resolver_config is not None else blueprint.config.resolver
        if use_resolver is None:
            use_resolver = config.enabled
        self.resolver: HUResolver | None = (
            HUResolver(
                blueprint=blueprint,
                action_model=blueprint.action_model,
                rules=blueprint.rules,
                config=config,
                rng=rng,
            )
            if use_resolver
            else None
        )

    def act(
        self,
        state: GameState,
        *,
        time_budget_ms: int | None = None,
        use_average: bool = True,
    ) -> Action:
        """Choose an action from the resolver (when armed) or the blueprint."""
        if self.resolver is not None:
            return self.resolver.act(state, time_budget_ms=time_budget_ms)
        return self.blueprint.sample_action_from_strategy(state, use_average=use_average)

    def observe(self, state: GameState, action: Action) -> None:
        """Notify the resolver's range inference of a realized action."""
        if self.resolver is not None:
            self.resolver.observe(state, action)
