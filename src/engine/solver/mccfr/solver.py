"""Monte Carlo Counterfactual Regret Minimization (MCCFR) solver."""

from __future__ import annotations

import random

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import GameState
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.base import Storage
from src.shared.config import Config

from . import chance, policy, traversal


class MCCFRSolver:
    """
    Monte Carlo CFR with external sampling or outcome sampling.

    External sampling (default):
    - Explores all actions for traversing player
    - Samples single action for opponent
    - Samples chance outcomes

    Outcome sampling:
    - Samples single action for all players
    - Samples chance outcomes
    - Faster but higher variance
    """

    def __init__(
        self,
        action_model: ActionModel,
        card_abstraction: BucketingStrategy,
        storage: Storage,
        config: Config,
    ):
        self.action_model = action_model
        self.card_abstraction = card_abstraction
        self.storage = storage
        self.config = config

        self.iteration = 0
        self.rules = GameRules(self.config.game.small_blind, self.config.game.big_blind)

        if self.config.system.seed is not None:
            random.seed(self.config.system.seed)
            np.random.seed(self.config.system.seed)

    def checkpoint(self) -> None:
        """Save a checkpoint of the current solver state."""
        self.storage.checkpoint(self.iteration)

    def num_infosets(self) -> int:
        """Get total number of infosets discovered."""
        return self.storage.num_infosets()

    def train_iteration(self) -> float:
        """Execute one MCCFR iteration using configured sampling method."""
        state = self.deal_initial_state()
        traversing_player = self.iteration % 2

        if self.config.solver.sampling_method == "external":
            util = self._cfr_external_sampling(state, traversing_player, [1.0, 1.0])
        else:
            util = self._cfr_outcome_sampling(state, traversing_player, [1.0, 1.0])

        self.iteration += 1
        if traversing_player == 1:
            util = -util
        return util

    def deal_initial_state(self) -> GameState:
        return chance.deal_initial_state(self)

    def is_chance_node(self, state: GameState) -> bool:
        return chance.is_chance_node(self, state)

    def sample_chance_outcome(self, state: GameState) -> GameState:
        return chance.sample_chance_outcome(self, state)

    def deal_remaining_cards(self, state: GameState) -> GameState:
        return chance.deal_remaining_cards(self, state)

    def _cfr_external_sampling(
        self,
        state: GameState,
        traversing_player: int,
        reach_probs: list[float],
    ) -> float:
        return traversal.cfr_external_sampling(self, state, traversing_player, reach_probs)

    def _cfr_outcome_sampling(
        self,
        state: GameState,
        traversing_player: int,
        reach_probs: list[float],
    ) -> float:
        return traversal.cfr_outcome_sampling(self, state, traversing_player, reach_probs)

    def sample_action_from_strategy(self, state: GameState, *, use_average: bool = True) -> Action:
        return policy.sample_action_from_strategy(self, state, use_average=use_average)

    def __str__(self) -> str:
        return (
            f"MCCFRSolver(iteration={self.iteration}, infosets={self.num_infosets()}, "
            f"sampling={self.config.solver.sampling_method}, "
            f"stack={self.config.game.starting_stack})"
        )
