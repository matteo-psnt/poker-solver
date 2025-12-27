"""
Base solver class for CFR algorithms.

Defines the interface that all CFR variants must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from src.actions.betting_actions import BettingActions
from src.bucketing.base import BucketingStrategy
from src.bucketing.utils.infoset import InfoSetKey
from src.solver.storage import Storage
from src.utils.config import Config


class BaseSolver(ABC):
    """
    Abstract base class for CFR solvers.

    All CFR variants (vanilla CFR, CFR+, MCCFR, etc.) should inherit from this.
    """

    def __init__(
        self,
        action_abstraction: BettingActions,
        card_abstraction: BucketingStrategy,
        storage: Storage,
        config: Optional[Config] = None,
    ):
        """
        Initialize solver.

        Args:
            action_abstraction: Action abstraction for betting
            card_abstraction: Card abstraction for bucketing
            storage: Storage backend for infosets
            config: Config object (defaults to Config.default() if not provided)
        """
        self.action_abstraction = action_abstraction
        self.card_abstraction = card_abstraction
        self.storage = storage
        self.config = config if config is not None else Config.default()

        # Training state
        self.iteration = 0
        self.total_utility = 0.0

    @abstractmethod
    def train_iteration(self) -> float:
        """
        Execute one training iteration.

        Returns:
            Utility for player 0
        """
        pass

    def train(self, num_iterations: int, verbose: bool = True) -> Dict:
        """
        Run multiple training iterations.

        Args:
            num_iterations: Number of iterations to run
            verbose: Print progress

        Returns:
            Training statistics
        """
        start_iteration = self.iteration
        log_frequency = max(1, self.config.training.log_frequency)

        for i in range(num_iterations):
            utility = self.train_iteration()
            self.total_utility += utility

            if verbose and (i + 1) % log_frequency == 0:
                avg_utility = self.total_utility / (i + 1)
                print(f"Iteration {self.iteration}: avg utility = {avg_utility:.6f}")

        end_iteration = self.iteration

        return {
            "start_iteration": start_iteration,
            "end_iteration": end_iteration,
            "total_iterations": num_iterations,
            "final_avg_utility": self.total_utility / num_iterations if num_iterations > 0 else 0,
        }

    def get_average_strategy(self, infoset_key: InfoSetKey) -> np.ndarray:
        """
        Get average strategy for an infoset.

        Args:
            infoset_key: Information set key

        Returns:
            Probability distribution over actions
        """
        infoset = self.storage.get_infoset(infoset_key)
        if infoset is None:
            raise ValueError(f"Infoset not found: {infoset_key}")

        return infoset.get_average_strategy()

    def get_current_strategy(self, infoset_key: InfoSetKey) -> np.ndarray:
        """
        Get current strategy (from regret matching) for an infoset.

        Args:
            infoset_key: Information set key

        Returns:
            Probability distribution over actions
        """
        infoset = self.storage.get_infoset(infoset_key)
        if infoset is None:
            raise ValueError(f"Infoset not found: {infoset_key}")

        return infoset.get_strategy()

    def checkpoint(self):
        """Save a checkpoint of the current solver state."""
        self.storage.checkpoint(self.iteration)

    def num_infosets(self) -> int:
        """Get total number of infosets discovered."""
        return self.storage.num_infosets()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(iteration={self.iteration}, infosets={self.num_infosets()})"
        )
