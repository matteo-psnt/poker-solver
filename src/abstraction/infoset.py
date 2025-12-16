"""
Information set representation for CFR.

An information set groups together all game states that are indistinguishable
to a player given their information (hole cards, board, betting history).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.game.actions import Action
from src.game.state import Street


@dataclass(frozen=True)
class InfoSetKey:
    """
    Hashable identifier for an information set.

    This key uniquely identifies an information set by abstracting:
    - Player's position (button vs non-button)
    - Current street
    - Betting sequence (normalized action history)
    - Card bucket (abstracted hand strength)
    - SPR bucket (stack-to-pot ratio category)

    Being frozen makes it hashable and suitable as a dictionary key.
    """

    player_position: int  # 0 or 1
    street: Street
    betting_sequence: str  # Normalized betting history (e.g., "f", "c", "b0.75-c")
    card_bucket: int  # Abstracted hand strength bucket
    spr_bucket: int  # Stack-to-pot ratio bucket (0=shallow, 1=medium, 2=deep)

    def __hash__(self) -> int:
        """Hash for dictionary storage."""
        return hash((
            self.player_position,
            self.street,
            self.betting_sequence,
            self.card_bucket,
            self.spr_bucket,
        ))

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, InfoSetKey):
            return False
        return (
            self.player_position == other.player_position
            and self.street == other.street
            and self.betting_sequence == other.betting_sequence
            and self.card_bucket == other.card_bucket
            and self.spr_bucket == other.spr_bucket
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"InfoSet(P{self.player_position}|{self.street.name}|"
            f"seq={self.betting_sequence}|bucket={self.card_bucket}|spr={self.spr_bucket})"
        )

    def __repr__(self) -> str:
        return (
            f"InfoSetKey(player_position={self.player_position}, "
            f"street={self.street}, betting_sequence='{self.betting_sequence}', "
            f"card_bucket={self.card_bucket}, spr_bucket={self.spr_bucket})"
        )


class InfoSet:
    """
    Information set with associated CFR data.

    An information set stores:
    - The infoset key (identifier)
    - Legal actions at this infoset
    - Cumulative regrets (for regret matching)
    - Cumulative strategy (for computing average strategy)
    """

    def __init__(self, key: InfoSetKey, legal_actions: List[Action]):
        """
        Initialize information set.

        Args:
            key: InfoSetKey identifier
            legal_actions: List of legal actions at this infoset
        """
        self.key = key
        self.legal_actions = legal_actions
        self.num_actions = len(legal_actions)

        # CFR data structures
        self.regrets = np.zeros(self.num_actions, dtype=np.float32)
        self.strategy_sum = np.zeros(self.num_actions, dtype=np.float32)

        # Iteration counter for this infoset
        self.reach_count = 0

    def get_strategy(self) -> np.ndarray:
        """
        Compute current strategy via regret matching.

        The regret matching algorithm:
        1. Take positive regrets (floor at 0)
        2. Normalize to sum to 1
        3. If all regrets <= 0, use uniform strategy

        Returns:
            Probability distribution over actions (sums to 1)
        """
        # Positive regrets only
        positive_regrets = np.maximum(self.regrets, 0)
        sum_positive = np.sum(positive_regrets)

        if sum_positive > 0:
            # Normalize to probability distribution
            strategy = positive_regrets / sum_positive
        else:
            # Uniform strategy if all regrets non-positive
            strategy = np.ones(self.num_actions) / self.num_actions

        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        Compute average strategy over all iterations.

        The average strategy converges to Nash equilibrium.

        Returns:
            Probability distribution over actions (sums to 1)
        """
        sum_strategy = np.sum(self.strategy_sum)

        if sum_strategy > 0:
            # Normalize to probability distribution
            avg_strategy = self.strategy_sum / sum_strategy
        else:
            # Uniform if never updated (shouldn't happen in practice)
            avg_strategy = np.ones(self.num_actions) / self.num_actions

        return avg_strategy

    def update_regret(self, action_idx: int, regret: float):
        """
        Update cumulative regret for an action.

        Args:
            action_idx: Index of action in legal_actions
            regret: Regret value to add (can be negative)
        """
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"Invalid action index: {action_idx}")

        self.regrets[action_idx] += regret

    def update_strategy(self, reach_prob: float):
        """
        Update cumulative strategy (weighted by reach probability).

        Args:
            reach_prob: Probability of reaching this infoset
        """
        strategy = self.get_strategy()
        self.strategy_sum += strategy * reach_prob
        self.reach_count += 1

    def reset_regrets(self):
        """Reset all regrets to zero (for some CFR variants)."""
        self.regrets = np.zeros(self.num_actions, dtype=np.float32)

    def reset_strategy_sum(self):
        """Reset strategy sum to zero (for some CFR variants)."""
        self.strategy_sum = np.zeros(self.num_actions, dtype=np.float32)

    def prune(self, threshold: float = 1e-9):
        """
        Prune very small regrets for memory efficiency.

        Args:
            threshold: Regrets below this (in absolute value) are set to 0
        """
        self.regrets[np.abs(self.regrets) < threshold] = 0
        self.strategy_sum[self.strategy_sum < threshold] = 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        strategy = self.get_strategy()
        avg_strategy = self.get_average_strategy()

        actions_str = ", ".join([str(a) for a in self.legal_actions])
        strategy_str = ", ".join([f"{p:.3f}" for p in strategy])
        avg_str = ", ".join([f"{p:.3f}" for p in avg_strategy])

        return (
            f"InfoSet({self.key})\n"
            f"  Actions: [{actions_str}]\n"
            f"  Current Strategy: [{strategy_str}]\n"
            f"  Average Strategy: [{avg_str}]\n"
            f"  Regrets: {self.regrets}\n"
            f"  Reach Count: {self.reach_count}"
        )

    def __repr__(self) -> str:
        return f"InfoSet(key={repr(self.key)}, num_actions={self.num_actions})"


def create_infoset_key(
    player: int,
    street: Street,
    betting_sequence: str,
    card_bucket: int,
    spr_bucket: int,
) -> InfoSetKey:
    """
    Convenience function to create an InfoSetKey.

    Args:
        player: Player position (0 or 1)
        street: Current street
        betting_sequence: Normalized betting history
        card_bucket: Card abstraction bucket
        spr_bucket: SPR bucket (0=shallow, 1=medium, 2=deep)

    Returns:
        InfoSetKey instance
    """
    return InfoSetKey(
        player_position=player,
        street=street,
        betting_sequence=betting_sequence,
        card_bucket=card_bucket,
        spr_bucket=spr_bucket,
    )
