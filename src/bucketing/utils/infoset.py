"""
Information set representation for CFR.

An information set groups together all game states that are indistinguishable
to a player given their information (hole cards, board, betting history).
"""

from dataclasses import dataclass

import numpy as np

from src.game.actions import Action
from src.game.state import Street
from src.utils.numba_ops import average_strategy, regret_matching


@dataclass(frozen=True)
class InfoSetKey:
    """
    Hashable identifier for an information set.

    This key uniquely identifies an information set by abstracting:
    - Player's position (button vs non-button)
    - Current street
    - Betting sequence (normalized action history)
    - Hand representation (hybrid preflop/postflop)
    - SPR bucket (stack-to-pot ratio category)

    **Hybrid Hand Representation:**
    - Preflop: Uses hand string (e.g., "AKs", "72o", "TT")
              → 169 unique hands, no bucketing
    - Postflop: Uses bucket ID (e.g., 0-49 on flop)
               → Equity-based bucketing

    Being frozen makes it hashable and suitable as a dictionary key.
    """

    player_position: int  # 0 or 1
    street: Street
    betting_sequence: str  # Normalized betting history (e.g., "f", "c", "b0.75-c")

    # Hybrid hand representation
    preflop_hand: str | None  # "AKs", "72o", etc. (None if postflop)
    postflop_bucket: int | None  # 0-49/99/199 (None if preflop)

    spr_bucket: int  # Stack-to-pot ratio bucket (0=shallow, 1=medium, 2=deep)

    def __post_init__(self):
        """Validate that exactly one of preflop_hand or postflop_bucket is set."""
        if self.street == Street.PREFLOP:
            if self.preflop_hand is None:
                raise ValueError("preflop_hand must be set for PREFLOP street")
            if self.postflop_bucket is not None:
                raise ValueError("postflop_bucket must be None for PREFLOP street")
        else:
            if self.postflop_bucket is None:
                raise ValueError(f"postflop_bucket must be set for {self.street.name} street")
            if self.preflop_hand is not None:
                raise ValueError(f"preflop_hand must be None for {self.street.name} street")

    def __hash__(self) -> int:
        """Hash for dictionary storage."""
        return hash(
            (
                self.player_position,
                self.street,
                self.betting_sequence,
                self.preflop_hand,
                self.postflop_bucket,
                self.spr_bucket,
            )
        )

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, InfoSetKey):
            return False
        return (
            self.player_position == other.player_position
            and self.street == other.street
            and self.betting_sequence == other.betting_sequence
            and self.preflop_hand == other.preflop_hand
            and self.postflop_bucket == other.postflop_bucket
            and self.spr_bucket == other.spr_bucket
        )

    def get_hand_repr(self) -> str:
        """Get human-readable hand representation."""
        if self.street == Street.PREFLOP:
            return self.preflop_hand or ""
        else:
            return f"B{self.postflop_bucket}"

    def __str__(self) -> str:
        """Human-readable string representation."""
        hand_repr = self.get_hand_repr()
        return (
            f"InfoSet(P{self.player_position}|{self.street.name}|"
            f"seq={self.betting_sequence}|hand={hand_repr}|spr={self.spr_bucket})"
        )

    def __repr__(self) -> str:
        return (
            f"InfoSetKey(player_position={self.player_position}, "
            f"street={self.street}, betting_sequence='{self.betting_sequence}', "
            f"preflop_hand={self.preflop_hand!r}, postflop_bucket={self.postflop_bucket}, "
            f"spr_bucket={self.spr_bucket})"
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

    def __init__(self, key: InfoSetKey, legal_actions: list[Action]):
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
        self.regrets = np.zeros(self.num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(self.num_actions, dtype=np.float64)

        # Statistics tracking
        self.reach_count = 0  # Number of times this infoset was reached
        self.cumulative_utility = 0.0  # Sum of node utilities (for average)
        self._reach_counts_view: np.ndarray | None = None
        self._cumulative_utility_view: np.ndarray | None = None
        self._stats_index: int | None = None
        self._stats_read_only = False

    def sync_stats_to_storage(
        self,
        reach_count: int,
        cumulative_utility: float,
    ) -> None:
        """Update stats fields from storage-backed arrays."""
        self.reach_count = int(reach_count)
        self.cumulative_utility = float(cumulative_utility)

    def attach_stats_views(
        self,
        reach_counts: np.ndarray,
        cumulative_utilities: np.ndarray,
        infoset_id: int,
        read_only: bool = False,
    ) -> None:
        """Attach shared-memory views for stats updates."""
        self._reach_counts_view = reach_counts
        self._cumulative_utility_view = cumulative_utilities
        self._stats_index = infoset_id
        self._stats_read_only = read_only

    def increment_reach_count(self, delta: int = 1) -> None:
        """Increment reach count for this infoset."""
        if self._reach_counts_view is not None and self._stats_index is not None:
            if not self._stats_read_only:
                self._reach_counts_view[self._stats_index] += delta
            self.reach_count = int(self._reach_counts_view[self._stats_index])
        else:
            self.reach_count += delta

    def add_cumulative_utility(self, value: float) -> None:
        """Add to cumulative utility for this infoset."""
        if self._cumulative_utility_view is not None and self._stats_index is not None:
            if not self._stats_read_only:
                self._cumulative_utility_view[self._stats_index] += value
            self.cumulative_utility = float(self._cumulative_utility_view[self._stats_index])
        else:
            self.cumulative_utility += value

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
        # Use Numba-optimized regret matching (or fallback)
        return regret_matching(self.regrets)

    def get_average_strategy(self) -> np.ndarray:
        """
        Compute average strategy over all iterations.

        The average strategy converges to Nash equilibrium.

        Returns:
            Probability distribution over actions (sums to 1)
        """
        # Use Numba-optimized average strategy computation (or fallback)
        return average_strategy(self.strategy_sum)

    def get_filtered_strategy(
        self, valid_indices: list[int] | None = None, use_average: bool = True
    ) -> np.ndarray:
        """
        Get strategy filtered to valid actions and normalized.

        This method replaces the manual filter-normalize pattern that was
        duplicated across mccfr.py, exploitability.py, and head_to_head.py.

        Args:
            valid_indices: Indices of valid actions to filter to. If None, returns full strategy.
            use_average: If True, use average strategy. If False, use current strategy.

        Returns:
            Normalized probability distribution over valid actions (float64, sums to 1.0)

        Examples:
            # Get full average strategy
            strategy = infoset.get_filtered_strategy()

            # Get current strategy for specific actions
            strategy = infoset.get_filtered_strategy(valid_indices=[0, 2, 3], use_average=False)
        """
        # Get base strategy (already normalized)
        if use_average:
            full_strategy = average_strategy(self.strategy_sum)
        else:
            full_strategy = regret_matching(self.regrets)

        # Convert to float64 for consistency
        full_strategy = full_strategy.astype(np.float64)

        # Filter if needed
        if valid_indices is not None:
            strategy = full_strategy[valid_indices]
            total = np.sum(strategy)
            if total > 0:
                return strategy / total
            else:
                # Fallback to uniform over valid actions
                return np.ones(len(valid_indices), dtype=np.float64) / len(valid_indices)

        return full_strategy

    def get_strategy_safe(
        self, legal_actions: list[Action], use_average: bool = True
    ) -> tuple[np.ndarray, list[int]]:
        """
        Get strategy for given legal actions with validation and remapping.

        This method handles the case where the legal actions at query time
        may differ from the legal actions stored in the infoset. It finds
        the overlap and returns a strategy over the provided legal actions.

        Args:
            legal_actions: List of legal actions to get strategy for
            use_average: If True, use average strategy. If False, use current strategy.

        Returns:
            Tuple of (normalized_strategy, valid_indices):
            - normalized_strategy: Probability distribution over legal_actions (float64)
            - valid_indices: Indices in legal_actions that matched stored actions

        Examples:
            # Get strategy for specific legal actions
            strategy, valid_idx = infoset.get_strategy_safe(
                legal_actions=[fold(), call(), raise_(100)],
                use_average=True
            )
        """
        # Find valid indices by matching actions
        # We compare actions for equality (fold == fold, raise_(100) == raise_(100))
        valid_indices = [
            i
            for i, stored_action in enumerate(self.legal_actions)
            if stored_action in legal_actions
        ]

        if not valid_indices:
            # No overlap - return uniform over all legal actions
            uniform = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
            return uniform, list(range(len(legal_actions)))

        # Get filtered and normalized strategy
        strategy = self.get_filtered_strategy(valid_indices, use_average)

        return strategy, valid_indices

    def get_average_utility(self) -> float:
        """
        Compute average utility (expected value) at this infoset.

        Returns:
            Average utility over all iterations (0 if never reached)
        """
        if self.reach_count > 0:
            return self.cumulative_utility / self.reach_count
        else:
            return 0.0

    def update_regret(
        self,
        action_idx: int,
        regret: float,
        cfr_plus: bool = False,
        linear_cfr: bool = False,
        iteration: int = 1,
    ):
        """
        Update cumulative regret for an action.

        Supports multiple CFR variants:
        - Vanilla CFR: regrets can be negative
        - CFR+: regrets are floored at 0 (much faster convergence)
        - Linear CFR: regrets weighted by iteration number

        Args:
            action_idx: Index of action in legal_actions
            regret: Regret value to add (can be negative)
            cfr_plus: If True, floor regrets at 0 (CFR+)
            linear_cfr: If True, weight by iteration (Linear CFR)
            iteration: Current iteration (for linear weighting)
        """
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"Invalid action index: {action_idx}")

        # Apply weighting for Linear CFR
        weighted_regret = regret
        if linear_cfr:
            # Linear weighting: multiply by iteration number
            # Can also use (iteration + 1) ** 0.5 for square-root weighting
            weighted_regret = regret * iteration

        # Update regret
        if cfr_plus:
            # CFR+: Floor cumulative regrets at 0
            self.regrets[action_idx] = max(0, self.regrets[action_idx] + weighted_regret)
        else:
            # Vanilla CFR: Allow negative regrets
            self.regrets[action_idx] += weighted_regret

    def update_strategy(
        self,
        reach_prob: float,
        node_utility: float = 0.0,
        linear_cfr: bool = False,
        iteration: int = 1,
    ):
        """
        Update cumulative strategy (weighted by reach probability).

        Supports multiple CFR variants:
        - Vanilla/CFR+: Uniform weighting
        - Linear CFR: Weight by iteration number

        Args:
            reach_prob: Probability of reaching this infoset
            node_utility: Expected utility at this node (optional)
            linear_cfr: If True, weight by iteration (Linear CFR)
            iteration: Current iteration (for linear weighting)
        """
        strategy = self.get_strategy()

        # Apply weighting for Linear CFR
        weight = reach_prob
        if linear_cfr:
            # Linear weighting: multiply by iteration number
            weight = reach_prob * iteration

        self.strategy_sum += strategy * weight
        self.reach_count += 1
        self.cumulative_utility += node_utility

    def reset_regrets(self):
        """Reset all regrets to zero (for some CFR variants)."""
        self.regrets = np.zeros(self.num_actions, dtype=np.float64)

    def reset_strategy_sum(self):
        """Reset strategy sum to zero (for some CFR variants)."""
        self.strategy_sum = np.zeros(self.num_actions, dtype=np.float64)

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
            f"  Reach Count: {self.reach_count}\n"
            f"  Average Utility: {self.get_average_utility():+.4f}"
        )

    def __repr__(self) -> str:
        return f"InfoSet(key={repr(self.key)}, num_actions={self.num_actions})"


def create_infoset_key(
    player: int,
    street: Street,
    betting_sequence: str,
    spr_bucket: int,
    preflop_hand: str | None = None,
    postflop_bucket: int | None = None,
) -> InfoSetKey:
    """
    Convenience function to create an InfoSetKey with hybrid representation.

    Args:
        player: Player position (0 or 1)
        street: Current street
        betting_sequence: Normalized betting history
        spr_bucket: SPR bucket (0=shallow, 1=medium, 2=deep)
        preflop_hand: Hand string for preflop (e.g., "AKs")
        postflop_bucket: Bucket ID for postflop (e.g., 0-49)

    Returns:
        InfoSetKey instance

    Examples:
        # Preflop
        create_infoset_key(0, Street.PREFLOP, "r2.5", 2, preflop_hand="AKs")

        # Postflop
        create_infoset_key(0, Street.FLOP, "c-b0.75", 1, postflop_bucket=15)
    """
    return InfoSetKey(
        player_position=player,
        street=street,
        betting_sequence=betting_sequence,
        preflop_hand=preflop_hand,
        postflop_bucket=postflop_bucket,
        spr_bucket=spr_bucket,
    )
