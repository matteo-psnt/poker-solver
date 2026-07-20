"""
Information set representation for CFR.

An information set groups together all game states that are indistinguishable
to a player given their information (hole cards, board, betting history).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from src.core.game.actions import Action
from src.engine.solver.numba_ops import (
    average_strategy,
    compute_dcfr_weight,
    regret_matching,
)

if TYPE_CHECKING:
    from src.core.game.state import Street


@dataclass(frozen=True, slots=True)
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
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate that exactly one of preflop_hand or postflop_bucket is set."""
        if self.street.name == "PREFLOP":
            if self.preflop_hand is None:
                raise ValueError("preflop_hand must be set for PREFLOP street")
            if self.postflop_bucket is not None:
                raise ValueError("postflop_bucket must be None for PREFLOP street")
        else:
            if self.postflop_bucket is None:
                raise ValueError(f"postflop_bucket must be set for {self.street.name} street")
            if self.preflop_hand is not None:
                raise ValueError(f"preflop_hand must be None for {self.street.name} street")

        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.player_position,
                    self.street,
                    self.betting_sequence,
                    self.preflop_hand,
                    self.postflop_bucket,
                    self.spr_bucket,
                )
            ),
        )

    def __getstate__(self) -> tuple:
        # Exclude the cached ``_hash``: Python randomizes string hashing per process
        # (PYTHONHASHSEED), so a hash pickled in one process is invalid in another.
        # Pickling it caused checkpointed keys to never match freshly-encoded keys on
        # load, so blueprint lookups (eval/resume) missed 100% of the time.
        return (
            self.player_position,
            self.street,
            self.betting_sequence,
            self.preflop_hand,
            self.postflop_bucket,
            self.spr_bucket,
        )

    def __setstate__(self, state: tuple | list) -> None:
        # New pickles are the 6-field tuple above; legacy default-pickled keys are a
        # 7-element list ``[*fields, stale_hash]``. Take the six fields (dropping any
        # stale hash) and recompute the hash in the CURRENT process.
        values = tuple(state[:6])
        names = (
            "player_position",
            "street",
            "betting_sequence",
            "preflop_hand",
            "postflop_bucket",
            "spr_bucket",
        )
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_hash", hash(values))

    def __hash__(self) -> int:
        """Hash for dictionary storage."""
        return self._hash

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, InfoSetKey):
            return False
        if self._hash != other._hash:
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
        if self.street.name == "PREFLOP":
            return self.preflop_hand or ""
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

    def __init__(
        self, key: InfoSetKey, legal_actions: Sequence[Action], *, allocate_arrays: bool = True
    ):
        """
        Initialize information set.

        Args:
            key: InfoSetKey identifier
            legal_actions: List of legal actions at this infoset
            allocate_arrays: Skip regret/strategy allocation when the caller
                immediately replaces them with storage-backed views (this
                constructor runs once per node visit in traversal).
        """
        self.key = key
        self.legal_actions = legal_actions
        self.num_actions = len(legal_actions)

        # CFR data structures
        if allocate_arrays:
            self.regrets = np.zeros(self.num_actions, dtype=np.float64)
            self.strategy_sum = np.zeros(self.num_actions, dtype=np.float64)

        # Statistics tracking
        self.reach_count = 0  # Number of times this infoset was reached
        self.cumulative_utility = 0.0  # Sum of node utilities (for average)
        self._reach_counts_view: np.ndarray | None = None
        self._cumulative_utility_view: np.ndarray | None = None
        self._stats_index: int | None = None
        self._stats_read_only = False

        # False for placeholder views onto storage rows that must not be written
        # (e.g. a remote infoset whose global ID this worker has not learned yet).
        # Traversal skips regret/strategy updates on non-writable infosets.
        self.writable = True

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
        # Fast path: no filtering requested
        if valid_indices is None:
            if use_average:
                return average_strategy(self.strategy_sum)
            return regret_matching(self.regrets)

        if len(valid_indices) == self.num_actions:
            if use_average:
                return average_strategy(self.strategy_sum)
            return regret_matching(self.regrets)

        # Compute strategy only for valid actions to avoid full-array work.
        if use_average:
            return average_strategy(self.strategy_sum[valid_indices])
        return regret_matching(self.regrets[valid_indices])

    def get_average_utility(self) -> float:
        """
        Compute average utility (expected value) at this infoset.

        Returns:
            Average utility over all iterations (0 if never reached)
        """
        if self.reach_count > 0:
            return self.cumulative_utility / self.reach_count
        return 0.0

    def update_regret(
        self,
        action_idx: int,
        regret: float,
        cfr_plus: bool = False,
        iteration: int = 1,
        iteration_weighting: Literal["none", "linear", "dcfr"] = "none",
        dcfr_alpha: float = 1.5,
        dcfr_beta: float = 0.0,
    ):
        """
        Update cumulative regret for an action.

        Supports multiple CFR variants:
        - Vanilla CFR: regrets can be negative
        - CFR+: regrets are floored at 0 (much faster convergence)
        - Linear CFR: regrets weighted by iteration number
        - DCFR: cumulative regrets discounted each iteration (Brown & Sandholm 2019)

        Args:
            action_idx: Index of action in legal_actions
            regret: Regret value to add (can be negative)
            cfr_plus: If True, floor regrets at 0 (CFR+)
            iteration: Current iteration (for linear/DCFR weighting)
            iteration_weighting: One of {'none', 'linear', 'dcfr'}.
            dcfr_alpha: Positive regret discount exponent (DCFR)
            dcfr_beta: Negative regret discount exponent (DCFR)
        """
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"Invalid action index: {action_idx}")

        # DCFR: Discount cumulative regret BEFORE adding new regret
        if iteration_weighting == "dcfr":
            # Apply discount factor to existing cumulative regret
            # Discount based on whether cumulative regret is positive or negative
            is_positive = self.regrets[action_idx] > 0
            discount_factor = compute_dcfr_weight(iteration, dcfr_alpha, dcfr_beta, is_positive)
            self.regrets[action_idx] *= discount_factor

        # Apply weighting to incoming regret
        weighted_regret = regret
        if iteration_weighting == "linear":
            # Linear CFR: multiply by iteration number
            weighted_regret = regret * iteration

        # Update regret
        if cfr_plus:
            # CFR+: Floor cumulative regrets at 0
            self.regrets[action_idx] = max(0, self.regrets[action_idx] + weighted_regret)
        else:
            # Vanilla CFR: Allow negative regrets
            self.regrets[action_idx] += weighted_regret

    def pruned_mask(
        self,
        iteration: int,
        pruning_threshold: float,
        prune_start_iteration: int,
        prune_reactivate_frequency: int,
    ) -> np.ndarray:
        """Per-action prune flags for this visit, derived live from the regrets.

        Regret-based pruning (Brown & Sandholm 2019): an action whose cumulative
        regret is below ``-pruning_threshold`` is skipped, so its subtree is not
        traversed and its regret is not updated this visit. It needs no stored
        state — ``regrets`` already persists in shared storage, and because a
        pruned action's regret is frozen while pruned, the decision is naturally
        sticky across visits. A periodic reactivation window (before
        ``prune_start_iteration`` and whenever ``iteration`` is a multiple of
        ``prune_reactivate_frequency``) returns an all-false mask so every action
        is re-explored and a prematurely pruned action can recover — the
        convergence safeguard. Never prunes every action: there must be something
        to sample and to renormalise the node value over.
        """
        if iteration < prune_start_iteration or iteration % prune_reactivate_frequency == 0:
            return np.zeros(self.num_actions, dtype=bool)
        mask = self.regrets < -pruning_threshold
        if mask.all():
            return np.zeros(self.num_actions, dtype=bool)
        return mask

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
        return f"InfoSet(key={self.key!r}, num_actions={self.num_actions})"
