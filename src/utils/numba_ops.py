"""
Numba-compiled operations for performance-critical paths.

This module contains JIT-compiled versions of hot functions
identified through profiling.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def regret_matching(regrets):
    """
    Compute strategy from regrets using regret matching.

    This is the core operation in CFR, called millions of times.

    Args:
        regrets: NumPy array of regrets for each action (float64)

    Returns:
        Probability distribution over actions (sums to 1, always float64)
    """
    # Get positive regrets only (ensure float64)
    positive_regrets = np.maximum(regrets, 0.0)
    sum_positive = np.sum(positive_regrets)
    num_actions = len(regrets)

    if sum_positive > 0:
        # Normalize to probability distribution
        return positive_regrets / sum_positive
    else:
        # Uniform distribution (float64)
        return np.ones(num_actions, dtype=np.float64) / np.float64(num_actions)


@jit(nopython=True, cache=True)
def average_strategy(strategy_sum):
    """
    Compute average strategy from strategy_sum.

    Args:
        strategy_sum: Cumulative strategy sum across iterations

    Returns:
        Normalized average strategy (always float64)
    """
    sum_total = np.sum(strategy_sum)
    num_actions = len(strategy_sum)

    if sum_total > 0:
        return strategy_sum / sum_total
    else:
        # Return uniform strategy (float64)
        return np.ones(num_actions, dtype=np.float64) / np.float64(num_actions)
