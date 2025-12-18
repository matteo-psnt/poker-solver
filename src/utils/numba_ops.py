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
        regrets: NumPy array of regrets for each action

    Returns:
        Probability distribution over actions (sums to 1, always float64)
    """
    # Get positive regrets only (ensure float64)
    positive_regrets = np.maximum(regrets, 0.0).astype(np.float64)
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

    # Ensure float64 for consistency
    strategy_sum_f64 = strategy_sum.astype(np.float64)

    if sum_total > 0:
        return strategy_sum_f64 / sum_total
    else:
        # Return uniform strategy (float64)
        return np.ones(num_actions, dtype=np.float64) / np.float64(num_actions)


@jit(nopython=True, cache=True)
def update_regrets_cfr_plus(regrets, new_regrets, reach_prob):
    """
    Update regrets using CFR+ (floor at 0).

    Args:
        regrets: Current cumulative regrets (modified in-place)
        new_regrets: New regrets to add
        reach_prob: Reach probability weight
    """
    weighted_regrets = new_regrets * reach_prob
    regrets[:] = np.maximum(0.0, regrets + weighted_regrets)


@jit(nopython=True, cache=True)
def update_regrets_vanilla(regrets, new_regrets, reach_prob):
    """
    Update regrets using vanilla CFR (allow negative).

    Args:
        regrets: Current cumulative regrets (modified in-place)
        new_regrets: New regrets to add
        reach_prob: Reach probability weight
    """
    weighted_regrets = new_regrets * reach_prob
    regrets[:] = regrets + weighted_regrets


@jit(nopython=True, cache=True)
def update_regrets_linear(regrets, new_regrets, reach_prob, iteration):
    """
    Update regrets using linear CFR weighting.

    Args:
        regrets: Current cumulative regrets (modified in-place)
        new_regrets: New regrets to add
        reach_prob: Reach probability weight
        iteration: Current iteration number (for linear weighting)
    """
    weighted_regrets = new_regrets * reach_prob * iteration
    regrets[:] = np.maximum(0.0, regrets + weighted_regrets)


@jit(nopython=True, cache=True)
def update_strategy_sum(strategy_sum, strategy, reach_prob):
    """
    Update cumulative strategy sum.

    Args:
        strategy_sum: Cumulative strategy (modified in-place)
        strategy: Current strategy to add
        reach_prob: Reach probability weight
    """
    strategy_sum[:] = strategy_sum + strategy * reach_prob


@jit(nopython=True, cache=True)
def update_strategy_sum_linear(strategy_sum, strategy, reach_prob, iteration):
    """
    Update cumulative strategy sum with linear weighting.

    Args:
        strategy_sum: Cumulative strategy (modified in-place)
        strategy: Current strategy to add
        reach_prob: Reach probability weight
        iteration: Current iteration number (for linear weighting)
    """
    strategy_sum[:] = strategy_sum + strategy * reach_prob * iteration


@jit(nopython=True, cache=True)
def compute_action_utilities(action_utilities, strategy):
    """
    Compute expected utility given action utilities and strategy.

    Args:
        action_utilities: Utility for each action
        strategy: Probability of each action

    Returns:
        Expected utility
    """
    return np.dot(action_utilities, strategy)


@jit(nopython=True, cache=True)
def compute_regrets(action_utilities, node_utility):
    """
    Compute instantaneous regrets.

    Args:
        action_utilities: Utility for each action
        node_utility: Expected utility at this node

    Returns:
        Regret for each action
    """
    return action_utilities - node_utility
