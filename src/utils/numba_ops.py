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


@jit(nopython=True, cache=True)
def compute_dcfr_weight(iteration, alpha, beta, is_positive):
    """
    Compute DCFR discount weight for the current iteration.

    Uses linear discounting formula: t^(exponent-1)
    where exponent = alpha for positive regrets, beta for negative regrets.

    This gives more weight to recent iterations, implementing
    Discounted CFR (Brown & Sandholm 2019).

    Args:
        iteration: Current iteration number (1-indexed)
        alpha: Exponent for positive regrets (typically 1.5)
        beta: Exponent for negative regrets (typically 0.0)
        is_positive: True if regret is positive, False if negative

    Returns:
        Weight multiplier for this regret update (float64)
    """
    if iteration <= 1:
        return 1.0

    t = np.float64(iteration)
    exponent = alpha if is_positive else beta

    if exponent == 0.0:
        return 1.0

    # Linear discount: t^(exponent-1)
    # For current iteration t, this gives t^(exponent-1)
    # Higher t and higher exponent → higher weight → more emphasis on recent
    return t ** (exponent - 1.0)


@jit(nopython=True, cache=True)
def compute_dcfr_strategy_weight(iteration, gamma):
    """
    Compute DCFR discount weight for strategy_sum updates.

    Uses linear discounting formula: t^(gamma-1)

    Args:
        iteration: Current iteration number (1-indexed)
        gamma: Strategy discount exponent (typically 2.0)

    Returns:
        Weight multiplier for strategy_sum update (float64)
    """
    if iteration <= 1:
        return 1.0

    t = np.float64(iteration)

    if gamma == 0.0:
        return 1.0

    return t ** (gamma - 1.0)
