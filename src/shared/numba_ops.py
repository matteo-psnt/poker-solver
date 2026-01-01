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
    Compute DCFR discount factor for cumulative regrets.

    Uses standard DCFR formula: t^exponent / (t^exponent + 1)
    where exponent = alpha for positive regrets, beta for negative regrets.

    This discount is applied to cumulative regrets each iteration, giving
    exponentially less weight to older iterations (Brown & Sandholm 2019).

    Args:
        iteration: Current iteration number (1-indexed)
        alpha: Exponent for positive regrets (typically 1.5)
        beta: Exponent for negative regrets (typically 0.0)
        is_positive: True if cumulative regret is positive, False if negative

    Returns:
        Discount factor to multiply cumulative regret by (float64, range [0, 1])
    """
    if iteration <= 1:
        return 1.0

    t = np.float64(iteration)
    exponent = alpha if is_positive else beta

    if exponent == 0.0:
        return 1.0

    # Standard DCFR discount: t^exponent / (t^exponent + 1)
    # As t increases, this approaches 1.0, meaning less discount
    # Higher exponent → stronger discounting early on
    t_exp = t**exponent
    return t_exp / (t_exp + 1.0)


@jit(nopython=True, cache=True)
def compute_dcfr_strategy_weight(iteration, gamma):
    """
    Compute DCFR weight for strategy_sum updates.

    Uses standard DCFR formula: t^gamma
    This weights the contribution to the average strategy by iteration number.

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

    # Standard DCFR strategy weight: t^gamma
    # Higher gamma → more weight to recent iterations
    return t**gamma
