"""Fast root-only CFR update for real-time resolving."""

from __future__ import annotations

import time

import numpy as np


def solve_root_strategy(
    action_values: np.ndarray,
    *,
    budget_ms: int,
    min_iterations: int = 4,
) -> np.ndarray:
    """
    Compute a root strategy from action values with a strict wall-clock budget.

    This uses regret-matching iterations over a fixed value vector, which is
    sufficient for the resolver MVP where deeper values are already rolled out.
    """
    n = len(action_values)
    if n == 0:
        raise ValueError("action_values cannot be empty")
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    regrets = np.zeros(n, dtype=np.float64)
    strategy_sum = np.zeros(n, dtype=np.float64)
    deadline = time.perf_counter() + (budget_ms / 1000.0)
    iterations = 0

    while iterations < min_iterations or time.perf_counter() < deadline:
        pos = np.maximum(regrets, 0.0)
        if pos.sum() <= 1e-12:
            strategy = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            strategy = pos / pos.sum()

        strategy_sum += strategy
        node_value = float(np.dot(strategy, action_values))
        regrets += action_values - node_value
        iterations += 1

    if strategy_sum.sum() <= 1e-12:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return strategy_sum / strategy_sum.sum()
