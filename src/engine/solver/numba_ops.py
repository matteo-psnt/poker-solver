"""
Numba-compiled operations for performance-critical paths.

This module contains JIT-compiled versions of hot functions
identified through profiling.
"""

import numpy as np
from numba import jit

# Weighting-scheme codes shared by every caller of ``apply_regret_updates``
# (numba kernels take ints, not strings).
WEIGHTING_CODES = {"none": 0, "linear": 1, "dcfr": 2}


@jit(nopython=True, cache=True)
def regret_matching(regrets):
    """Strategy from regrets by regret matching -- the core CFR operation, called
    millions of times.

    Intentionally distinct from the resolver's batched row normalization
    (``subgame_cfr._normalize_or_uniform``): this is the 1-D training-path kernel
    with exact ``sum > 0`` semantics, while the resolver normalizes
    (combos x actions) matrices under ``NORMALIZE_EPS``. Do not merge them --
    either direction changes numerics or slows its hot path.
    """
    # Upcast so float32 storage rows still yield a float64 distribution
    positive_regrets = np.maximum(regrets.astype(np.float64), 0.0)
    sum_positive = np.sum(positive_regrets)
    num_actions = len(regrets)

    if sum_positive > 0:
        # Normalize to probability distribution
        return positive_regrets / sum_positive
    # Uniform distribution (float64)
    return np.ones(num_actions, dtype=np.float64) / np.float64(num_actions)


@jit(nopython=True, cache=True)
def average_strategy(strategy_sum):
    """Normalised average strategy from ``strategy_sum``. Always float64."""
    # Upcast so float32 storage rows still yield a float64 distribution
    sums = strategy_sum.astype(np.float64)
    sum_total = np.sum(sums)
    num_actions = len(strategy_sum)

    if sum_total > 0:
        return sums / sum_total
    # Return uniform strategy (float64)
    return np.ones(num_actions, dtype=np.float64) / np.float64(num_actions)


@jit(nopython=True, cache=True)
def apply_regret_updates(
    regrets,
    target_indices,
    utilities,
    node_utility,
    opponent_reach,
    cfr_plus,
    iteration,
    weighting,
    dcfr_alpha,
    dcfr_beta,
):
    """THE regret-update kernel: every CFR-variant regret write goes through here.

    Applies, per target slot, the DCFR discount (Brown & Sandholm 2019: the
    cumulative regret is multiplied by t^e/(t^e+1), e = alpha for positive / beta
    for negative regrets, where e = 0 means x0.5 -- not a no-op) followed by the
    weighted counterfactual-regret add and the optional CFR+ floor. Callers select
    which slots to touch via ``target_indices`` (full row, partial-legal subset, or
    a single action); there is no other implementation of this math -- keep it that
    way.

    ``regrets`` is mutated in place, ``target_indices`` must be ASCENDING,
    ``iteration`` is 1-indexed, and ``weighting`` takes ``WEIGHTING_CODES``:
    0 = none, 1 = linear, 2 = DCFR.
    """
    # DISCOUNT PLACEMENT (deliberate deviation from the paper — do not "fix").
    # Published DCFR discounts every infoset's cumulative regret at every
    # iteration. This applies the factor only when an infoset is VISITED, using
    # the current t, so an infoset seen at t=5,100,5000 receives three factors
    # where the paper applies ~5000. That is not an oversight: under full-
    # information CFR every infoset updates every iteration and the two schedules
    # coincide, but under sampling they diverge, and eagerly decaying an infoset
    # through iterations that gave it no new information destroys accumulated
    # signal without replacing it.
    #
    # Measured on Leduc against exact exploitability (2026-07-28), eager applied
    # in the published order (discount R_{t-1}, then add r_t):
    #     10k iters   lazy 0.404   eager 0.454   (eager 1.12x worse)
    #     30k iters   lazy 0.206   eager 0.242   (eager 1.18x worse)
    # Per-visit also keeps the schedule independent of visit frequency; eager
    # would have deep river infosets and preflop infosets running effectively
    # different algorithms.
    for j in range(target_indices.shape[0]):
        i = target_indices[j]
        if weighting == 2 and iteration > 1:
            exponent = dcfr_alpha if regrets[i] > 0 else dcfr_beta
            if exponent == 0.0:
                # t^0 / (t^0 + 1) = 1/2
                regrets[i] *= 0.5
            else:
                t_exp = np.float64(iteration) ** exponent
                regrets[i] *= t_exp / (t_exp + 1.0)

        weighted_regret = (utilities[j] - node_utility) * opponent_reach
        if weighting == 1:
            weighted_regret = weighted_regret * iteration

        updated = regrets[i] + weighted_regret
        if cfr_plus and updated < 0:
            updated = 0.0
        regrets[i] = updated


@jit(nopython=True, cache=True)
def compute_dcfr_strategy_weight(iteration, gamma):
    """The DCFR weight for a ``strategy_sum`` update: t^gamma, 1-indexed.

    Weights a contribution to the average strategy by iteration number.
    """
    if iteration <= 1:
        return 1.0

    t = np.float64(iteration)

    if gamma == 0.0:
        return 1.0

    # Standard DCFR strategy weight: t^gamma
    # Higher gamma → more weight to recent iterations
    return t**gamma
