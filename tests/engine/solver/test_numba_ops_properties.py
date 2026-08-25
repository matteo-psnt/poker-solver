"""Properties of the two kernels every CFR iteration ends in.

`regret_matching` and `average_strategy` are called millions of times per
iteration and their output is a probability distribution by contract: everything
downstream -- sampling, the average-strategy blueprint, every exploitability
number -- assumes it sums to 1 and is non-negative. The example tests name the
cases someone thought of; these cover the vectors a shuffle actually produces,
including the all-negative rows that DCFR's negative-regret halving keeps
around and the exhausted rows a starved solve leaves at exactly zero (a
`_prepare_nodes` bug once turned those into a uniform strategy worth -486 mbb,
so "zero regrets means uniform" is deliberate, load-bearing behaviour).

Storage is float32 (`REGRET_DTYPE`) while a standalone `InfoSet` allocates
float64, so both dtypes are generated: the kernels upcast, and the contract
holds for either.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from src.engine.solver.numba_ops import average_strategy, regret_matching

DTYPES = st.sampled_from([np.float32, np.float64])


def _vectors(min_value: float = -1e6) -> st.SearchStrategy[np.ndarray]:
    """Action vectors of 1-8 entries. Bounded well inside float32 so the
    property under test is the kernel's, not the format's."""
    values = st.floats(
        min_value=min_value, max_value=1e6, allow_nan=False, allow_infinity=False, width=32
    )
    return st.builds(
        lambda xs, dtype: np.array(xs, dtype=dtype),
        st.lists(values, min_size=1, max_size=8),
        DTYPES,
    )


@pytest.mark.timeout(60)
@given(_vectors())
def test_regret_matching_returns_a_distribution(regrets):
    strategy = regret_matching(regrets)

    assert strategy.dtype == np.float64
    assert strategy.shape == regrets.shape
    assert np.all(np.isfinite(strategy))
    assert np.all(strategy >= 0.0)
    assert np.isclose(strategy.sum(), 1.0, rtol=0, atol=1e-12)


@pytest.mark.timeout(60)
@example(np.zeros(3, dtype=np.float64))
@example(np.array([-1.0, -2.0, -3.0], dtype=np.float32))
@given(_vectors())
def test_regret_matching_puts_no_weight_on_a_non_positive_regret(regrets):
    """Regret matching plays only actions it regrets not having played --
    unless nothing is positive, where the whole vector goes uniform.

    The two pinned examples are WIDER THAN ONE ACTION on purpose: at a single
    action `1/n`, `n/1`, `1*n` and `1//n` all give the same answer, and mutation
    testing found that generation alone never widened the all-non-positive case.
    """
    strategy = regret_matching(regrets)
    positive = np.asarray(regrets, dtype=np.float64) > 0.0

    if positive.any():
        assert np.all(strategy[~positive] == 0.0)
    else:
        assert np.allclose(strategy, 1.0 / len(regrets), rtol=0, atol=1e-12)


@pytest.mark.timeout(60)
@given(_vectors(), st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False))
def test_regret_matching_is_scale_invariant(regrets, factor):
    """Only the ratios between regrets are strategy -- their magnitude is an
    iteration count. Two runs at different weightings must play the same."""
    scaled = np.asarray(regrets, dtype=np.float64) * factor

    assert np.allclose(regret_matching(regrets), regret_matching(scaled), rtol=1e-9, atol=1e-12)


@pytest.mark.timeout(60)
@example(np.zeros(3, dtype=np.float64))
@given(_vectors(min_value=0.0))
def test_average_strategy_returns_a_distribution(strategy_sum):
    """`strategy_sum` accumulates reach-weighted probabilities, so it is
    non-negative by construction -- an all-zero row is an infoset never
    visited, and uniform is the only answer available."""
    average = average_strategy(strategy_sum)

    assert average.dtype == np.float64
    assert np.all(np.isfinite(average))
    assert np.all(average >= 0.0)
    assert np.isclose(average.sum(), 1.0, rtol=0, atol=1e-12)
    if not np.any(np.asarray(strategy_sum, dtype=np.float64) > 0.0):
        assert np.allclose(average, 1.0 / len(strategy_sum), rtol=0, atol=1e-12)


@pytest.mark.timeout(60)
@example(np.array([0.25, 0.5], dtype=np.float64))
@given(_vectors(min_value=0.0))
def test_average_strategy_is_proportional_to_the_sums(strategy_sum):
    """The average strategy IS the normalised `strategy_sum` -- the blueprint is
    read straight off it. Being a distribution is not enough: mutation testing
    replaced the `sum > 0` guard so the kernel returned uniform for every input,
    and every other assertion here still passed."""
    sums = np.asarray(strategy_sum, dtype=np.float64)
    total = sums.sum()
    if total <= 0.0:
        return

    assert np.allclose(average_strategy(strategy_sum), sums / total, rtol=1e-9, atol=1e-12)
