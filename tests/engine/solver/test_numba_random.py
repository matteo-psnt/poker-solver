"""Does the kernel's generator produce the SAME stream as the one it replaces?

This is what buys bit-identity for the compiled walk. If the draws differ by
one value the trajectory diverges, and every published number becomes
incomparable — the exact cost the tree-walk change was designed to avoid.

Mid-stream states are the interesting case, so the fixtures advance the
generator by a non-multiple of 624 first: an implementation that only refills
at block boundaries passes from a fresh seed and fails here.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.engine.solver.numba_random import (
    numpy_state,
    python_state,
    random_sample,
    randrange_deck,
    restore_numpy_state,
    restore_python_state,
)

DRAWS = 50_000


@pytest.mark.parametrize("warmup", [0, 1, 623, 624, 625, 1_000])
def test_randrange_matches_python_from_any_point(warmup):
    """`random.randrange(52)`, including the rejection of 52..63."""
    random.seed(20260814)
    for _ in range(warmup):
        random.randrange(52)

    state, index = python_state()
    expected = [random.randrange(52) for _ in range(DRAWS)]

    cursor = index
    got = []
    for _ in range(DRAWS):
        value, cursor = randrange_deck(state, cursor)
        got.append(int(value))

    assert got == expected


@pytest.mark.parametrize("warmup", [0, 623, 777])
def test_random_sample_matches_numpy_from_any_point(warmup):
    np.random.seed(4242)
    for _ in range(warmup):
        np.random.random()

    state, index = numpy_state()
    expected = [np.random.random() for _ in range(DRAWS)]

    cursor = index
    got = []
    for _ in range(DRAWS):
        value, cursor = random_sample(state, cursor)
        got.append(value)

    # Bit-identical, not close: these are the same doubles or the stream differs.
    assert got == expected


class TestTheStateRoundTrips:
    """Advancing in the kernel must leave the real generator where it would be."""

    def test_python_continues_where_the_kernel_left_off(self):
        random.seed(99)
        for _ in range(11):
            random.randrange(52)
        checkpoint = random.getstate()
        expected = [random.randrange(52) for _ in range(2_000)]

        random.setstate(checkpoint)
        state, index = python_state()
        cursor = index
        for _ in range(1_500):
            _, cursor = randrange_deck(state, cursor)
        restore_python_state(state, cursor)

        # The module generator should now yield draws 1,500 onward.
        assert [random.randrange(52) for _ in range(500)] == expected[1_500:]

    def test_numpy_continues_where_the_kernel_left_off(self):
        np.random.seed(7)
        for _ in range(5):
            np.random.random()
        checkpoint = np.random.get_state()
        expected = [np.random.random() for _ in range(2_000)]

        np.random.set_state(checkpoint)
        state, index = numpy_state()
        cursor = index
        for _ in range(1_500):
            _, cursor = random_sample(state, cursor)
        restore_numpy_state(state, cursor)

        assert [np.random.random() for _ in range(500)] == expected[1_500:]
