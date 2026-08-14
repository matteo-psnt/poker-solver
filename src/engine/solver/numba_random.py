"""Python's and numpy's random streams, reproduced inside a kernel.

WHY NOT JUST BUFFER THE DRAWS. The obvious plan — pre-generate draws in Python
and let the kernel consume them — does not preserve the stream. Generating N
advances the generator by N even when the walk consumes fewer, so the NEXT
iteration starts from the wrong place; and restoring the state and re-advancing
by the number actually used costs more draws than the buffering saved.

So the kernel carries the generator instead. Both streams the traversal uses
are MT19937 and both expose their state to Python, so the kernel can advance
the REAL generator and hand the state back:

    random.randrange(52)   dealing        `getrandbits(6)`, rejecting 52..63
    np.random.random()     action sampling  a 53-bit double from two words

That is what keeps the compiled walk bit-identical to the traversal it
replaces, which is the property that lets it deploy without re-baselining
every published number.

Round-tripping the state:

    state, index = python_state()          # from random.getstate()
    ... kernel advances them ...
    restore_python_state(state, index)     # back into random.setstate()
"""

from __future__ import annotations

import random

import numpy as np
from numba import jit

# MT19937 constants, shared by both generators.
_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF

# `random.randrange(52)` asks for bit_length(52) = 6 bits and redraws on 52..63.
_DEAL_BITS = 6
_DECK = 52


@jit(nopython=True, cache=True)
def _twist(state):
    for i in range(_N):
        y = (state[i] & _UPPER) | (state[(i + 1) % _N] & _LOWER)
        word = state[(i + _M) % _N] ^ (y >> 1)
        if y & 1:
            word ^= _MATRIX_A
        state[i] = word


@jit(nopython=True, cache=True)
def next_word(state, index):
    """One tempered 32-bit word, refilling the block when the index runs out."""
    if index >= _N:
        _twist(state)
        index = 0
    y = state[index]
    index += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & 0xFFFFFFFF, index


@jit(nopython=True, cache=True)
def randrange_deck(state, index):
    """``random.randrange(52)``: six top bits, rejecting anything past the deck."""
    while True:
        word, index = next_word(state, index)
        value = word >> (32 - _DEAL_BITS)
        if value < _DECK:
            return value, index


@jit(nopython=True, cache=True)
def random_sample(state, index):
    """``np.random.random()``: a 53-bit double built from two words."""
    high, index = next_word(state, index)
    low, index = next_word(state, index)
    return ((high >> 5) * 67108864.0 + (low >> 6)) / 9007199254740992.0, index


def python_state() -> tuple[np.ndarray, int]:
    """The `random` module's MT state, as arrays a kernel can take."""
    version, internal, _ = random.getstate()
    if version != 3:
        raise RuntimeError(f"unexpected random.getstate() version {version}")
    return np.array(internal[:_N], dtype=np.uint32), int(internal[_N])


def restore_python_state(state: np.ndarray, index: int) -> None:
    """Put an advanced state back, so the module generator continues from it."""
    random.setstate((3, (*(int(word) for word in state), int(index)), None))


def numpy_state() -> tuple[np.ndarray, int]:
    """The legacy ``np.random`` MT state, as arrays a kernel can take."""
    name, keys, position, _, _ = np.random.get_state()
    if name != "MT19937":
        raise RuntimeError(f"unexpected np.random bit generator {name}")
    return np.array(keys, dtype=np.uint32), int(position)


def restore_numpy_state(state: np.ndarray, index: int) -> None:
    """Put an advanced state back into the legacy ``np.random`` generator."""
    np.random.set_state(("MT19937", state.astype(np.uint32), int(index), 0, 0.0))
