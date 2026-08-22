"""Build vector-CFR board contexts from the real precomputed abstraction.

The vector kernels take a :class:`HandContext` per board — live hands, their
bucket on each street, their showdown rank, and who blocks whom. Nothing in
``engine`` may reach into ``pipeline`` (import-linter enforces the layering), so
the bridge from the precomputed artifact to those contexts lives here.

This is what turns a measurement on invented buckets into a measurement on the
abstraction that would actually ship. It is cheap enough that there is no excuse
for the synthetic version: a full 1,326-hand, three-street bucket lookup for one
board takes ~10 ms against the mmapped matrices, so a universe of thousands of
boards builds in seconds.

Note what a single board does *not* cover. Under the production 100/300/600
abstraction only ~50 flop, ~60 turn and ~100 river buckets are occupied on any
one board — a bucket is a slice of the *(hand, board)* space, not of the hand
space. Transition and terminal matrices therefore need many boards before every
cell is sampled, which is the whole reason
:func:`~src.engine.solver.vector.bucket_game.derive` takes a sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import eval7
import numpy as np

from src.core.game.state import FULL_DECK, Card, Street
from src.engine.solver.vector.coupling import RELABEL_STREETS, board_relative
from src.engine.solver.vector.hand_context import (
    HandContext,
    blocking_matrix,
    enumerate_live_hands,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.engine.solver.protocols import BucketingStrategy

# eval7 handles for the canonical deck, built once. Rank lookups are the only
# per-hand cost here that is not a matrix read.
_EVAL7_DECK: tuple[eval7.Card, ...] = tuple(eval7.Card(repr(card)) for card in FULL_DECK)

POSTFLOP_PREFIX: tuple[tuple[Street, int], ...] = (
    (Street.FLOP, 3),
    (Street.TURN, 4),
    (Street.RIVER, 5),
)


def build_hand_context(
    board: Sequence[int] | np.ndarray,
    abstraction: BucketingStrategy,
    *,
    board_relative_buckets: bool = False,
) -> HandContext:
    """One board's context: live hands, per-street buckets, showdown ranks.

    ``board`` is five card indices into :data:`FULL_DECK`. Buckets come from the
    abstraction exactly as the solver would ask for them — same canonicalisation,
    same artifact — so a bucket id here means what it means in training.

    ``board_relative_buckets`` renumbers each postflop street's occupied buckets
    to within-board rank, which is a DIFFERENT abstraction rather than a
    re-encoding: ids become comparable across boards, and what a bucket means is
    strength relative to the range this board produces. It costs absolute
    strength — rank r on a wet board is not rank r on a dry one — which is
    exactly what `abstraction-coupling` cannot see and exploitability can.
    Ranking is only meaningful because the artifact numbers buckets by strength.
    """
    if len(board) != 5:
        raise ValueError(f"A runout is five cards; got {len(board)}.")

    hands = enumerate_live_hands(board)
    cards: list[Card] = [FULL_DECK[index] for index in board]
    buckets = np.zeros((4, hands.shape[0]), dtype=np.int64)

    for street, seen in POSTFLOP_PREFIX:
        prefix = tuple(cards[:seen])
        buckets[street.value - 1] = [
            abstraction.get_bucket((FULL_DECK[first], FULL_DECK[second]), prefix, street)
            for first, second in hands
        ]

    # Preflop dispatches to the 169 canonical classes inside the abstraction, so
    # it is asked for the same way rather than special-cased here.
    buckets[Street.PREFLOP.value - 1] = [
        abstraction.get_bucket((FULL_DECK[first], FULL_DECK[second]), (), Street.PREFLOP)
        for first, second in hands
    ]

    if board_relative_buckets:
        for street in RELABEL_STREETS:
            row = street.value - 1
            buckets[row] = board_relative(buckets[row])

    board_cards = [_EVAL7_DECK[index] for index in board]
    ranks = np.array(
        [
            eval7.evaluate([*board_cards, _EVAL7_DECK[first], _EVAL7_DECK[second]])
            for first, second in hands
        ],
        dtype=np.int64,
    )
    return HandContext(hands, buckets, ranks, blocking_matrix(hands))


def sample_boards(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    """``count`` distinct five-card runouts drawn uniformly."""
    return [rng.choice(52, 5, replace=False) for _ in range(count)]


def build_universe(
    abstraction: BucketingStrategy,
    count: int,
    *,
    rng: np.random.Generator,
    board_relative_buckets: bool = False,
) -> list[HandContext]:
    """A universe of sampled runouts under the real abstraction.

    Materialised, so it can be handed to a scorer that walks it repeatedly. For
    a one-pass consumer such as
    :func:`~src.engine.solver.vector.bucket_game.derive`, prefer
    :func:`iter_universe` — a context holds an ``(H, H)`` blocking matrix, so a
    list of twenty thousand is 23 GB where the generator is a few megabytes.
    """
    return list(
        iter_universe(abstraction, count, rng=rng, board_relative_buckets=board_relative_buckets)
    )


def iter_universe(
    abstraction: BucketingStrategy,
    count: int,
    *,
    rng: np.random.Generator,
    board_relative_buckets: bool = False,
) -> Iterator[HandContext]:
    """The same universe, one context at a time and none of them retained."""
    for board in sample_boards(rng, count):
        yield build_hand_context(board, abstraction, board_relative_buckets=board_relative_buckets)


__all__: Sequence[str] = (
    "POSTFLOP_PREFIX",
    "build_hand_context",
    "build_universe",
    "iter_universe",
    "sample_boards",
)
