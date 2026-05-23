"""Per-board card data the vector traversal needs.

The betting tree is card-independent; everything about cards enters here. One
context describes a single sampled public board: which hands are live, which
storage row each hand reads its strategy from on each street, and how hands
compare at showdown.

Why hand space and not bucket space:
    The strategy is per *bucket*, so the CFR math could in principle run on
    bucket vectors and be several times cheaper. Terminals are what forbid it.
    Card removal ("my ace blocks yours") and showdown ranking are properties of
    the two concrete holdings, and a bucket is a set of holdings with different
    blockers and different ranks. Running the traversal on hands and collapsing
    to buckets only at the regret write keeps terminals exact; the collapse is
    the segment-sum in ``kernel.scatter_to_buckets``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.core.game.state import Street

# Street ordinals, matching ``Street.value``, used to index per-street arrays.
STREET_ORDER: tuple[Street, ...] = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)


@dataclass(frozen=True, slots=True)
class HandContext:
    """Live hands on one board, their bucket rows, and their showdown order.

    Attributes:
        hand_cards: ``(H, 2)`` card indices (0-51) of each live hand.
        bucket_of_hand: ``(4, H)`` storage row offset within a node, per street.
            Preflop rows are the 169 canonical classes; postflop rows are the
            abstraction's buckets.
        showdown_rank: ``(H,)`` rank of each hand on the full board. Higher
            beats lower; equal ranks tie.
        blocks: ``(H, H)`` boolean, true where two hands share a card and so
            cannot be held simultaneously.
    """

    hand_cards: np.ndarray
    bucket_of_hand: np.ndarray
    showdown_rank: np.ndarray
    blocks: np.ndarray

    @property
    def num_hands(self) -> int:
        return int(self.hand_cards.shape[0])

    def buckets_for(self, street: Street) -> np.ndarray:
        return self.bucket_of_hand[street.value - 1]


def enumerate_live_hands(board: Sequence[int] | np.ndarray, num_cards: int = 52) -> np.ndarray:
    """Every two-card holding that does not collide with the board."""
    available = np.setdiff1d(
        np.arange(num_cards, dtype=np.int64), np.asarray(board, dtype=np.int64)
    )
    first, second = np.triu_indices(available.shape[0], k=1)
    return np.stack([available[first], available[second]], axis=1)


def blocking_matrix(hand_cards: np.ndarray) -> np.ndarray:
    """``(H, H)`` mask of hand pairs that share a card.

    Kept dense because every terminal consults it and the alternative — the
    card-indexed running sums that make the same computation O(H) — is an
    optimisation of the *terminal kernel*, not of this description of the board.
    """
    a = hand_cards[:, None, :, None]
    b = hand_cards[None, :, None, :]
    return (a == b).any(axis=(2, 3))


def showdown_matrix(showdown_rank: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """``(H, H)`` payoff sign: ``+1`` where the row hand beats the column hand.

    Ties and blocked pairs are zero, so a matrix-vector product against an
    opponent range yields (mass beaten − mass losing to) directly, which is
    exactly a showdown terminal's counterfactual value up to the pot factor.
    """
    sign = np.sign(showdown_rank[:, None] - showdown_rank[None, :]).astype(np.float32)
    sign[blocks] = 0.0
    return sign


__all__: Sequence[str] = (
    "STREET_ORDER",
    "HandContext",
    "blocking_matrix",
    "enumerate_live_hands",
    "showdown_matrix",
)
