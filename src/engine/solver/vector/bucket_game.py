"""The abstract game with the board dimension removed.

A bucketed abstraction has *already integrated over boards*: a bucket is a set of
``(hand, board)`` combos grouped by strength. So a blueprint does not need boards
in its kernel at all. Chance stops being "deal a card" and becomes "move from
this bucket to that one", and the whole game collapses to one tree with no board
axis — which is the difference between a run that costs one pass per board (there
are 134,459 of them) and one that costs a single pass, full stop.

This module derives that game from a universe of runouts:

    transitions     ``P(next-street bucket | this-street bucket)``, one row-
                    stochastic matrix per street boundary
    compatible      ``P(two holdings do not collide | one drawn from each
                    bucket)`` — card removal, averaged
    showdown        ``E[sign(b beats b')]`` at the river — who wins, averaged

Every one is an *average over the boards inside a bucket*, and that averaging is
the entire approximation. Two things are lost, and both are stated here rather
than buried because they are what the design has to be judged on:

1. **Card removal becomes a bucket average.** Holding an ace blocks specific
   opponent holdings; after averaging it blocks a fraction of a bucket.
2. **Transitions become independent per player.** Both players see the same turn
   card, so their bucket moves are correlated. Applying a transition matrix to
   each player separately drops that correlation, and the abstract game then
   contains states no real board produces — one player leaping to a high river
   bucket while the other collapses, on the same card.

Neither is a CFR-correctness question; both are abstraction-quality questions,
and the cost is measurable rather than arguable: solve this game, then score its
strategy in the hand-space game where card removal and showdowns are exact.

Mass is conserved rather than leaked. Buckets are derived over the hands live on
the *full* runout, so a hand never disappears mid-tree and every transition row
sums to one. The alternative — letting rows sum to less than one as board cards
kill holdings — puts blocking in two places at once, which the zero-sum check
cannot catch because both players lose mass symmetrically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Street
from src.engine.solver.vector.hand_context import HandContext, showdown_matrix

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# float64 throughout: these matrices are tiny (a few hundred squared) and they
# are the constants every value in the kernel is built from, so rounding them is
# the one economy with no upside. See ``bucket_kernel.DTYPE``.
DTYPE = np.float64

# Street boundaries a transition matrix is needed for, in order.
STREET_STEPS: tuple[tuple[Street, Street], ...] = (
    (Street.PREFLOP, Street.FLOP),
    (Street.FLOP, Street.TURN),
    (Street.TURN, Street.RIVER),
)


@dataclass(frozen=True, slots=True)
class BucketGame:
    """Everything the board-free kernel needs, and nothing about boards.

    Attributes:
        buckets_per_street: Bucket count per street, keyed by ``Street``.
        transitions: ``{(from, to): (n_from, n_to)}`` row-stochastic matrices.
        compatible: ``{street: (n, n)}`` probability two holdings drawn from the
            two buckets do not share a card.
        showdown: ``(n_river, n_river)`` expected payoff sign. Antisymmetric,
            which is what carries the zero-sum property into bucket space.
    """

    buckets_per_street: dict[Street, int]
    transitions: dict[tuple[Street, Street], np.ndarray]
    compatible: dict[Street, np.ndarray]
    showdown: np.ndarray

    def num_buckets(self, street: Street) -> int:
        return self.buckets_per_street[street]

    @property
    def max_buckets(self) -> int:
        return max(self.buckets_per_street.values())

    def validate(self) -> None:
        """Assert the invariants the kernel relies on."""
        for step, matrix in self.transitions.items():
            rows = matrix.sum(axis=1)
            occupied = rows > 0
            if not np.allclose(rows[occupied], 1.0, atol=1e-4):
                raise ValueError(f"Transition {step} is not row-stochastic: {rows[occupied]}")
        if not np.allclose(self.showdown, -self.showdown.T, atol=1e-5):
            raise ValueError("Showdown matrix is not antisymmetric; zero-sum would not hold.")


def _one_hot(bucket_of_hand: np.ndarray, num_buckets: int) -> np.ndarray:
    """``(hands, buckets)`` indicator, the aggregator for every matrix here."""
    out = np.zeros((bucket_of_hand.shape[0], num_buckets), dtype=DTYPE)
    out[np.arange(bucket_of_hand.shape[0]), bucket_of_hand] = 1.0
    return out


def derive(
    contexts: Iterable[HandContext],
    buckets_per_street: dict[Street, int],
) -> BucketGame:
    """Build the board-free game by averaging over a universe of runouts.

    Each context is one runout with its live hands, per-street bucket assignment
    and showdown ranks — the same objects the hand-space kernel consumes, so the
    two games are guaranteed to describe the same universe and a strategy is
    comparable across them.

    Takes an *iterable* and consumes it exactly once, so the universe can be
    streamed rather than held. That is not a nicety: a context carries an
    ``(H, H)`` blocking matrix, about 1.2 MB at a full board, so twenty thousand
    of them is 23 GB in a list and a few megabytes as a generator. Deriving from
    enough boards to populate a 600-bucket river is only possible streamed.
    """
    streets = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)

    transition_num = {
        step: np.zeros((buckets_per_street[step[0]], buckets_per_street[step[1]]), dtype=np.float64)
        for step in STREET_STEPS
    }
    compat_num = {
        street: np.zeros((buckets_per_street[street],) * 2, dtype=np.float64) for street in streets
    }
    pair_den = {
        street: np.zeros((buckets_per_street[street],) * 2, dtype=np.float64) for street in streets
    }
    showdown_num = np.zeros((buckets_per_street[Street.RIVER],) * 2, dtype=np.float64)

    for context in contexts:
        indicators = {
            street: _one_hot(context.buckets_for(street), buckets_per_street[street])
            for street in streets
        }
        # Compatible == not blocked. The hand-space kernel spends this matrix at
        # every terminal; here it is spent once and folded into bucket pairs.
        compatible = (~context.blocks).astype(DTYPE)
        signs = showdown_matrix(context.showdown_rank, context.blocks)

        for street in streets:
            gate = indicators[street]
            counts = gate.sum(axis=0)
            compat_num[street] += (gate.T @ compatible @ gate).astype(np.float64)
            pair_den[street] += np.outer(counts, counts).astype(np.float64)

        showdown_num += (indicators[Street.RIVER].T @ signs @ indicators[Street.RIVER]).astype(
            np.float64
        )

        for step in STREET_STEPS:
            transition_num[step] += (indicators[step[0]].T @ indicators[step[1]]).astype(np.float64)

    transitions = {}
    for step, counts in transition_num.items():
        totals = counts.sum(axis=1, keepdims=True)
        transitions[step] = np.divide(
            counts, totals, out=np.zeros_like(counts), where=totals > 0
        ).astype(DTYPE)

    compatible_rates = {}
    for street in streets:
        den = pair_den[street]
        compatible_rates[street] = np.divide(
            compat_num[street], den, out=np.zeros_like(den), where=den > 0
        ).astype(DTYPE)

    river_den = pair_den[Street.RIVER]
    showdown = np.divide(
        showdown_num, river_den, out=np.zeros_like(river_den), where=river_den > 0
    ).astype(DTYPE)

    game = BucketGame(
        buckets_per_street=dict(buckets_per_street),
        transitions=transitions,
        compatible=compatible_rates,
        showdown=showdown,
    )
    game.validate()
    return game


__all__: Sequence[str] = ("STREET_STEPS", "BucketGame", "derive")
