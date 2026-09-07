"""Perfect recall of the bucket path, composed over a per-street abstraction.

A per-street abstraction forgets a hand's earlier buckets: two hands that sat
in different flop buckets share a turn row whenever their turn buckets agree,
so the game the solver sees has imperfect recall and CFR carries no
convergence guarantee on it. Composing the ids restores recall without a new
artifact -- the turn row is (flop, turn) and the river row (flop, turn, river),
mixed-radix over the base's per-street counts -- and every consumer that keys
on ``(node, bucket)`` picks it up unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.game.state import Street

if TYPE_CHECKING:
    from src.core.game.state import Card
    from src.engine.solver.protocols import BucketingStrategy


@dataclass(frozen=True, slots=True)
class PathRecallBucketer:
    """Key each postflop street on the whole bucket path, not the current bucket.

    Preflop stays the 169 lossless classes and the flop is the base's own
    bucket; recall starts at the turn. Deliberately exposes none of the
    artifact's private arrays, so the compiled scalar walk declines it and
    falls back to the Python walk rather than reading base buckets under a
    tree sized for composite ones.
    """

    base: BucketingStrategy

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        if street in (Street.PREFLOP, Street.FLOP):
            return self.base.get_bucket(hole_cards, board, street)
        code = self.base.get_bucket(hole_cards, board[:3], Street.FLOP)
        code = code * self.base.num_buckets(Street.TURN) + self.base.get_bucket(
            hole_cards, board[:4], Street.TURN
        )
        if street == Street.TURN:
            return code
        return code * self.base.num_buckets(Street.RIVER) + self.base.get_bucket(
            hole_cards, board[:5], Street.RIVER
        )

    def num_buckets(self, street: Street) -> int:
        if street in (Street.PREFLOP, Street.FLOP):
            return self.base.num_buckets(street)
        count = self.base.num_buckets(Street.FLOP) * self.base.num_buckets(Street.TURN)
        if street == Street.TURN:
            return count
        return count * self.base.num_buckets(Street.RIVER)

    def __str__(self) -> str:
        return f"path-recall({self.base})"
