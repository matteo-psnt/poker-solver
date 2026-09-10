"""What decides a turn. The trivial ones exist to be CHECKED, not to compete.

GTO Wizard post their own baselines to the public leaderboard, so two of these
have a published right answer: **Always Fold -63.18 +/- 1.16 and Check Call
-184.19 +/- 7.96 AIVAT bb/100** (v2, read 2026-08-31). Reproducing either over a
couple of thousand hands is what ground-truth-tests the wire encoding, the
cumulative-bet convention and the hand loop -- with no blueprint involved, so a
mismatch cannot be blamed on the strategy.

A depth-mismatched blueprint is NOT such a test: a 100 bb arm at 200 bb is off
our tree from the first action, and a bad score there separates nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.interfaces.gtowizard.protocol import BET, CALL, CHECK, FOLD

if TYPE_CHECKING:
    from src.interfaces.gtowizard.protocol import Frame


@dataclass(frozen=True)
class Move:
    """One answer, in their vocabulary and their chips.

    ``amount`` is the CUMULATIVE wager for the round, and is required for a bet.

    ``off_tree`` counts the opponent actions this spot had to snap to a legal
    size, and ``truncated`` marks a history our tree ran out of room for. Both
    are the honest measure of how far the label drifted from the table, and the
    numbers to watch when a blueprint plays worse here than it scores at home.
    """

    action: str
    amount: int | None = None
    off_tree: int = 0
    truncated: bool = False


class Agent(Protocol):
    def decide(self, frame: Frame) -> Move: ...


class CheckCall:
    """Checks where it can, calls otherwise. Published: -184.19 +/- 7.96."""

    name = "check-call"

    def decide(self, frame: Frame) -> Move:
        return Move(CHECK if frame.turn.allows(CHECK) else CALL)


class AlwaysFold:
    """Folds where it can, checks otherwise. Published: -63.18 +/- 1.16."""

    name = "always-fold"

    def decide(self, frame: Frame) -> Move:
        return Move(FOLD if frame.turn.allows(FOLD) else CHECK)


class AllIn:
    """Shoves whenever a bet is offered. Their board has this near -380 raw."""

    name = "all-in"

    def decide(self, frame: Frame) -> Move:
        if frame.turn.allows(BET):
            return Move(BET, frame.turn.raise_max)
        return Move(CALL)


class RandomUniform:
    """Uniform over the base actions, with a uniform size when it bets.

    Seeded per agent rather than from the global RNG: a benchmark run is worth
    reproducing, and `python hash()`-style ambient randomness is exactly what
    made an earlier experiment unrepeatable.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def decide(self, frame: Frame) -> Move:
        action = self._rng.choice(frame.turn.legal_actions)
        if action != BET:
            return Move(action)
        return Move(BET, self._rng.randint(frame.turn.raise_min, frame.turn.raise_max))


BASELINES: tuple[str, ...] = ("check-call", "always-fold", "all-in", "random")


def baseline(name: str, seed: int = 0) -> Agent:
    """One of the trivial agents by name. ``seed`` is only read by ``random``."""
    match name:
        case "check-call":
            return CheckCall()
        case "always-fold":
            return AlwaysFold()
        case "all-in":
            return AllIn()
        case "random":
            return RandomUniform(seed)
    raise ValueError(f"'{name}' is not a baseline; expected one of {BASELINES}.")


# What the public leaderboard says each baseline scores, AIVAT bb/100 +/- se,
# against GTO Wizard AI v2, read 2026-08-31. The probe's whole purpose is to
# land inside one of these, so they are pinned here rather than remembered.
PUBLISHED: dict[str, tuple[float, float]] = {
    "check-call": (-184.19, 7.96),
    "always-fold": (-63.18, 1.16),
}
