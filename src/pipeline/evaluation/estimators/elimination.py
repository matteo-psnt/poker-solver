"""Whole elimination matches, which is the only format Chipzen actually deals.

Every bot-vs-bot match there runs until one bot holds every chip, with blinds
escalating, so a fixed-depth duel measures a game the platform never deals. Here
stacks carry across hands, the button alternates, and the match ends when one
side cannot post. A hand is played for the EFFECTIVE stack -- all either side can
win -- and the deeper side's excess sits out, which is exact heads-up.

ESCALATION IS SHRINKING STACKS, NOT RISING BLINDS: a blueprint is cut for one
blind level and `CrossDepthPlayer` carries amounts across unchanged, so moving the
blinds would read every raise in the wrong currency. Both describe the same depth
curve and both stacks shrink together, so the ratio -- the match -- is untouched.
The schedule is a MODEL (live logs show one 100 -> 150 step, hence 1.5x); results
move with it, so hold it fixed across arms and quote it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import FULL_DECK
from src.pipeline.evaluation.estimators.depth_duel import play_hand
from src.pipeline.evaluation.statistics import summarize_samples

if TYPE_CHECKING:
    from src.core.game.actions import Action
    from src.core.game.rules import GameRules
    from src.core.game.state import GameState
    from src.pipeline.evaluation.estimators.cross_depth import CrossDepthPlayer

MAX_HANDS = 400


# How a ladder picks a rung for a table depth: `up` (the seat's rule since 10-05)
# or `nearest` in log space, the rule it replaced, kept as the comparison arm.
SELECTION_RULES = ("nearest", "up")


class LadderSeat:
    """Answers each hand with the rung `DepthLadder.select` would pick.

    The seat's rule restated, because `pipeline` may not import `interfaces`.
    Nothing binds the two: change both or neither, or this measures another bot.
    """

    def __init__(self, players: list[CrossDepthPlayer], rule: str = "up") -> None:
        if rule not in SELECTION_RULES:
            raise ValueError(f"unknown rung selection rule {rule!r}; one of {SELECTION_RULES}")
        self.players = sorted(players, key=lambda p: p.depth_in_blinds)
        self.rule = rule
        self.chosen = self.players[-1]
        self.switches = 0

    def for_depth(self, depth: float) -> CrossDepthPlayer:
        if depth <= 0:
            return self.players[0]
        if self.rule == "up":
            # The seat's rule: the shallowest rung at least this deep, within 2x.
            above = [p for p in self.players if p.depth_in_blinds >= depth * (1 - 1e-9)]
            if above and above[0].depth_in_blinds <= 2.0 * depth:
                return above[0]
        target = math.log(depth)
        return min(self.players, key=lambda p: abs(math.log(p.depth_in_blinds) - target))

    def select(self, depth: float) -> None:
        picked = self.for_depth(depth)
        if picked is not self.chosen:
            self.switches += 1
        self.chosen = picked

    def act(self, state: GameState) -> Action:
        return self.chosen.act(state)

    def share_stream(self, stream: np.random.Generator) -> None:
        for player in self.players:
            player.rng = stream

    @property
    def truncation_rate(self) -> float:
        seen = sum(p.tally.decisions for p in self.players)
        return sum(p.tally.truncated for p in self.players) / seen if seen else 0.0


@dataclass(frozen=True)
class MatchOutcome:
    winner: int
    hands: int
    final_depth: float  # the shorter stack when the match ended, in big blinds


def play_match(
    seats: tuple[LadderSeat, LadderSeat],
    *,
    rules: GameRules,
    starting_stack: int,
    cards: np.random.Generator,
    escalate_every: int = 10,
    escalate_by: float = 1.5,
) -> MatchOutcome:
    """Play until one seat cannot post. ``rules`` are the BLUEPRINTS' own, always."""
    stacks = [float(starting_stack), float(starting_stack)]
    big_blind = rules.big_blind
    played = 0
    while played < MAX_HANDS:
        if played and played % escalate_every == 0:
            stacks = [stack / escalate_by for stack in stacks]
        effective = int(min(stacks))
        if effective < 2 * big_blind:
            break
        depth = effective / big_blind
        for seat in seats:
            seat.select(depth)
        order = cards.permutation(len(FULL_DECK))
        hole_cards = (
            (FULL_DECK[order[0]], FULL_DECK[order[1]]),
            (FULL_DECK[order[2]], FULL_DECK[order[3]]),
        )
        board_stack = [FULL_DECK[i] for i in order[4:9]]
        net = play_hand(
            seats,
            rules=rules,
            hole_cards=hole_cards,
            board_stack=board_stack,
            button=played % 2,
            starting_stack=effective,
        )
        stacks[0] += net
        stacks[1] -= net
        played += 1
    return MatchOutcome(
        winner=0 if stacks[0] > stacks[1] else 1,
        hands=played,
        final_depth=min(stacks) / big_blind,
    )


@dataclass(frozen=True)
class EliminationResult:
    """Ladder A's match win rate, from duplicate PAIRS of matches."""

    a_win_rate: float
    se: float
    confidence_95: tuple[float, float]
    p_value: float  # against 0.5
    num_pairs: int
    num_matches: int
    mean_hands: float
    pair_samples: list[float] = field(repr=False)  # A's wins per pair, 0 / 0.5 / 1


def play_elimination(
    a: LadderSeat,
    b: LadderSeat,
    *,
    rules: GameRules,
    starting_stack: int,
    num_pairs: int,
    seed: int = 1,
    escalate_every: int = 10,
    escalate_by: float = 1.5,
) -> EliminationResult:
    """Each pair is one card sequence played twice, ladders swapped between seats.

    One strategy stream per match, shared by both seats as in `depth_duel`, so two
    identical ladders split every pair exactly; seat bias cancels by construction.
    """
    samples: list[float] = []
    hands: list[int] = []
    for pair in range(num_pairs):
        wins = 0
        for a_seat in (0, 1):
            stream = np.random.default_rng(np.random.SeedSequence([seed, pair, 1]))
            a.share_stream(stream)
            b.share_stream(stream)
            outcome = play_match(
                (a, b) if a_seat == 0 else (b, a),
                rules=rules,
                starting_stack=starting_stack,
                cards=np.random.default_rng(np.random.SeedSequence([seed, pair, 0])),
                escalate_every=escalate_every,
                escalate_by=escalate_by,
            )
            wins += outcome.winner == a_seat
            hands.append(outcome.hands)
        samples.append(wins / 2.0)
    # Centred on an even split, so the t-test's zero is "neither ladder better".
    summary = summarize_samples([sample - 0.5 for sample in samples])
    return EliminationResult(
        a_win_rate=0.5 + summary["mean"],
        se=summary["se"],
        confidence_95=(0.5 + summary["ci_lower"], 0.5 + summary["ci_upper"]),
        p_value=summary["p_value"],
        num_pairs=num_pairs,
        num_matches=2 * num_pairs,
        mean_hands=float(np.mean(hands)),
        pair_samples=samples,
    )
