"""Leduc Hold'em as an :class:`ExtensiveGame`, a second ground-truth game.

Leduc has a *mid-tree* public card (a chance node deeper than the root) and two
betting rounds, so it exercises code paths that Kuhn poker (single root deal,
one round) does not — in particular counterfactual reach and value propagation
through a non-root chance node.

Rules (standard Leduc, this encoding):
- 6-card deck: ids 0..5 with rank = id // 2, i.e. two each of J(0), Q(1), K(2).
- Each player antes 1. Bet size is 2 in round 0 and 4 in round 1.
- At most 2 raises per betting round.
- Round 0 betting, then one public card is dealt, then round 1 betting.
- Showdown: a player whose private rank matches the public rank (a pair) wins;
  otherwise the higher private rank wins; equal ranks split (net 0). Only one
  player can pair, since each rank has exactly two cards.

This module is a test helper (not named ``test_*``) so pytest does not collect
it as a test file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from src.pipeline.evaluation.game_tree import CHANCE

MAX_RAISES = 2
_BET_SIZE = {0: 2, 1: 4}
_NUM_CARDS = 6

# Chance actions are dealt cards (a pair, then a single); betting actions are
# 'f' (fold), 'c' (check/call), 'r' (bet/raise).
LeducAction = str | int | tuple[int, int]

_ORDERED_DEALS: list[tuple[int, int]] = [
    (i, j) for i in range(_NUM_CARDS) for j in range(_NUM_CARDS) if i != j
]


@dataclass(frozen=True, slots=True)
class LeducState:
    """Immutable Leduc state.

    ``committed`` is chips put in during the current round; ``total`` is each
    player's full contribution including the ante. ``bet_history`` is the public
    action sequence, with ``'/'`` marking the public-card deal, and together with
    the acting player's own card forms their information set.
    """

    p0: int | None = None
    p1: int | None = None
    public: int | None = None
    round_no: int = 0
    committed: tuple[int, int] = (0, 0)
    total: tuple[int, int] = (0, 0)
    num_raises: int = 0
    round_acts: int = 0
    to_act: int = 0
    phase: str = "private"  # private | public | bet | folded | showdown
    folder: int | None = None
    bet_history: tuple[str, ...] = ()


def _rank(card: int) -> int:
    return card // 2


def _showdown_winner(p0: int, p1: int, public: int) -> int | None:
    pair0 = _rank(p0) == _rank(public)
    pair1 = _rank(p1) == _rank(public)
    if pair0 and not pair1:
        return 0
    if pair1 and not pair0:
        return 1
    if _rank(p0) > _rank(p1):
        return 0
    if _rank(p1) > _rank(p0):
        return 1
    return None


class LeducPoker:
    """Two-player Leduc Hold'em implementing the ExtensiveGame protocol."""

    num_players = 2

    def initial_state(self) -> LeducState:
        return LeducState()

    def is_terminal(self, state: LeducState) -> bool:
        return state.phase in ("folded", "showdown")

    def current_player(self, state: LeducState) -> int:
        if state.phase in ("private", "public"):
            return CHANCE
        return state.to_act

    def chance_outcomes(self, state: LeducState) -> Sequence[tuple[LeducAction, float]]:
        if state.phase == "private":
            prob = 1.0 / len(_ORDERED_DEALS)
            return [(deal, prob) for deal in _ORDERED_DEALS]
        # Public card: uniform over cards not held privately.
        remaining = [c for c in range(_NUM_CARDS) if c not in (state.p0, state.p1)]
        prob = 1.0 / len(remaining)
        return [(card, prob) for card in remaining]

    def legal_actions(self, state: LeducState) -> Sequence[str]:
        p = state.to_act
        diff = state.committed[1 - p] - state.committed[p]
        if diff > 0:
            actions = ["f", "c"]
        else:
            actions = ["c"]
        if state.num_raises < MAX_RAISES:
            actions.append("r")
        return actions

    def next_state(self, state: LeducState, action: LeducAction) -> LeducState:
        if state.phase == "private":
            assert isinstance(action, tuple)
            p0, p1 = action
            return replace(state, p0=p0, p1=p1, phase="bet", total=(1, 1))
        if state.phase == "public":
            assert isinstance(action, int)
            return replace(
                state,
                public=action,
                phase="bet",
                round_no=1,
                committed=(0, 0),
                num_raises=0,
                round_acts=0,
                to_act=0,
                bet_history=(*state.bet_history, "/"),
            )
        return self._apply_bet(state, str(action))

    def _apply_bet(self, state: LeducState, action: str) -> LeducState:
        p = state.to_act
        other = 1 - p
        diff = state.committed[other] - state.committed[p]
        history = (*state.bet_history, action)

        if action == "f":
            return replace(state, phase="folded", folder=p, bet_history=history)

        add = diff + _BET_SIZE[state.round_no] if action == "r" else diff
        committed = list(state.committed)
        total = list(state.total)
        committed[p] += add
        total[p] += add
        committed_t = (committed[0], committed[1])
        total_t = (total[0], total[1])

        if action == "r":
            return replace(
                state,
                committed=committed_t,
                total=total_t,
                num_raises=state.num_raises + 1,
                round_acts=state.round_acts + 1,
                to_act=other,
                bet_history=history,
            )

        # action == "c": a call (diff > 0) or a check. The round closes on a call,
        # or on the second check (i.e. when the opponent has already acted).
        closes = diff > 0 or state.round_acts >= 1
        if not closes:
            return replace(
                state,
                committed=committed_t,
                total=total_t,
                round_acts=state.round_acts + 1,
                to_act=other,
                bet_history=history,
            )
        next_phase = "public" if state.round_no == 0 else "showdown"
        return replace(
            state,
            committed=committed_t,
            total=total_t,
            phase=next_phase,
            bet_history=history,
        )

    def information_state_key(self, state: LeducState, player: int) -> tuple:
        own = state.p0 if player == 0 else state.p1
        return (own, state.public, state.bet_history)

    def returns(self, state: LeducState) -> Sequence[float]:
        if state.phase == "folded":
            if state.folder == 0:
                return (-float(state.total[0]), float(state.total[0]))
            return (float(state.total[1]), -float(state.total[1]))

        assert state.p0 is not None and state.p1 is not None and state.public is not None
        winner = _showdown_winner(state.p0, state.p1, state.public)
        if winner is None:
            return (0.0, 0.0)
        amount = float(state.total[1 - winner])
        return (amount, -amount) if winner == 0 else (-amount, amount)
