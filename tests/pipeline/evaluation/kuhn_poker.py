"""Kuhn poker as an :class:`ExtensiveGame`, for validating exact evaluation.

Kuhn poker is a 2-player, 3-card game with a fully known analytic solution,
which makes it the canonical ground-truth check for best-response and CFR code.

Rules (this encoding):
- Deck is {0, 1, 2} (J, Q, K). Each player antes 1 and is dealt one card.
- Actions are ``"p"`` (pass/check/fold) and ``"b"`` (bet/call), each of size 1.
- Betting sequences and terminals:
    ""    -> P0 acts
    "p"   -> P1 acts (p = check to showdown, b = bet)
    "b"   -> P1 acts (p = fold, b = call to showdown)
    "pb"  -> P0 acts (p = fold, b = call to showdown)
    "pp", "bp", "bb", "pbp", "pbb" -> terminal
- Terminal payoffs (net chips, player 0's perspective; player 1 is the negation):
    "bp"  -> +1 (P1 folded to a bet)
    "pbp" -> -1 (P0 folded to a bet)
    "pp"  -> +/-1 by showdown (pot = 2)
    "bb"  -> +/-2 by showdown (pot = 4)
    "pbb" -> +/-2 by showdown (pot = 4)

The unique game value to the first player at equilibrium is -1/18.

This module is a test helper (not named ``test_*``) so pytest does not collect
it as a test file.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.pipeline.evaluation.game_tree import CHANCE

# State is (cards, history):
#   cards: tuple[int, int] | None  -- None before the deal (chance root)
#   history: str                   -- betting sequence so far
KuhnState = tuple[tuple[int, int] | None, str]

_TERMINALS = frozenset({"pp", "bp", "bb", "pbp", "pbb"})
_ACTIONS: tuple[str, str] = ("p", "b")

# All 6 ordered deals of 2 distinct cards from {0, 1, 2}, each with probability 1/6.
_DEALS: list[tuple[int, int]] = [(i, j) for i in range(3) for j in range(3) if i != j]


class KuhnPoker:
    """Two-player Kuhn poker implementing the ExtensiveGame protocol."""

    num_players = 2

    def initial_state(self) -> KuhnState:
        return (None, "")

    def is_terminal(self, state: KuhnState) -> bool:
        cards, history = state
        return cards is not None and history in _TERMINALS

    def current_player(self, state: KuhnState) -> int:
        cards, history = state
        if cards is None:
            return CHANCE
        # P0 acts on even-length histories, P1 on odd-length ones.
        return len(history) % 2

    def chance_outcomes(self, state: KuhnState) -> Sequence[tuple[tuple[int, int], float]]:
        prob = 1.0 / len(_DEALS)
        return [(deal, prob) for deal in _DEALS]

    def legal_actions(self, state: KuhnState) -> Sequence[str]:
        return _ACTIONS

    def next_state(self, state: KuhnState, action) -> KuhnState:
        cards, history = state
        if cards is None:
            # Chance action is the dealt pair of cards.
            return (action, history)
        return (cards, history + action)

    def information_state_key(self, state: KuhnState, player: int) -> tuple[int, str]:
        cards, history = state
        assert cards is not None, "information set requested at the chance root"
        # A player sees only their own card and the public betting history.
        return (cards[player], history)

    def returns(self, state: KuhnState) -> Sequence[float]:
        cards, history = state
        assert cards is not None
        payoff0 = self._payoff_player0(cards, history)
        return (payoff0, -payoff0)

    @staticmethod
    def _payoff_player0(cards: tuple[int, int], history: str) -> float:
        if history == "bp":
            return 1.0  # P1 folded to P0's bet
        if history == "pbp":
            return -1.0  # P0 folded to P1's bet
        # Remaining terminals are showdowns; stake is the amount each committed
        # beyond the ante: 1 chip in "pp", 2 chips in "bb"/"pbb".
        stake = 2.0 if history in ("bb", "pbb") else 1.0
        p0_wins = cards[0] > cards[1]
        return stake if p0_wins else -stake
