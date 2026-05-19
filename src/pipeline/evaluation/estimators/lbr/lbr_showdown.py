"""Showdown valuation: what a hand is worth once the cards are in.

Needs the rules, an evaluator, the runout budget and an RNG -- and nothing
about opponents, beliefs or the action menu. Separating it keeps the showdown
maths from acquiring an opinion about strategy.
"""

from __future__ import annotations

import numpy as np

from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Card, GameState, Street
from src.engine.search.range_inference import ALL_COMBOS, COMBO_MASKS
from src.pipeline.evaluation.estimators.lbr.opponent_model import known_mask
from src.shared.numeric import NORMALIZE_EPS

_DECK_MASKS: np.ndarray = np.array([card.mask for card in FULL_DECK], dtype=np.int64)


class ShowdownValuer:
    """Values showdowns, completing the board by sampled runout when needed."""

    def __init__(self, owner) -> None:
        # Reads rules/evaluator/config/rng THROUGH the owner rather than copying
        # them. The engine rebinds its RNG per hand, so a captured reference
        # silently keeps drawing from the previous hand's stream -- which is
        # exactly what two earlier attempts at this split got wrong.
        self._owner = owner

    @property
    def rules(self) -> GameRules:
        return self._owner.rules

    @property
    def evaluator(self):
        return self._owner.evaluator

    @property
    def config(self):
        return self._owner.config

    @property
    def rng(self) -> np.random.Generator:
        return self._owner.rng

    def showdown_value(
        self, state: GameState, lbr_player: int, opp: int, belief: np.ndarray
    ) -> float:
        """Belief-weighted payoff over the surviving range on a complete board.

        Payoffs are pot arithmetic on hand-rank comparisons (win: pot - invested,
        tie: pot/2 - invested, lose: -invested — exactly ``get_payoff``'s showdown
        cases), so no per-combo GameState construction is needed.
        """
        known = known_mask(state, opp)
        weights = np.where((COMBO_MASKS & known) == 0, belief, 0.0)
        total = weights.sum()
        if total <= NORMALIZE_EPS:
            return float(state.get_payoff(lbr_player, self.rules))

        pot = float(state.pot)
        invested = self.rules.invested_chips(state)[lbr_player]
        win_payoff = pot - invested
        tie_payoff = pot / 2.0 - invested
        lose_payoff = -invested
        lbr_rank = self.evaluator.evaluate(state.hole_cards[lbr_player], state.board)

        ev = 0.0
        for idx in np.nonzero(weights)[0]:
            opp_rank = self.evaluator.evaluate(ALL_COMBOS[idx], state.board)
            if lbr_rank < opp_rank:
                payoff = win_payoff
            elif lbr_rank == opp_rank:
                payoff = tie_payoff
            else:
                payoff = lose_payoff
            ev += float(weights[idx]) * payoff
        return ev / float(total)

    def allin_showdown_value(
        self, state: GameState, lbr_player: int, opp: int, belief: np.ndarray
    ) -> float:
        """All-in showdown value averaged over board runouts.

        Runouts draw from the cards the LBR player cannot see (its holes + board);
        the opponent's *dealt* hand is deliberately not excluded — it is a fiction
        this evaluator integrates out, so letting runouts cover those cards matches
        the range convention (combos colliding with a runout drop out per runout,
        exactly as on a real river). One missing card is enumerated exactly; more
        are sampled ``allin_runouts`` times from the per-hand deterministic RNG.
        """
        known = known_mask(state, opp)
        missing = 5 - len(state.board)
        unseen = [card for card in FULL_DECK if not (card.mask & known)]
        runouts: list[tuple[Card, ...]]
        if missing == 1:
            runouts = [(card,) for card in unseen]
        else:
            count = max(1, self.config.allin_runouts)
            runouts = []
            for _ in range(count):
                picks = self.rng.choice(len(unseen), size=missing, replace=False)
                runouts.append(tuple(unseen[int(i)] for i in picks))

        total = 0.0
        for extra in runouts:
            runout_state = self._with_runout(state, extra)
            total += self.showdown_value(runout_state, lbr_player, opp, belief)
        return total / len(runouts)

    @staticmethod
    def _with_runout(state: GameState, extra: tuple[Card, ...]) -> GameState:
        """Terminal copy of ``state`` with the board completed by ``extra``."""
        return state.replace(
            street=Street.RIVER,
            board=state.board + extra,
            is_terminal=True,
            to_call=0,
            validate=False,
        )

    def showdown_equity(
        self,
        lbr_hand: tuple[Card, Card],
        board: tuple[Card, ...],
        opp_weights: np.ndarray,
        active: np.ndarray,
    ) -> float:
        board_mask = 0
        for card in board:
            board_mask |= card.mask
        lbr_rank = self.evaluator.evaluate(lbr_hand, board)
        acc = 0.0
        weight = 0.0
        for idx in active:
            if COMBO_MASKS[idx] & board_mask:
                continue
            w = opp_weights[idx]
            opp_rank = self.evaluator.evaluate(ALL_COMBOS[idx], board)
            if lbr_rank < opp_rank:
                acc += w
            elif lbr_rank == opp_rank:
                acc += 0.5 * w
            weight += w
        return acc / weight if weight > NORMALIZE_EPS else 0.5

    def complete_board(self, board: tuple[Card, ...], known: int) -> tuple[Card, ...]:
        needed = 5 - len(board)
        drawn: list[Card] = []
        used = known
        while len(drawn) < needed:
            idx = int(self.rng.integers(0, 52))
            mask = int(_DECK_MASKS[idx])
            if used & mask:
                continue
            used |= mask
            drawn.append(FULL_DECK[idx])
        return tuple(board) + tuple(drawn)
