"""Reference vanilla CFR over the generic game protocol.

This is a small, full-tree Counterfactual Regret Minimization solver used to
validate the evaluation harness end-to-end: on any small game implementing
:class:`~src.pipeline.evaluation.game_tree.ExtensiveGame`, running enough
iterations should drive :func:`~src.pipeline.evaluation.best_response.exploitability`
toward zero and the on-policy value toward the game's known value.

It is deliberately simple (no sampling, no CFR variants) and is not intended for
large games; the production solver is :class:`~src.engine.solver.mccfr.MCCFRSolver`.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np

from src.pipeline.evaluation.game_tree import (
    CHANCE,
    Action,
    ExtensiveGame,
    InfoKey,
    TabularStrategy,
)


def _regret_matching(regret_sum: np.ndarray) -> np.ndarray:
    positive = np.maximum(regret_sum, 0.0)
    total = positive.sum()
    if total > 0.0:
        return positive / total
    return np.full(len(regret_sum), 1.0 / len(regret_sum))


class TabularCFRSolver[StateT: Hashable, ActionT: Hashable]:
    """Full-tree vanilla CFR for small extensive-form games."""

    def __init__(self, game: ExtensiveGame[StateT, ActionT]):
        self.game = game
        self.regret_sum: dict[InfoKey, np.ndarray] = {}
        self.strategy_sum: dict[InfoKey, np.ndarray] = {}
        self.actions: dict[InfoKey, list[Action]] = {}

    def _node(self, info_key: InfoKey, legal: list[Action]) -> tuple[np.ndarray, np.ndarray]:
        regret = self.regret_sum.get(info_key)
        if regret is None:
            n = len(legal)
            regret = np.zeros(n)
            self.regret_sum[info_key] = regret
            self.strategy_sum[info_key] = np.zeros(n)
            self.actions[info_key] = list(legal)
        return regret, self.strategy_sum[info_key]

    def _cfr(self, state: StateT, reach: np.ndarray) -> np.ndarray:
        game = self.game
        if game.is_terminal(state):
            return np.asarray(game.returns(state), dtype=np.float64)

        player = game.current_player(state)
        if player == CHANCE:
            value = np.zeros(game.num_players)
            for action, prob in game.chance_outcomes(state):
                value += prob * self._cfr(game.next_state(state, action), reach)
            return value

        legal = list(game.legal_actions(state))
        info_key = game.information_state_key(state, player)
        regret_sum, strategy_sum = self._node(info_key, legal)
        strategy = _regret_matching(regret_sum)

        node_value = np.zeros(game.num_players)
        action_values = np.zeros(len(legal))
        child_values: list[np.ndarray] = []
        for idx, action in enumerate(legal):
            child_reach = reach.copy()
            child_reach[player] *= strategy[idx]
            child_value = self._cfr(game.next_state(state, action), child_reach)
            child_values.append(child_value)
            action_values[idx] = child_value[player]
            node_value += strategy[idx] * child_value

        # Counterfactual reach: product of every other player's reach probability.
        cf_reach = 1.0
        for other in range(game.num_players):
            if other != player:
                cf_reach *= reach[other]

        regret_sum += cf_reach * (action_values - node_value[player])
        strategy_sum += reach[player] * strategy
        return node_value

    def train(self, iterations: int) -> None:
        """Run ``iterations`` full-tree CFR passes."""
        for _ in range(iterations):
            self._cfr(self.game.initial_state(), np.ones(self.game.num_players))

    def average_strategy(self) -> TabularStrategy:
        """Return the average strategy, which converges to a Nash equilibrium."""
        table: dict[InfoKey, dict[Action, float]] = {}
        for info_key, strat_sum in self.strategy_sum.items():
            total = strat_sum.sum()
            actions = self.actions[info_key]
            if total > 0.0:
                probs = strat_sum / total
            else:
                probs = np.full(len(actions), 1.0 / len(actions))
            table[info_key] = dict(zip(actions, probs))
        return TabularStrategy(table)
