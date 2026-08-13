"""Recursive MCCFR traversal implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.engine.solver.numba_ops import (
    WEIGHTING_CODES,
    apply_regret_updates,
    compute_dcfr_strategy_weight,
)
from src.engine.solver.policy.lookup import filter_stored_actions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.game.actions import Action
    from src.core.game.state import GameState
    from src.engine.solver.infoset.model import InfoSet

    from .solver import MCCFRSolver

# Identity index rows for the full-row kernel call, cached per action count so
# the hot path stays allocation-free. Read-only by convention (numba kernels
# never write target_indices).
_IDENTITY_INDICES: dict[int, np.ndarray] = {}


def identity_indices(num_actions: int) -> np.ndarray:
    indices = _IDENTITY_INDICES.get(num_actions)
    if indices is None:
        indices = _IDENTITY_INDICES.setdefault(num_actions, np.arange(num_actions, dtype=np.int64))
    return indices


def _terminal_utility(self: MCCFRSolver, state: GameState, traversing_player: int) -> float:
    """Evaluate payoff at terminal states, completing the board when necessary.

    ``deal_remaining_cards`` owns the is-the-board-complete test and returns
    ``state`` untouched when there is nothing to deal, so the traversal never has
    to know that a board exists at all.
    """
    return self.deal_remaining_cards(state).get_payoff(traversing_player, self.rules)


def _infoset_context(
    self: MCCFRSolver,
    state: GameState,
    current_player: int,
) -> tuple[InfoSet, Sequence[Action], list[int], np.ndarray]:
    """Resolve the acting infoset, its actions, and its current strategy.

    Dispatched through the solver because infoset *identity* is not a property
    of the CFR math: HUNL indexes a betting tree, Kuhn/Leduc keys a different
    game. Everything below this call is identity-agnostic.
    """
    return self.lookup_infoset(state, current_player)


def keyed_infoset_context(
    self: MCCFRSolver,
    state: GameState,
    current_player: int,
) -> tuple[InfoSet, Sequence[Action], list[int], np.ndarray]:
    """Lookup by hashed key, for games with no betting tree to index.

    ``StaticTreeSolver`` overrides it; in practice the only caller left is the
    Kuhn/Leduc conformance harness.
    """
    infoset_key = self.encode_infoset_key(state, current_player)
    legal_actions = self.rules.get_legal_actions(state, action_model=self.action_model)

    if not legal_actions:
        raise ValueError(f"No legal actions at state: {state}")

    infoset = self.storage.get_or_create_infoset(infoset_key, legal_actions)

    if infoset.legal_actions is legal_actions:
        # Same list object => the stored-action filter is an identity; skip it.
        valid_actions = legal_actions
        valid_indices = list(range(len(legal_actions)))
    else:
        valid_indices, valid_actions = filter_stored_actions(
            infoset, state, self.rules, legal_actions
        )
        if not valid_actions:
            valid_actions = legal_actions
            valid_indices = list(range(len(legal_actions)))

    strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=False)
    return infoset, valid_actions, valid_indices, strategy


def sample_action_index(strategy: np.ndarray) -> int:
    """Sample an index from a probability vector.

    Equivalent in distribution to ``np.random.choice(len(strategy), p=strategy)``
    but ~30x cheaper for the short action vectors CFR deals in.
    """
    r = np.random.random()
    acc = 0.0
    probs = strategy.tolist()
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def _accumulate_average_strategy(
    self: MCCFRSolver,
    infoset: InfoSet,
    valid_indices: list[int],
    strategy: np.ndarray,
    reach_weight: float,
) -> None:
    """Accumulate the current iterate into the average strategy.

    Zinkevich's average weights each iterate by the acting player's OWN reach
    ``pi_i(I)``; ``reach_weight`` is whatever part of that weight the visit
    frequency of the call site does not already supply. Under external sampling
    the update runs at OPPONENT nodes, which are visited exactly when the
    sampled opponent/chance actions lead there — visit frequency contributes
    ``pi_i * pi_chance`` on its own (chance is iteration-invariant and
    normalizes out per infoset), so the correct ``reach_weight`` is 1.0 and any
    explicit reach term would double-count (OpenSpiel's ``AverageType.SIMPLE``
    placement).
    """
    weight = reach_weight
    if self.config.solver.iteration_weighting == "dcfr":
        weight *= compute_dcfr_strategy_weight(self.iteration, self.config.solver.dcfr_gamma)
    elif self.config.solver.iteration_weighting == "linear":
        weight *= self.iteration

    strategy_sum = infoset.strategy_sum
    for local_idx, strategy_prob in enumerate(strategy.tolist()):
        strategy_sum[valid_indices[local_idx]] += strategy_prob * weight


def cfr_external_sampling(
    self: MCCFRSolver,
    state: GameState,
    traversing_player: int,
) -> float:
    """Recursive MCCFR traversal with external sampling.

    Carries no reach vector: the traverser enumerates its own actions, so its
    accumulators here take their weight from the visit frequency instead: the
    opponent's and chance's actions are sampled, so a node is reached exactly
    pi_{-i} of the time, and threading an explicit reach would apply that same
    factor twice (see the regret update and ``_accumulate_average_strategy``).
    """
    if state.is_terminal:
        return _terminal_utility(self, state, traversing_player)

    if self.is_chance_node(state):
        next_state = self.sample_chance_outcome(state)
        return cfr_external_sampling(self, next_state, traversing_player)

    current_player = state.current_player
    infoset, legal_actions, valid_indices, strategy = _infoset_context(
        self,
        state,
        current_player,
    )

    if current_player == traversing_player:
        action_utilities = np.zeros(len(legal_actions))
        solver_config = self.config.solver

        for local_idx, action in enumerate(legal_actions):
            next_state = state.apply_action(action, self.rules)
            if self.is_chance_node(next_state):
                next_state = self.sample_chance_outcome(next_state)

            action_utilities[local_idx] = cfr_external_sampling(
                self,
                next_state,
                traversing_player,
            )

        node_utility = float(np.dot(strategy, action_utilities))

        # Lock-free shared writes: every worker applies the full per-update
        # CFR+/DCFR math directly to shared memory for every infoset it visits.
        # Skipped only for placeholder views whose global ID is still unknown.
        if infoset.writable:
            # One kernel call for either shape: the full row (identity indices,
            # allocation-free) or a partial-legal subset.
            if len(valid_indices) == infoset.num_actions:
                target_indices = identity_indices(infoset.num_actions)
            else:
                target_indices = np.asarray(valid_indices, dtype=np.int64)
            utilities = action_utilities
            # Opponent reach is deliberately 1.0. Under external sampling the
            # opponent's actions are SAMPLED, so this node is visited with
            # probability pi_{-i} and the visit frequency already supplies that
            # weight -- the same argument _accumulate_average_strategy makes for
            # its own weight below. Passing reach_probs[opponent] here (as this
            # did until 2026-07-25) applied the opponent's sampled action reach a
            # second time, squaring it, and the regrets then minimised a
            # reweighted objective whose fixed point is not the equilibrium.
            apply_regret_updates(
                infoset.regrets,
                target_indices,
                utilities,
                node_utility,
                1.0,
                solver_config.cfr_plus,
                self.iteration,
                WEIGHTING_CODES[solver_config.iteration_weighting],
                solver_config.dcfr_alpha,
                solver_config.dcfr_beta,
            )

            # Diagnostics only (no strategy consumer reads these): visit count
            # and running utility of the traverser's own nodes. The average
            # strategy itself accumulates at OPPONENT nodes below — a
            # traverser-node update would be pi_{-i}-weighted, since the
            # traverser enumerates its own actions and its visit frequency
            # carries no pi_i.
            infoset.increment_reach_count()
            infoset.add_cumulative_utility(node_utility)
            self.applied_updates += 1
        else:
            self.dropped_unknown_id_updates += 1

        return node_utility

    # Opponent node: this is where the average strategy accumulates — visit
    # frequency supplies the pi_i weighting (see _accumulate_average_strategy).
    if infoset.writable:
        _accumulate_average_strategy(self, infoset, valid_indices, strategy, reach_weight=1.0)
        self.applied_updates += 1
    else:
        self.dropped_unknown_id_updates += 1

    action_idx = sample_action_index(strategy)
    action = legal_actions[action_idx]

    next_state = state.apply_action(action, self.rules)
    if self.is_chance_node(next_state):
        next_state = self.sample_chance_outcome(next_state)

    return cfr_external_sampling(self, next_state, traversing_player)
