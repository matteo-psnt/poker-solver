"""Recursive MCCFR traversal implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import GameState
from src.engine.solver.infoset import InfoSet
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.engine.solver.numba_ops import (
    WEIGHTING_CODES,
    apply_regret_updates,
    compute_dcfr_strategy_weight,
)
from src.engine.solver.policy_lookup import filter_stored_actions

if TYPE_CHECKING:
    from .solver import MCCFRSolver

# Identity index rows for the full-row kernel call, cached per action count so
# the hot path stays allocation-free. Read-only by convention (numba kernels
# never write target_indices).
_IDENTITY_INDICES: dict[int, np.ndarray] = {}


def _identity_indices(num_actions: int) -> np.ndarray:
    indices = _IDENTITY_INDICES.get(num_actions)
    if indices is None:
        indices = _IDENTITY_INDICES.setdefault(num_actions, np.arange(num_actions, dtype=np.int64))
    return indices


def _terminal_utility(self: MCCFRSolver, state: GameState, traversing_player: int) -> float:
    """Evaluate payoff at terminal states, completing board when necessary."""
    if len(state.board) < 5:
        complete_state = self.deal_remaining_cards(state)
        return complete_state.get_payoff(traversing_player, self.rules)
    return state.get_payoff(traversing_player, self.rules)


def _infoset_context(
    self: MCCFRSolver,
    state: GameState,
    current_player: int,
) -> tuple[InfoSet, Sequence[Action], list[int], np.ndarray]:
    """Build infoset, filter valid actions, and compute strategy over valid actions."""
    infoset_key = encode_infoset_key(state, current_player, self.card_abstraction)
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


def _sample_action_index(strategy: np.ndarray) -> int:
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
    placement; see docs/AVERAGE_STRATEGY_WEIGHTING.md, option A).
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
    reach_probs: list[float],
) -> float:
    """Recursive MCCFR traversal with external sampling."""
    if state.is_terminal:
        return _terminal_utility(self, state, traversing_player)

    if self.is_chance_node(state):
        next_state = self.sample_chance_outcome(state)
        return cfr_external_sampling(self, next_state, traversing_player, reach_probs)

    current_player = state.current_player
    infoset, legal_actions, valid_indices, strategy = _infoset_context(
        self,
        state,
        current_player,
    )

    if current_player == traversing_player:
        action_utilities = np.zeros(len(legal_actions))
        solver_config = self.config.solver
        prune = (
            infoset.pruned_mask(
                self.iteration,
                solver_config.pruning_threshold,
                solver_config.prune_start_iteration,
                solver_config.prune_reactivate_frequency,
            )
            if solver_config.enable_pruning
            else None
        )
        # Collapse "nothing pruned this visit" to None so every node where pruning
        # does nothing — the overwhelming majority — takes the vectorised fast path
        # exactly like a pruning-disabled run. Pruning then pays the masked
        # Python-loop paths only on the few nodes that actually skip an action.
        if prune is not None and not prune.any():
            prune = None

        for local_idx, action in enumerate(legal_actions):
            original_idx = valid_indices[local_idx]

            if prune is not None and prune[original_idx]:
                continue

            next_state = state.apply_action(action, self.rules)
            if self.is_chance_node(next_state):
                next_state = self.sample_chance_outcome(next_state)

            action_utilities[local_idx] = cfr_external_sampling(
                self,
                next_state,
                traversing_player,
                reach_probs,
            )

        if prune is not None:
            unpruned_mask = np.array(
                [not prune[valid_indices[i]] for i in range(len(legal_actions))]
            )
            if np.any(unpruned_mask):
                unpruned_strategy = strategy[unpruned_mask]
                unpruned_sum = unpruned_strategy.sum()
                if unpruned_sum > 0:
                    unpruned_strategy = unpruned_strategy / unpruned_sum
                else:
                    unpruned_strategy = np.ones(unpruned_mask.sum()) / unpruned_mask.sum()
                node_utility = float(np.dot(unpruned_strategy, action_utilities[unpruned_mask]))
            else:
                node_utility = float(np.dot(strategy, action_utilities))
        else:
            node_utility = float(np.dot(strategy, action_utilities))

        # Lock-free shared writes: every worker applies the full per-update
        # CFR+/DCFR math directly to shared memory for every infoset it visits.
        # Skipped only for placeholder views whose global ID is still unknown.
        if infoset.writable:
            opponent = 1 - current_player
            # One kernel call for every shape: full row (identity indices,
            # allocation-free), partial-legal subset, or unpruned subset.
            if prune is None:
                if len(valid_indices) == infoset.num_actions:
                    target_indices = _identity_indices(infoset.num_actions)
                else:
                    target_indices = np.asarray(valid_indices, dtype=np.int64)
                utilities = action_utilities
            else:
                unpruned = [j for j in range(len(legal_actions)) if not prune[valid_indices[j]]]
                target_indices = np.array([valid_indices[j] for j in unpruned], dtype=np.int64)
                utilities = action_utilities[unpruned]
            apply_regret_updates(
                infoset.regrets,
                target_indices,
                utilities,
                node_utility,
                reach_probs[opponent],
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
            # carries no pi_i (see docs/AVERAGE_STRATEGY_WEIGHTING.md).
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

    action_idx = _sample_action_index(strategy)
    action = legal_actions[action_idx]

    new_reach_probs = reach_probs.copy()
    new_reach_probs[current_player] *= float(strategy[action_idx])

    next_state = state.apply_action(action, self.rules)
    if self.is_chance_node(next_state):
        next_state = self.sample_chance_outcome(next_state)

    return cfr_external_sampling(self, next_state, traversing_player, new_reach_probs)


def cfr_outcome_sampling(
    self: MCCFRSolver,
    state: GameState,
    traversing_player: int,
    reach_probs: list[float],
) -> float:
    """Recursive MCCFR traversal with outcome sampling."""
    if state.is_terminal:
        return _terminal_utility(self, state, traversing_player)

    if self.is_chance_node(state):
        next_state = self.sample_chance_outcome(state)
        return cfr_outcome_sampling(self, next_state, traversing_player, reach_probs)

    current_player = state.current_player
    infoset, legal_actions, valid_indices, strategy = _infoset_context(
        self,
        state,
        current_player,
    )

    action_idx = _sample_action_index(strategy)
    action = legal_actions[action_idx]

    next_state = state.apply_action(action, self.rules)
    if self.is_chance_node(next_state):
        next_state = self.sample_chance_outcome(next_state)

    new_reach_probs = reach_probs.copy()
    new_reach_probs[current_player] *= float(strategy[action_idx])

    sampled_utility = cfr_outcome_sampling(self, next_state, traversing_player, new_reach_probs)

    if current_player == traversing_player and infoset.writable:
        opponent = 1 - current_player
        baseline = sampled_utility

        for local_idx in range(len(legal_actions)):
            original_idx = valid_indices[local_idx]
            strategy_prob = float(strategy[local_idx])
            if strategy_prob <= 0:
                continue

            importance_weight = reach_probs[opponent] / strategy_prob
            if local_idx == action_idx:
                regret = (sampled_utility - baseline) * importance_weight
            else:
                regret = -baseline * importance_weight

            infoset.update_regret(
                original_idx,
                regret,
                cfr_plus=self.config.solver.cfr_plus,
                iteration=self.iteration,
                iteration_weighting=self.config.solver.iteration_weighting,
                dcfr_alpha=self.config.solver.dcfr_alpha,
                dcfr_beta=self.config.solver.dcfr_beta,
            )

        # Unchanged pending the outcome-sampling audit: here the traverser's
        # reach IS threaded (both players multiply into reach_probs above), so
        # the explicit weight is live, unlike the external-sampling site.
        _accumulate_average_strategy(
            self,
            infoset,
            valid_indices,
            strategy,
            reach_weight=reach_probs[current_player],
        )
        infoset.increment_reach_count()
        infoset.add_cumulative_utility(sampled_utility)

    return sampled_utility
