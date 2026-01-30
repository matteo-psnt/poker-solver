"""Recursive MCCFR traversal implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import GameState
from src.engine.solver.infoset import InfoSet
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.engine.solver.policy_lookup import filter_stored_actions
from src.shared.numba_ops import compute_dcfr_strategy_weight

if TYPE_CHECKING:
    from .solver import MCCFRSolver


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
) -> tuple[InfoSet, list[Action], list[int], np.ndarray]:
    """Build infoset, filter valid actions, and compute strategy over valid actions."""
    infoset_key = encode_infoset_key(state, current_player, self.card_abstraction)
    legal_actions = self.rules.get_legal_actions(state, action_model=self.action_model)

    if not legal_actions:
        raise ValueError(f"No legal actions at state: {state}")

    infoset = self.storage.get_or_create_infoset(infoset_key, legal_actions)

    valid_indices, valid_actions = filter_stored_actions(infoset, state, self.rules, legal_actions)

    if not valid_actions:
        valid_actions = legal_actions
        valid_indices = list(range(len(legal_actions)))

    strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=False)
    return infoset, valid_actions, valid_indices, strategy


def _update_average_strategy(
    self: MCCFRSolver,
    infoset: InfoSet,
    valid_indices: list[int],
    strategy: np.ndarray,
    player_reach_prob: float,
    node_utility: float,
) -> None:
    """Update cumulative strategy and infoset statistics."""
    for local_idx, strategy_prob in enumerate(strategy):
        original_idx = valid_indices[local_idx]
        weight = strategy_prob * player_reach_prob

        if self.config.solver.iteration_weighting == "dcfr":
            gamma_weight = compute_dcfr_strategy_weight(
                self.iteration,
                self.config.solver.dcfr_gamma,
            )
            weight *= gamma_weight
        elif self.config.solver.iteration_weighting == "linear":
            weight *= self.iteration

        infoset.strategy_sum[original_idx] += weight

    infoset.increment_reach_count()
    infoset.add_cumulative_utility(node_utility)


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

        for local_idx, action in enumerate(legal_actions):
            original_idx = valid_indices[local_idx]

            if self.config.solver.enable_pruning and infoset.is_pruned(original_idx):
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

        if self.config.solver.enable_pruning:
            unpruned_mask = np.array(
                [not infoset.is_pruned(valid_indices[i]) for i in range(len(legal_actions))]
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
            for local_idx in range(len(legal_actions)):
                original_idx = valid_indices[local_idx]

                if self.config.solver.enable_pruning and infoset.is_pruned(original_idx):
                    continue

                regret = action_utilities[local_idx] - node_utility
                infoset.update_regret(
                    original_idx,
                    regret * reach_probs[opponent],
                    cfr_plus=self.config.solver.cfr_plus,
                    iteration=self.iteration,
                    iteration_weighting=self.config.solver.iteration_weighting,
                    dcfr_alpha=self.config.solver.dcfr_alpha,
                    dcfr_beta=self.config.solver.dcfr_beta,
                )

            if self.config.solver.enable_pruning:
                infoset.update_pruning(
                    iteration=self.iteration,
                    pruning_threshold=self.config.solver.pruning_threshold,
                    prune_start_iteration=self.config.solver.prune_start_iteration,
                    prune_reactivate_frequency=self.config.solver.prune_reactivate_frequency,
                )

            _update_average_strategy(
                self,
                infoset,
                valid_indices,
                strategy,
                player_reach_prob=reach_probs[current_player],
                node_utility=node_utility,
            )

        return node_utility

    action_idx = int(np.random.choice(len(legal_actions), p=strategy))
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

    action_idx = int(np.random.choice(len(legal_actions), p=strategy))
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

        _update_average_strategy(
            self,
            infoset,
            valid_indices,
            strategy,
            player_reach_prob=reach_probs[current_player],
            node_utility=sampled_utility,
        )

    return sampled_utility
