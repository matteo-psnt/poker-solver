"""
Monte Carlo Counterfactual Regret Minimization (MCCFR) solver.

Implements MCCFR with external sampling (default) or outcome sampling for scalable poker solving.
"""

import random
from typing import List, Optional

import numpy as np
from treys import Deck

from src.actions.betting_actions import BettingActions
from src.bucketing.base import BucketingStrategy
from src.game.actions import ActionType
from src.game.rules import GameRules
from src.game.state import Card, GameState, Street
from src.solver.base import BaseSolver
from src.solver.storage import Storage
from src.utils.config import Config


class MCCFRSolver(BaseSolver):
    """
    Monte Carlo CFR with external sampling or outcome sampling.

    External sampling (default):
    - Explores all actions for traversing player
    - Samples single action for opponent
    - Samples chance outcomes

    Outcome sampling:
    - Samples single action for all players
    - Samples chance outcomes
    - Faster but higher variance

    Both methods scale to large games like HUNLHE.
    """

    def __init__(
        self,
        action_abstraction: BettingActions,
        card_abstraction: BucketingStrategy,
        storage: Storage,
        config: Optional[Config] = None,
    ):
        """
        Initialize MCCFR solver.

        Args:
            action_abstraction: Action abstraction
            card_abstraction: Card abstraction
            storage: Storage backend
            config: Config object (defaults to Config.default() if not provided)
        """
        super().__init__(action_abstraction, card_abstraction, storage, config)

        # Extract config values from typed Config object
        cfg = self.config
        self.cfr_plus = cfg.solver.cfr_plus
        self.linear_cfr = cfg.solver.linear_cfr
        self.sampling_method = cfg.solver.sampling_method
        self.starting_stack = cfg.game.starting_stack
        self.small_blind = cfg.game.small_blind
        self.big_blind = cfg.game.big_blind
        seed = cfg.system.seed

        if self.sampling_method not in ["external", "outcome"]:
            raise ValueError(
                f"Invalid sampling_method: {self.sampling_method}. Must be 'external' or 'outcome'."
            )

        # Game rules
        self.rules = GameRules(self.small_blind, self.big_blind)

        # RNG
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Cache: Reuse deck instance to avoid repeated Random() creation
        self._deck = Deck()
        self._deck_cards = self._deck.cards[:]  # Keep original card list

    def train_iteration(self) -> float:
        """
        Execute one MCCFR iteration using configured sampling method.

        Uses player alternation optimization: each iteration only traverses
        from one player's perspective, alternating between iterations.
        This halves traversal cost while preserving convergence.

        Returns:
            Utility for player 0
        """
        # Deal random cards and create initial state
        state = self._deal_initial_state()

        # Alternate traversing player per iteration
        # This halves computational cost while preserving CFR convergence
        traversing_player = self.iteration % 2

        if self.sampling_method == "external":
            util = self._cfr_external_sampling(state, traversing_player, [1.0, 1.0])
        else:  # outcome sampling
            util = self._cfr_outcome_sampling(state, traversing_player, [1.0, 1.0])

        self.iteration += 1

        # Return utility from player 0's perspective
        # If we traversed from player 1's perspective, negate the utility
        if traversing_player == 1:
            util = -util

        return util

    def _deal_initial_state(self) -> GameState:
        """
        Deal random cards and create initial game state.

        Returns:
            Initial GameState with cards dealt
        """
        # Reuse deck instance and shuffle
        # Much faster than creating new Deck() which creates new Random()
        self._deck.cards = self._deck_cards[:]
        random.shuffle(self._deck.cards)

        # Deal hole cards
        hole_cards = (
            (Card(self._deck.cards[0]), Card(self._deck.cards[1])),
            (Card(self._deck.cards[2]), Card(self._deck.cards[3])),
        )

        # Create initial state (after blinds)
        state = self.rules.create_initial_state(
            starting_stack=self.starting_stack,
            hole_cards=hole_cards,
            button=self.iteration % 2,
        )

        return state

    def _cfr_external_sampling(
        self,
        state: GameState,
        traversing_player: int,
        reach_probs: List[float],
    ) -> float:
        """
        Recursive MCCFR traversal with external sampling.

        External sampling explores all actions for the traversing player
        and samples a single action for the opponent.

        Args:
            state: Current game state
            traversing_player: Player we're computing regrets for (0 or 1)
            reach_probs: Reach probabilities [p0, p1]

        Returns:
            Expected utility for traversing_player
        """
        # Terminal node - return payoff
        if state.is_terminal:
            # If terminal but board incomplete (all-in before river), deal remaining cards
            if len(state.board) < 5:
                # Deal all remaining cards to complete the board
                complete_state = self._deal_remaining_cards(state)
                return complete_state.get_payoff(traversing_player, self.rules)
            return state.get_payoff(traversing_player, self.rules)

        # Chance node - sample card
        if self._is_chance_node(state):
            next_state = self._sample_chance_outcome(state)
            return self._cfr_external_sampling(next_state, traversing_player, reach_probs)

        # Decision node
        current_player = state.current_player

        # Get infoset
        infoset_key = state.get_infoset_key(current_player, self.card_abstraction)
        legal_actions = self.action_abstraction.get_legal_actions(state)

        if not legal_actions:
            # No legal actions (shouldn't happen)
            raise ValueError(f"No legal actions at state: {state}")

        infoset = self.storage.get_or_create_infoset(infoset_key, legal_actions)

        # Validate stored actions against current state
        # (states with same InfoSetKey might have different stack sizes)
        valid_actions = []
        valid_indices = []
        for i, action in enumerate(infoset.legal_actions):
            try:
                # Test if action is valid for current state
                state.apply_action(action, self.rules)
                valid_actions.append(action)
                valid_indices.append(i)
            except ValueError:
                # Action invalid for current state (e.g., bet exceeds stack)
                pass

        # If no stored actions are valid, use current legal actions
        if not valid_actions:
            valid_actions = legal_actions
            valid_indices = list(range(len(legal_actions)))

        # Filter strategy/regrets to only valid actions
        full_strategy = infoset.get_strategy()
        strategy = full_strategy[valid_indices]

        # Renormalize strategy
        strategy_sum = np.sum(strategy)
        if strategy_sum > 0:
            strategy = strategy / strategy_sum
        else:
            strategy = np.ones(len(valid_actions)) / len(valid_actions)

        # Use valid actions for this traversal
        legal_actions = valid_actions

        if current_player == traversing_player:
            # ON-POLICY: Compute counterfactual values for all actions
            action_utilities = np.zeros(len(legal_actions))

            for i, action in enumerate(legal_actions):
                # Apply action (all actions are pre-validated now)
                next_state = state.apply_action(action, self.rules)

                # Check if we need to deal cards after this action
                if self._is_chance_node(next_state):
                    next_state = self._sample_chance_outcome(next_state)

                # Recursively compute utility
                action_utilities[i] = self._cfr_external_sampling(
                    next_state, traversing_player, reach_probs
                )

            # Node utility (expected value)
            node_utility = np.dot(strategy, action_utilities)

            # Only update regrets/strategy if we OWN this infoset
            # (For partitioned storage: hash(key) % num_workers == worker_id)
            # (For non-partitioned storage: always True)
            if self.storage.is_owned(infoset_key):
                # Update regrets (weighted by opponent reach probability)
                # Map back to original infoset indices
                opponent = 1 - current_player
                for i in range(len(legal_actions)):
                    regret = action_utilities[i] - node_utility
                    original_idx = valid_indices[i]
                    infoset.update_regret(
                        original_idx,
                        regret * reach_probs[opponent],
                        cfr_plus=self.cfr_plus,
                        linear_cfr=self.linear_cfr,
                        iteration=self.iteration,
                    )

                # Update average strategy (weighted by player reach probability)
                # Only update valid actions in the strategy sum
                for i in range(len(legal_actions)):
                    original_idx = valid_indices[i]
                    weight = strategy[i] * reach_probs[current_player]
                    if self.linear_cfr:
                        weight *= self.iteration
                    infoset.strategy_sum[original_idx] += weight
                infoset.reach_count += 1
                infoset.cumulative_utility += node_utility

            return node_utility

        else:
            # OFF-POLICY: Sample single action according to strategy
            action_idx = np.random.choice(len(legal_actions), p=strategy)
            action = legal_actions[action_idx]

            # Update reach probability for sampled action
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[action_idx]

            # Apply action and continue (all actions are pre-validated)
            next_state = state.apply_action(action, self.rules)

            # Check if we need to deal cards after this action
            if self._is_chance_node(next_state):
                next_state = self._sample_chance_outcome(next_state)

            return self._cfr_external_sampling(next_state, traversing_player, new_reach_probs)

    def _cfr_outcome_sampling(
        self,
        state: GameState,
        traversing_player: int,
        reach_probs: List[float],
    ) -> float:
        """
        Recursive MCCFR traversal with outcome sampling.

        Outcome sampling samples a single action for ALL players (including traversing player).
        This is faster but has higher variance than external sampling.

        Args:
            state: Current game state
            traversing_player: Player we're computing regrets for (0 or 1)
            reach_probs: Reach probabilities [p0, p1]

        Returns:
            Expected utility for traversing_player
        """
        # Terminal node - return payoff
        if state.is_terminal:
            # If terminal but board incomplete (all-in before river), deal remaining cards
            if len(state.board) < 5:
                complete_state = self._deal_remaining_cards(state)
                return complete_state.get_payoff(traversing_player, self.rules)
            return state.get_payoff(traversing_player, self.rules)

        # Chance node - sample card
        if self._is_chance_node(state):
            next_state = self._sample_chance_outcome(state)
            return self._cfr_outcome_sampling(next_state, traversing_player, reach_probs)

        # Decision node
        current_player = state.current_player

        # Get infoset
        infoset_key = state.get_infoset_key(current_player, self.card_abstraction)
        legal_actions = self.action_abstraction.get_legal_actions(state)

        if not legal_actions:
            raise ValueError(f"No legal actions at state: {state}")

        infoset = self.storage.get_or_create_infoset(infoset_key, legal_actions)

        # Validate stored actions against current state
        valid_actions = []
        valid_indices = []
        for i, action in enumerate(infoset.legal_actions):
            try:
                state.apply_action(action, self.rules)
                valid_actions.append(action)
                valid_indices.append(i)
            except ValueError:
                pass

        # If no stored actions are valid, use current legal actions
        if not valid_actions:
            valid_actions = legal_actions
            valid_indices = list(range(len(legal_actions)))

        # Filter strategy/regrets to only valid actions
        full_strategy = infoset.get_strategy()
        strategy = full_strategy[valid_indices]

        # Renormalize strategy
        strategy_sum = np.sum(strategy)
        if strategy_sum > 0:
            strategy = strategy / strategy_sum
        else:
            strategy = np.ones(len(valid_actions)) / len(valid_actions)

        legal_actions = valid_actions

        # Sample action for current player (regardless of traversing player)
        action_idx = np.random.choice(len(legal_actions), p=strategy)
        action = legal_actions[action_idx]

        # Apply action
        next_state = state.apply_action(action, self.rules)

        # Check if we need to deal cards after this action
        if self._is_chance_node(next_state):
            next_state = self._sample_chance_outcome(next_state)

        # Recursively compute utility for sampled action
        sampled_utility = self._cfr_outcome_sampling(next_state, traversing_player, reach_probs)

        # Update regrets only for traversing player AND only if we own the infoset
        if current_player == traversing_player and self.storage.is_owned(infoset_key):
            # Standard outcome sampling with proper baseline (Lanctot 2009)
            # Uses importance sampling with a baseline to remain unbiased and reduce variance
            opponent = 1 - current_player

            # Compute baseline: expected value under current strategy
            # Weighted baseline: baseline = u * σ(a) for faster practical convergence
            # This updates both sampled and unsampled actions, providing smoother learning
            baseline = sampled_utility * strategy[action_idx]

            # For each action, compute regret using the unbiased estimator
            for i in range(len(legal_actions)):
                original_idx = valid_indices[i]

                if strategy[i] <= 0:
                    # Skip actions with zero probability (can't importance sample them)
                    continue

                # Importance weight: opponent reach / sampling probability
                w = reach_probs[opponent] / strategy[i]

                if i == action_idx:
                    # Sampled action: regret = (utility - baseline) / σ(a) * π_{-i}
                    # Simplified to: (u - baseline) * w
                    regret = (sampled_utility - baseline) * w
                else:
                    # Unsampled actions: regret = -baseline / σ(a) * π_{-i}
                    # Simplified to: -baseline * w
                    regret = -baseline * w

                infoset.update_regret(
                    original_idx,
                    regret,
                    cfr_plus=self.cfr_plus,
                    linear_cfr=self.linear_cfr,
                    iteration=self.iteration,
                )

            # Update average strategy (weighted by player reach probability)
            for i in range(len(legal_actions)):
                original_idx = valid_indices[i]
                weight = strategy[i] * reach_probs[current_player]
                if self.linear_cfr:
                    weight *= self.iteration
                infoset.strategy_sum[original_idx] += weight

            infoset.reach_count += 1
            infoset.cumulative_utility += sampled_utility

        return sampled_utility

    def _is_chance_node(self, state: GameState) -> bool:
        """
        Check if state is a chance node (needs to deal cards).

        Returns:
            True if cards need to be dealt
        """
        # Check if board size doesn't match street
        expected_board_size = {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
        }

        # If board doesn't match expected size for this street, need to deal
        return len(state.board) < expected_board_size[state.street]

    def _street_action_complete(self, state: GameState) -> bool:
        """
        Check if betting action is complete on current street.

        Returns:
            True if street is complete and should advance
        """
        if len(state.betting_history) == 0:
            return False

        # If someone is all-in, action is complete
        if state.is_all_in():
            return True

        # Check if both players have acted and betting is closed
        # This is simplified - full logic would track actions per street
        if state.to_call == 0:
            # No bet facing, both checked
            return len(state.betting_history) >= 2

        # Someone bet, other called/folded
        last_action = state.betting_history[-1] if state.betting_history else None
        if last_action and last_action.type in (ActionType.CALL, ActionType.FOLD):
            return True

        return False

    def _sample_chance_outcome(self, state: GameState) -> GameState:
        """
        Sample cards to deal at chance node.

        Args:
            state: Current state needing cards

        Returns:
            New state with cards dealt
        """
        # Reuse deck instance
        self._deck.cards = self._deck_cards[:]
        known_cards = set()

        # Remove hole cards
        for player_cards in state.hole_cards:
            for card in player_cards:
                known_cards.add(card.card_int)

        # Remove board cards
        for card in state.board:
            known_cards.add(card.card_int)

        # Filter deck
        self._deck.cards = [c for c in self._deck.cards if c not in known_cards]
        random.shuffle(self._deck.cards)

        # Deal cards based on what's missing
        new_board = list(state.board)
        current_board_size = len(state.board)

        if state.street == Street.FLOP and current_board_size == 0:
            # Deal flop (3 cards)
            new_board.extend([Card(c) for c in self._deck.cards[:3]])
        elif state.street == Street.TURN and current_board_size == 3:
            # Deal turn (1 card)
            new_board.append(Card(self._deck.cards[0]))
        elif state.street == Street.RIVER and current_board_size == 4:
            # Deal river (1 card)
            new_board.append(Card(self._deck.cards[0]))
        else:
            # No cards to deal or already complete
            return state

        # Out of position acts first postflop
        first_to_act = 1 - state.button_position

        return GameState(
            street=state.street,  # Keep same street
            pot=state.pot,
            stacks=state.stacks,
            board=tuple(new_board),
            hole_cards=state.hole_cards,
            betting_history=state.betting_history,
            button_position=state.button_position,
            current_player=first_to_act,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
        )

    def _deal_remaining_cards(self, state: GameState) -> GameState:
        """
        Deal all remaining board cards for showdown (all-in situation).

        Args:
            state: Terminal state with incomplete board

        Returns:
            State with complete 5-card board
        """
        # Reuse deck instance
        self._deck.cards = self._deck_cards[:]
        known_cards = set()

        for player_cards in state.hole_cards:
            for card in player_cards:
                known_cards.add(card.card_int)

        for card in state.board:
            known_cards.add(card.card_int)

        self._deck.cards = [c for c in self._deck.cards if c not in known_cards]
        random.shuffle(self._deck.cards)

        # Deal remaining cards to complete board
        new_board = list(state.board)
        cards_needed = 5 - len(state.board)

        new_board.extend([Card(c) for c in self._deck.cards[:cards_needed]])

        # Return state with complete board
        return GameState(
            street=Street.RIVER,  # Always river at showdown
            pot=state.pot,
            stacks=state.stacks,
            board=tuple(new_board),
            hole_cards=state.hole_cards,
            betting_history=state.betting_history,
            button_position=state.button_position,
            current_player=state.current_player,
            is_terminal=True,
            to_call=0,
            last_aggressor=state.last_aggressor,
        )

    def __str__(self) -> str:
        return (
            f"MCCFRSolver(iteration={self.iteration}, infosets={self.num_infosets()}, "
            f"sampling={self.sampling_method}, stack={self.starting_stack})"
        )
