"""
Monte Carlo Counterfactual Regret Minimization (MCCFR) solver.

Implements MCCFR with external sampling (default) or outcome sampling for scalable poker solving.
"""

import random

import eval7
import numpy as np

from src.actions.action_model import ActionModel
from src.bucketing.base import BucketingStrategy
from src.game.actions import Action
from src.game.rules import GameRules
from src.game.state import Card, GameState, Street
from src.solver.infoset_encoder import encode_infoset_key
from src.solver.storage.base import Storage
from src.utils.config import Config
from src.utils.numba_ops import compute_dcfr_strategy_weight


class MCCFRSolver:
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
        action_model: ActionModel,
        card_abstraction: BucketingStrategy,
        storage: Storage,
        config: Config,
    ):
        """
        Initialize MCCFR solver.

        Args:
            action_model: Action model
            card_abstraction: Card abstraction
            storage: Storage backend
            config: Config object
        """
        self.action_model = action_model
        self.card_abstraction = card_abstraction
        self.storage = storage
        self.config = config

        # Training state
        self.iteration = 0

        # Game rules
        self.rules = GameRules(self.config.game.small_blind, self.config.game.big_blind)

        # RNG
        if self.config.system.seed is not None:
            random.seed(self.config.system.seed)
            np.random.seed(self.config.system.seed)

        # Cache: Reuse deck instance to avoid repeated Random() creation
        self._deck = eval7.Deck()
        self._deck_cards = list(self._deck.cards)  # Keep original card list

    def checkpoint(self) -> None:
        """Save a checkpoint of the current solver state."""
        self.storage.checkpoint(self.iteration)

    def num_infosets(self) -> int:
        """Get total number of infosets discovered."""
        return self.storage.num_infosets()

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

        if self.config.solver.sampling_method == "external":
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
        self._deck.cards = list(self._deck_cards)
        random.shuffle(self._deck.cards)

        # Deal hole cards (wrap eval7.Card in our Card wrapper)
        hole_cards = (
            (Card(self._deck.cards[0]), Card(self._deck.cards[1])),
            (Card(self._deck.cards[2]), Card(self._deck.cards[3])),
        )

        # Create initial state (after blinds)
        state = self.rules.create_initial_state(
            starting_stack=self.config.game.starting_stack,
            hole_cards=hole_cards,
            button=(self.iteration // 2)
            % 2,  # Alternate every 2 iterations (decoupled from traversing_player)
        )

        return state

    def _cfr_external_sampling(
        self,
        state: GameState,
        traversing_player: int,
        reach_probs: list[float],
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
        infoset_key = encode_infoset_key(state, current_player, self.card_abstraction)
        legal_actions = self.action_model.get_legal_actions(state)

        if not legal_actions:
            # No legal actions (shouldn't happen)
            raise ValueError(f"No legal actions at state: {state}")

        infoset = self.storage.get_or_create_infoset(infoset_key, legal_actions)

        # Validate stored actions against current state
        # (states with same InfoSetKey might have different stack sizes)
        # Use lightweight validation to avoid creating new state objects
        valid_actions = []
        valid_indices = []
        for i, action in enumerate(infoset.legal_actions):
            if self.rules.is_action_valid(state, action):
                valid_actions.append(action)
                valid_indices.append(i)

        # If no stored actions are valid, use current legal actions
        if not valid_actions:
            valid_actions = legal_actions
            valid_indices = list(range(len(legal_actions)))

        # Get strategy filtered to valid actions (automatically normalized)
        strategy = infoset.get_filtered_strategy(
            valid_indices=valid_indices,
            use_average=False,  # Use current strategy (regret matching) for training
        )

        # Use valid actions for this traversal
        legal_actions = valid_actions

        if current_player == traversing_player:
            # ON-POLICY: Compute counterfactual values for all actions
            action_utilities = np.zeros(len(legal_actions))

            for i, action in enumerate(legal_actions):
                original_idx = valid_indices[i]

                # PRUNING: Skip traversal of pruned actions
                if self.config.solver.enable_pruning and infoset.is_pruned(original_idx):
                    # Keep utility at 0 (don't traverse this branch)
                    continue

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
            # When pruning is enabled, only consider unpruned actions
            if self.config.solver.enable_pruning:
                # Create mask for unpruned actions
                unpruned_mask = np.array(
                    [not infoset.is_pruned(valid_indices[i]) for i in range(len(legal_actions))]
                )

                if np.any(unpruned_mask):
                    # Normalize strategy over unpruned actions only
                    unpruned_strategy = strategy[unpruned_mask]
                    unpruned_strategy_sum = unpruned_strategy.sum()
                    if unpruned_strategy_sum > 0:
                        unpruned_strategy = unpruned_strategy / unpruned_strategy_sum
                    else:
                        # Uniform over unpruned actions
                        unpruned_strategy = np.ones(unpruned_mask.sum()) / unpruned_mask.sum()

                    node_utility = np.dot(unpruned_strategy, action_utilities[unpruned_mask])
                else:
                    # All actions pruned (shouldn't happen due to safety check)
                    node_utility = np.dot(strategy, action_utilities)
            else:
                node_utility = np.dot(strategy, action_utilities)

            # Only update regrets/strategy if we OWN this infoset
            # (For partitioned storage: hash(key) % num_workers == worker_id)
            # (For non-partitioned storage: always True)
            if self.storage.is_owned(infoset_key):
                # Update regrets (weighted by opponent reach probability)
                # Map back to original infoset indices
                opponent = 1 - current_player
                for i in range(len(legal_actions)):
                    original_idx = valid_indices[i]

                    # Skip regret update for pruned actions (preserve negative regret)
                    if self.config.solver.enable_pruning and infoset.is_pruned(original_idx):
                        continue

                    regret = action_utilities[i] - node_utility
                    infoset.update_regret(
                        original_idx,
                        regret * reach_probs[opponent],
                        cfr_plus=self.config.solver.cfr_plus,
                        iteration=self.iteration,
                        iteration_weighting=self.config.solver.iteration_weighting,
                        dcfr_alpha=self.config.solver.dcfr_alpha,
                        dcfr_beta=self.config.solver.dcfr_beta,
                    )

                # Update pruning state after regret updates
                if self.config.solver.enable_pruning:
                    infoset.update_pruning(
                        iteration=self.iteration,
                        pruning_threshold=self.config.solver.pruning_threshold,
                        prune_start_iteration=self.config.solver.prune_start_iteration,
                        prune_reactivate_frequency=self.config.solver.prune_reactivate_frequency,
                    )

                # Update average strategy (weighted by player reach probability)
                # Only update valid actions in the strategy sum
                for i in range(len(legal_actions)):
                    original_idx = valid_indices[i]
                    weight = strategy[i] * reach_probs[current_player]

                    if self.config.solver.iteration_weighting == "dcfr":
                        # DCFR: Apply gamma-weighted discounting
                        gamma_weight = compute_dcfr_strategy_weight(
                            self.iteration, self.config.solver.dcfr_gamma
                        )
                        weight *= gamma_weight
                    elif self.config.solver.iteration_weighting == "linear":
                        # Linear CFR: multiply by iteration
                        weight *= self.iteration

                    infoset.strategy_sum[original_idx] += weight
                infoset.increment_reach_count()
                infoset.add_cumulative_utility(node_utility)

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
        reach_probs: list[float],
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
        infoset_key = encode_infoset_key(state, current_player, self.card_abstraction)
        legal_actions = self.action_model.get_legal_actions(state)

        if not legal_actions:
            raise ValueError(f"No legal actions at state: {state}")

        infoset = self.storage.get_or_create_infoset(infoset_key, legal_actions)

        # Validate stored actions against current state
        # Use lightweight validation to avoid creating new state objects
        valid_actions = []
        valid_indices = []
        for i, action in enumerate(infoset.legal_actions):
            if self.rules.is_action_valid(state, action):
                valid_actions.append(action)
                valid_indices.append(i)

        # If no stored actions are valid, use current legal actions
        if not valid_actions:
            valid_actions = legal_actions
            valid_indices = list(range(len(legal_actions)))

        # Get strategy filtered to valid actions (automatically normalized)
        strategy = infoset.get_filtered_strategy(
            valid_indices=valid_indices,
            use_average=False,  # Use current strategy (regret matching) for training
        )

        legal_actions = valid_actions

        # Sample action for current player (regardless of traversing player)
        action_idx = np.random.choice(len(legal_actions), p=strategy)
        action = legal_actions[action_idx]

        # Apply action
        next_state = state.apply_action(action, self.rules)

        # Check if we need to deal cards after this action
        if self._is_chance_node(next_state):
            next_state = self._sample_chance_outcome(next_state)

        # Update reach probability for sampled action (outcome sampling importance weights)
        new_reach_probs = reach_probs.copy()
        new_reach_probs[current_player] *= strategy[action_idx]

        # Recursively compute utility for sampled action
        sampled_utility = self._cfr_outcome_sampling(next_state, traversing_player, new_reach_probs)

        # Update regrets only for traversing player AND only if we own the infoset
        if current_player == traversing_player and self.storage.is_owned(infoset_key):
            # Standard outcome sampling with proper baseline (Lanctot 2009)
            # Uses importance sampling with a baseline to remain unbiased and reduce variance
            opponent = 1 - current_player

            # Compute baseline: expected value under current strategy
            # Use sampled utility as baseline (unbiased estimator per Lanctot 2009)
            # This provides variance reduction while maintaining unbiased regret estimates
            baseline = sampled_utility

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
                    cfr_plus=self.config.solver.cfr_plus,
                    iteration=self.iteration,
                    iteration_weighting=self.config.solver.iteration_weighting,
                    dcfr_alpha=self.config.solver.dcfr_alpha,
                    dcfr_beta=self.config.solver.dcfr_beta,
                )

            # Update average strategy (weighted by player reach probability)
            for i in range(len(legal_actions)):
                original_idx = valid_indices[i]
                weight = strategy[i] * reach_probs[current_player]

                if self.config.solver.iteration_weighting == "dcfr":
                    # DCFR: Apply gamma-weighted discounting
                    gamma_weight = compute_dcfr_strategy_weight(
                        self.iteration, self.config.solver.dcfr_gamma
                    )
                    weight *= gamma_weight
                elif self.config.solver.iteration_weighting == "linear":
                    # Linear CFR: multiply by iteration
                    weight *= self.iteration

                infoset.strategy_sum[original_idx] += weight

            # Update statistics once per infoset update (not per action)
            infoset.increment_reach_count()
            infoset.add_cumulative_utility(sampled_utility)

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

    def _prepare_shuffled_deck(self, state: GameState) -> None:
        """
        Prepare deck by removing known cards and shuffling.

        Mutates self._deck.cards in place for performance.

        Args:
            state: Current game state with hole cards and board
        """
        self._deck.cards = list(self._deck_cards)
        known_cards = set()

        for player_cards in state.hole_cards:
            for card in player_cards:
                known_cards.add(card.mask)

        for card in state.board:
            known_cards.add(card.mask)

        self._deck.cards = [c for c in self._deck.cards if c.mask not in known_cards]
        random.shuffle(self._deck.cards)

    def _sample_chance_outcome(self, state: GameState) -> GameState:
        """
        Sample cards to deal at chance node.

        Args:
            state: Current state needing cards

        Returns:
            New state with cards dealt
        """
        self._prepare_shuffled_deck(state)

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
        self._prepare_shuffled_deck(state)
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

    def sample_action_from_strategy(self, state: GameState, *, use_average: bool = True) -> Action:
        """
        Sample an action from the blueprint strategy at the current infoset.

        Falls back to uniform random over legal actions for unseen infosets.
        """
        legal_actions = self.action_model.get_legal_actions(state)
        if not legal_actions:
            raise ValueError(f"No legal actions at state: {state}")

        infoset_key = encode_infoset_key(state, state.current_player, self.card_abstraction)
        infoset = self.storage.get_infoset(infoset_key)
        if infoset is None:
            return legal_actions[np.random.choice(len(legal_actions))]

        legal_set = set(legal_actions)
        valid_indices = [i for i, action in enumerate(infoset.legal_actions) if action in legal_set]
        if not valid_indices:
            return legal_actions[np.random.choice(len(legal_actions))]

        filtered_actions = [infoset.legal_actions[i] for i in valid_indices]
        strategy = infoset.get_filtered_strategy(
            valid_indices=valid_indices, use_average=use_average
        )
        action_idx = int(np.random.choice(len(filtered_actions), p=strategy))
        return filtered_actions[action_idx]

    def act(
        self,
        state: GameState,
        *,
        use_resolver: bool | None = None,
        time_budget_ms: int | None = None,
        use_average: bool = True,
    ) -> Action:
        """
        Choose an action from blueprint strategy or runtime subgame resolver.
        """
        if use_resolver is None:
            use_resolver = self.config.resolver.enabled

        if use_resolver:
            # Lazy import to avoid circular dependency when training imports solver.
            from src.search.resolver import HUResolver

            resolver = getattr(self, "_resolver", None)
            if resolver is None:
                resolver = HUResolver(
                    blueprint=self,
                    action_model=self.action_model,
                    rules=self.rules,
                    config=self.config.resolver,
                )
                self._resolver = resolver
            return resolver.act(state, time_budget_ms=time_budget_ms)

        return self.sample_action_from_strategy(state, use_average=use_average)

    def __str__(self) -> str:
        return (
            f"MCCFRSolver(iteration={self.iteration}, infosets={self.num_infosets()}, "
            f"sampling={self.config.solver.sampling_method}, "
            f"stack={self.config.game.starting_stack})"
        )
