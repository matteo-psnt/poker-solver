"""
Head-to-head evaluation for poker strategies.

Plays matches between two strategies and collects statistics on wins,
losses, and chip distributions.
"""

from dataclasses import dataclass

import eval7
import numpy as np

from src.actions.action_model import ActionModel
from src.bucketing.base import BucketingStrategy
from src.game.actions import Action, ActionType
from src.game.rules import GameRules
from src.game.state import Card, GameState, Street
from src.solver.infoset_encoder import encode_infoset_key
from src.solver.mccfr import MCCFRSolver


@dataclass
class MatchResult:
    """
    Result of a single hand.

    Attributes:
        player0_payoff: Chips won/lost by player 0 (positive = won)
        player1_payoff: Chips won/lost by player 1
        hand_number: Hand number in match
        final_street: Street where hand ended
        showdown: Whether hand went to showdown
    """

    player0_payoff: float
    player1_payoff: float
    hand_number: int
    final_street: Street
    showdown: bool


@dataclass
class MatchStatistics:
    """
    Statistics for a match between two strategies.

    Attributes:
        num_hands: Number of hands played
        player0_wins: Number of hands player 0 won (ties count as 0.5)
        player1_wins: Number of hands player 1 won (ties count as 0.5)
        player0_total_won: Total chips won by player 0
        player1_total_won: Total chips won by player 1
        player0_bb_per_hand: Big blinds per hand for player 0
        player1_bb_per_hand: Big blinds per hand for player 1
        showdown_pct: Percentage of hands that went to showdown
        results: List of individual hand results
    """

    num_hands: int
    player0_wins: float
    player1_wins: float
    player0_total_won: float
    player1_total_won: float
    player0_bb_per_hand: float
    player1_bb_per_hand: float
    showdown_pct: float
    results: list[MatchResult]

    def __str__(self) -> str:
        """Format statistics as string."""
        return (
            f"MatchStatistics(\n"
            f"  Hands: {self.num_hands}\n"
            f"  Player 0: {self.player0_wins} wins ({self.player0_wins / self.num_hands * 100:.1f}%), "
            f"{self.player0_bb_per_hand:+.2f} bb/hand\n"
            f"  Player 1: {self.player1_wins} wins ({self.player1_wins / self.num_hands * 100:.1f}%), "
            f"{self.player1_bb_per_hand:+.2f} bb/hand\n"
            f"  Showdown: {self.showdown_pct:.1f}%\n"
            f")"
        )


class HeadToHeadEvaluator:
    """
    Evaluates two strategies by playing heads-up matches.

    Can compare:
    - Two trained solvers
    - Solver vs baseline (random, always fold, etc.)
    - Different checkpoints of same solver
    """

    def __init__(
        self,
        rules: GameRules,
        action_model: ActionModel | None = None,
        card_abstraction: BucketingStrategy | None = None,
        starting_stack: int = 200,
    ):
        """
        Initialize evaluator.

        Args:
            rules: Game rules
            action_model: Action model
            card_abstraction: Card abstraction
            starting_stack: Starting stack size in chips
        """
        if action_model is None:
            raise ValueError("action_model is required")
        if card_abstraction is None:
            raise ValueError("card_abstraction is required")
        self.rules = rules
        self.action_model = action_model
        self.card_abstraction = card_abstraction
        self.starting_stack = starting_stack

    def play_match(
        self,
        solver0: MCCFRSolver,
        solver1: MCCFRSolver,
        num_hands: int,
        alternate_button: bool = True,
        seed: int | None = None,
    ) -> MatchStatistics:
        """
        Play a match between two solvers.

        Args:
            solver0: First solver (player 0)
            solver1: Second solver (player 1)
            num_hands: Number of hands to play
            alternate_button: Whether to alternate button position
            seed: Random seed for reproducibility

        Returns:
            Match statistics
        """
        if seed is not None:
            np.random.seed(seed)

        results = []
        player0_wins = 0.0
        player1_wins = 0.0
        player0_total_won = 0.0
        player1_total_won = 0.0
        showdown_count = 0

        for hand_num in range(num_hands):
            # Determine button
            if alternate_button:
                button = hand_num % 2
            else:
                button = 0

            # Play one hand
            result = self._play_hand(solver0, solver1, button, hand_num)
            results.append(result)

            # Update statistics
            if result.player0_payoff > 0:
                player0_wins += 1.0
            elif result.player1_payoff > 0:
                player1_wins += 1.0
            else:
                # Split pot; count tie as half win for each player.
                player0_wins += 0.5
                player1_wins += 0.5

            player0_total_won += result.player0_payoff
            player1_total_won += result.player1_payoff

            if result.showdown:
                showdown_count += 1

        # Calculate statistics
        bb = self.rules.big_blind
        player0_bb_per_hand = player0_total_won / (num_hands * bb)
        player1_bb_per_hand = player1_total_won / (num_hands * bb)
        showdown_pct = (showdown_count / num_hands) * 100

        return MatchStatistics(
            num_hands=num_hands,
            player0_wins=player0_wins,
            player1_wins=player1_wins,
            player0_total_won=player0_total_won,
            player1_total_won=player1_total_won,
            player0_bb_per_hand=player0_bb_per_hand,
            player1_bb_per_hand=player1_bb_per_hand,
            showdown_pct=showdown_pct,
            results=results,
        )

    def _play_hand(
        self,
        solver0: MCCFRSolver,
        solver1: MCCFRSolver,
        button: int,
        hand_num: int,
    ) -> MatchResult:
        """
        Play a single hand.

        Args:
            solver0: Player 0 solver
            solver1: Player 1 solver
            button: Button position (0 or 1)
            hand_num: Hand number

        Returns:
            MatchResult for this hand
        """
        # Deal cards
        state = self._deal_initial_state(button)

        # Play until terminal
        showdown = False
        while not state.is_terminal:
            current_player = state.current_player

            # Get action from appropriate solver
            if current_player == 0:
                action = self._get_solver_action(solver0, state, current_player)
            else:
                action = self._get_solver_action(solver1, state, current_player)

            # Apply action
            state = state.apply_action(action, self.rules)

            # Check if we need to deal cards (chance node)
            if self._is_chance_node(state):
                state = self._deal_next_cards(state)

        # Determine if showdown
        showdown = False
        if state.betting_history:
            last_action = state.betting_history[-1]
            showdown = last_action.type != ActionType.FOLD

        # If showdown and board incomplete, deal remaining cards
        if showdown and len(state.board) < 5:
            state = self._deal_remaining_cards(state)

        # Get payoffs
        player0_payoff = state.get_payoff(0, self.rules)
        player1_payoff = state.get_payoff(1, self.rules)

        return MatchResult(
            player0_payoff=player0_payoff,
            player1_payoff=player1_payoff,
            hand_number=hand_num,
            final_street=state.street,
            showdown=showdown,
        )

    def _get_solver_action(
        self,
        solver: MCCFRSolver,
        state: GameState,
        player: int,
    ) -> Action:
        """
        Get action from solver for current state.

        Args:
            solver: Solver to query
            state: Current game state
            player: Player making decision

        Returns:
            Chosen action
        """
        # Get infoset
        infoset_key = encode_infoset_key(state, player, self.card_abstraction)
        infoset = solver.storage.get_infoset(infoset_key)

        if infoset is None:
            # No strategy for this infoset, use random legal action
            legal_actions = self.action_model.get_legal_actions(state)
            return legal_actions[np.random.choice(len(legal_actions))]

        # Filter stored actions to those legal in the concrete state.
        valid_indices: list[int] = []
        legal_actions: list[Action] = []
        for i, action in enumerate(infoset.legal_actions):
            if self.rules.is_action_valid(state, action):
                valid_indices.append(i)
                legal_actions.append(action)

        if not legal_actions:
            legal_actions = self.action_model.get_legal_actions(state)
            return legal_actions[np.random.choice(len(legal_actions))]

        strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=True)

        # Sample action according to strategy
        action_idx = int(np.random.choice(len(legal_actions), p=strategy))
        return legal_actions[action_idx]

    def _deal_initial_state(self, button: int) -> GameState:
        """Deal initial state with random cards."""
        # Create and shuffle deck
        deck = eval7.Deck()
        np.random.shuffle(deck.cards)

        # Deal hole cards (wrap eval7.Card in our Card wrapper)
        hole_cards = (
            (Card(deck.cards[0]), Card(deck.cards[1])),
            (Card(deck.cards[2]), Card(deck.cards[3])),
        )

        # Create initial state
        return self.rules.create_initial_state(
            starting_stack=self.starting_stack,
            hole_cards=hole_cards,
            button=button,
        )

    def _is_chance_node(self, state: GameState) -> bool:
        """Check if need to deal cards."""
        expected_board_size = {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
        }
        return len(state.board) < expected_board_size[state.street]

    def _deal_next_cards(self, state: GameState) -> GameState:
        """Deal next community cards."""
        # Create deck and remove known cards
        deck = eval7.Deck()
        known_cards = set()

        for player_cards in state.hole_cards:
            for card in player_cards:
                known_cards.add(card.mask)

        for card in state.board:
            known_cards.add(card.mask)

        deck.cards = [c for c in deck.cards if c.mask not in known_cards]
        np.random.shuffle(deck.cards)

        # Deal cards based on street
        new_board = list(state.board)
        current_board_size = len(state.board)

        if state.street == Street.FLOP and current_board_size == 0:
            new_board.extend([Card(c) for c in deck.cards[:3]])
        elif state.street == Street.TURN and current_board_size == 3:
            new_board.append(Card(deck.cards[0]))
        elif state.street == Street.RIVER and current_board_size == 4:
            new_board.append(Card(deck.cards[0]))
        else:
            return state

        # Out of position acts first postflop
        first_to_act = 1 - state.button_position

        return GameState(
            street=state.street,
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
        """Deal all remaining board cards for showdown (all-in situation)."""
        # Create deck and remove known cards
        deck = eval7.Deck()
        known_cards = set()

        for player_cards in state.hole_cards:
            for card in player_cards:
                known_cards.add(card.mask)

        for card in state.board:
            known_cards.add(card.mask)

        deck.cards = [c for c in deck.cards if c.mask not in known_cards]
        np.random.shuffle(deck.cards)

        # Deal remaining cards to complete board
        new_board = list(state.board)
        cards_needed = 5 - len(state.board)

        new_board.extend([Card(c) for c in deck.cards[:cards_needed]])

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
