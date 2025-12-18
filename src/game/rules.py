"""
Game rules and state transitions for Heads-Up No-Limit Hold'em.

This module implements the betting rules, pot calculations, and state transitions
that govern how the game progresses.
"""

from typing import List, Optional, Tuple

from src.game.actions import Action, ActionType, all_in, bet, call, check, fold, raises
from src.game.evaluator import get_evaluator
from src.game.state import Card, GameState, Street


class GameRules:
    """
    Rules engine for Heads-Up No-Limit Hold'em.

    This class handles:
    - Legal action generation
    - State transitions
    - Terminal state detection
    - Payoff calculation
    """

    def __init__(self, small_blind: int = 1, big_blind: int = 2):
        """
        Initialize game rules.

        Args:
            small_blind: Small blind size
            big_blind: Big blind size
        """
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.evaluator = get_evaluator()

    def create_initial_state(
        self,
        starting_stack: int,
        hole_cards: Tuple[Tuple[Card, Card], Tuple[Card, Card]],
        button: int = 0,
    ) -> GameState:
        """
        Create initial game state (preflop, after blinds posted).

        Args:
            starting_stack: Starting stack for both players
            hole_cards: Hole cards for (player0, player1)
            button: Which player has the button (0 or 1)

        Returns:
            Initial GameState
        """
        # In heads-up, button posts small blind and acts first preflop
        # Other player posts big blind
        sb_player = button
        bb_player = 1 - button

        # Post blinds
        stacks = [starting_stack, starting_stack]
        stacks[sb_player] -= self.small_blind
        stacks[bb_player] -= self.big_blind

        pot = self.small_blind + self.big_blind

        # Button acts first preflop (facing big blind)
        current_player = button
        to_call = self.big_blind - self.small_blind

        return GameState(
            street=Street.PREFLOP,
            pot=pot,
            stacks=tuple(stacks),
            board=tuple(),
            hole_cards=hole_cards,
            betting_history=tuple(),
            button_position=button,
            current_player=current_player,
            is_terminal=False,
            to_call=to_call,
            last_aggressor=bb_player,  # BB is initial aggressor preflop
        )

    def get_legal_actions(self, state: GameState, action_abstraction=None) -> List[Action]:
        """
        Get all legal actions for the current player.

        Args:
            state: Current game state
            action_abstraction: Optional action abstraction to discretize actions

        Returns:
            List of legal actions
        """
        if state.is_terminal:
            return []

        actions = []
        current_stack = state.stacks[state.current_player]

        # Always can fold (unless can check)
        if state.to_call > 0:
            actions.append(fold())

        # Can check if no bet to call
        if state.can_check():
            actions.append(check())

        # Can call if facing a bet and not all-in
        if state.to_call > 0 and state.to_call < current_stack:
            actions.append(call())

        # Can bet if no current bet
        if state.can_bet():
            if action_abstraction:
                bet_sizes = action_abstraction.get_bet_sizes(state)
                for size in bet_sizes:
                    if size < current_stack:
                        actions.append(bet(size))
                    else:
                        actions.append(all_in(current_stack))
            else:
                # Without abstraction, allow any bet up to stack
                # For now, just add all-in
                actions.append(all_in(current_stack))

        # Can raise if facing a bet
        if state.can_raise():
            if action_abstraction:
                raise_sizes = action_abstraction.get_raise_sizes(state)
                for size in raise_sizes:
                    total_needed = state.to_call + size
                    if total_needed < current_stack:
                        actions.append(raises(size))
                    else:
                        actions.append(all_in(current_stack))
            else:
                # Without abstraction, just allow all-in
                actions.append(all_in(current_stack))

        # Can always go all-in if have chips
        if current_stack > 0:
            # Check if all-in not already added
            if not any(a.type == ActionType.ALL_IN for a in actions):
                actions.append(all_in(current_stack))

        # If facing bet equal to stack, can only fold or call (all-in)
        if state.to_call > 0 and state.to_call >= current_stack:
            actions = [fold(), all_in(current_stack)]

        return actions

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """
        Apply an action to create a new game state.

        Args:
            state: Current game state
            action: Action to apply

        Returns:
            New game state after action
        """
        if state.is_terminal:
            raise ValueError("Cannot apply action to terminal state")

        current_player = state.current_player
        opponent = state.opponent(current_player)

        # Copy mutable state
        pot = state.pot
        stacks = list(state.stacks)
        betting_history = list(state.betting_history)
        betting_history.append(action)

        to_call = state.to_call
        last_aggressor = state.last_aggressor
        street = state.street

        # Handle different action types
        if action.type == ActionType.FOLD:
            # Opponent wins
            return self._create_terminal_state(state, opponent, tuple(betting_history))

        elif action.type == ActionType.CHECK:
            # Check is only legal if to_call == 0
            if to_call != 0:
                raise ValueError("Cannot check when facing a bet")

            # If both players have acted this street, move to next street
            if last_aggressor is None or self._street_complete(state, betting_history):
                return self._advance_street(state, tuple(betting_history))
            else:
                # Pass action to opponent
                return state.__class__(
                    street=street,
                    pot=pot,
                    stacks=tuple(stacks),
                    board=state.board,
                    hole_cards=state.hole_cards,
                    betting_history=tuple(betting_history),
                    button_position=state.button_position,
                    current_player=opponent,
                    is_terminal=False,
                    to_call=0,
                    last_aggressor=last_aggressor,
                )

        elif action.type == ActionType.CALL:
            # Add chips to pot
            call_amount = min(to_call, stacks[current_player])
            stacks[current_player] -= call_amount
            pot += call_amount

            # After call, street betting is complete
            # Check if anyone is all-in
            if stacks[current_player] == 0:
                # Current player all-in, go to showdown
                return self._advance_to_showdown(state, tuple(betting_history), pot, tuple(stacks))
            else:
                # Move to next street
                return self._advance_street(
                    state, tuple(betting_history), pot=pot, stacks=tuple(stacks)
                )

        elif action.type in (ActionType.BET, ActionType.RAISE):
            # Add chips to pot
            if action.type == ActionType.BET:
                bet_amount = action.amount
            else:  # RAISE
                bet_amount = to_call + action.amount

            if bet_amount > stacks[current_player]:
                raise ValueError(f"Bet {bet_amount} exceeds stack {stacks[current_player]}")

            stacks[current_player] -= bet_amount
            pot += bet_amount

            # Opponent now needs to call this amount
            new_to_call = bet_amount if action.type == ActionType.BET else action.amount

            return state.__class__(
                street=street,
                pot=pot,
                stacks=tuple(stacks),
                board=state.board,
                hole_cards=state.hole_cards,
                betting_history=tuple(betting_history),
                button_position=state.button_position,
                current_player=opponent,
                is_terminal=False,
                to_call=new_to_call,
                last_aggressor=current_player,
            )

        elif action.type == ActionType.ALL_IN:
            # Player goes all-in
            all_in_amount = stacks[current_player]
            stacks[current_player] = 0
            pot += all_in_amount

            # Determine if this is a call, bet, or raise
            if to_call > 0:
                if all_in_amount >= to_call:
                    # All-in call or raise
                    new_to_call = max(0, all_in_amount - to_call)
                    if new_to_call == 0:
                        # All-in call, go to showdown
                        return self._advance_to_showdown(
                            state, tuple(betting_history), pot, tuple(stacks)
                        )
                    else:
                        # All-in raise
                        return state.__class__(
                            street=street,
                            pot=pot,
                            stacks=tuple(stacks),
                            board=state.board,
                            hole_cards=state.hole_cards,
                            betting_history=tuple(betting_history),
                            button_position=state.button_position,
                            current_player=opponent,
                            is_terminal=False,
                            to_call=new_to_call,
                            last_aggressor=current_player,
                        )
                else:
                    # All-in for less than call, treat as call
                    return self._advance_to_showdown(
                        state, tuple(betting_history), pot, tuple(stacks)
                    )
            else:
                # All-in bet
                return state.__class__(
                    street=street,
                    pot=pot,
                    stacks=tuple(stacks),
                    board=state.board,
                    hole_cards=state.hole_cards,
                    betting_history=tuple(betting_history),
                    button_position=state.button_position,
                    current_player=opponent,
                    is_terminal=False,
                    to_call=all_in_amount,
                    last_aggressor=current_player,
                )

        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def _street_complete(self, state: GameState, betting_history: List[Action]) -> bool:
        """Check if betting is complete on current street."""
        # Count actions on this street
        actions_this_street = []
        for action in reversed(betting_history):
            actions_this_street.insert(0, action)
            # Stop at street transition (would need to track street changes)
            # For now, simplified: betting complete if last action was check or call

        if len(actions_this_street) == 0:
            return False

        last_action = actions_this_street[-1]
        return last_action.type in (ActionType.CALL, ActionType.CHECK)

    def _advance_street(
        self,
        state: GameState,
        betting_history: Tuple[Action, ...],
        pot: Optional[int] = None,
        stacks: Optional[Tuple[int, int]] = None,
    ) -> GameState:
        """Advance to next betting street."""
        next_street = state.street.next_street()

        if next_street is None:
            # After river, go to showdown
            return self._create_showdown_state(
                state, betting_history, pot or state.pot, stacks or state.stacks
            )

        # Deal next cards (in real game, would deal from deck)
        # For now, board must be provided externally
        # This is a placeholder that will be completed when integrating with actual game loop

        # Out of position player acts first postflop (non-button)
        first_to_act = 1 - state.button_position

        return state.__class__(
            street=next_street,
            pot=pot or state.pot,
            stacks=stacks or state.stacks,
            board=state.board,  # Board will be updated by caller (CFR will deal cards)
            hole_cards=state.hole_cards,
            betting_history=betting_history,
            button_position=state.button_position,
            current_player=first_to_act,
            is_terminal=False,
            to_call=0,
            last_aggressor=None,
            _skip_validation=True,  # Skip validation - board will be dealt by CFR
        )

    def _advance_to_showdown(
        self,
        state: GameState,
        betting_history: Tuple[Action, ...],
        pot: int,
        stacks: Tuple[int, int],
    ) -> GameState:
        """Advance directly to showdown (player all-in)."""
        # Run out remaining board cards and determine winner
        return self._create_showdown_state(state, betting_history, pot, stacks)

    def _create_showdown_state(
        self,
        state: GameState,
        betting_history: Tuple[Action, ...],
        pot: int,
        stacks: Tuple[int, int],
    ) -> GameState:
        """Create terminal showdown state and determine winner."""
        # Note: Board might not be complete (all-in before river)
        # CFR solver will deal remaining cards before evaluating
        # So we just create a terminal state without validating board size
        if len(state.board) < 5:
            # Return state signaling showdown needed but board incomplete
            # CFR will deal remaining cards
            return state.__class__(
                street=Street.RIVER if state.street == Street.RIVER else state.street,
                pot=pot,
                stacks=stacks,
                board=state.board,
                hole_cards=state.hole_cards,
                betting_history=betting_history,
                button_position=state.button_position,
                current_player=state.current_player,
                is_terminal=True,  # Mark as terminal
                to_call=0,
                last_aggressor=state.last_aggressor,
                _skip_validation=True,  # Board may be incomplete
            )

        winner = self._determine_winner(state.hole_cards, state.board)

        return self._create_terminal_state(state, winner, betting_history, pot, stacks)

    def _determine_winner(
        self,
        hole_cards: Tuple[Tuple[Card, Card], Tuple[Card, Card]],
        board: Tuple[Card, ...],
    ) -> int:
        """Determine winner at showdown."""
        result = self.evaluator.compare_hands(hole_cards[0], hole_cards[1], board)
        if result == -1:
            return 0  # Player 0 wins
        elif result == 1:
            return 1  # Player 1 wins
        else:
            return -1  # Tie (will split pot)

    def _create_terminal_state(
        self,
        state: GameState,
        winner: int,
        betting_history: Tuple[Action, ...],
        pot: Optional[int] = None,
        stacks: Optional[Tuple[int, int]] = None,
    ) -> GameState:
        """Create terminal state with payoffs."""
        return state.__class__(
            street=state.street,
            pot=pot or state.pot,
            stacks=stacks or state.stacks,
            board=state.board,
            hole_cards=state.hole_cards,
            betting_history=betting_history,
            button_position=state.button_position,
            current_player=state.current_player,
            is_terminal=True,
            to_call=0,
            last_aggressor=state.last_aggressor,
        )

    def get_payoff(self, state: GameState, player: int) -> int:
        """
        Get payoff for a player in a terminal state.

        Args:
            state: Terminal game state
            player: Player index (0 or 1)

        Returns:
            Chips won/lost (positive = won, negative = lost)
        """
        if not state.is_terminal:
            raise ValueError("Can only get payoff for terminal states")

        # If someone folded, last action tells us
        if state.betting_history and state.betting_history[-1].type == ActionType.FOLD:
            # Player who folded was current_player at time of fold
            # Winner is opponent
            folder = len(state.betting_history) % 2  # Simplified
            winner = 1 - folder

            if player == winner:
                return state.pot
            else:
                # Calculate how much this player lost
                # This is tricky - need to track contributions
                # For now, simplified
                return -state.pot // 2
        else:
            # Showdown
            winner = self._determine_winner(state.hole_cards, state.board)

            if winner == -1:  # Tie
                return 0
            elif player == winner:
                return state.pot
            else:
                return -state.pot // 2
