"""
Game rules and state transitions for Heads-Up No-Limit Hold'em.

This module implements the betting rules, pot calculations, and state transitions
that govern how the game progresses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.game.actions import Action, ActionType, all_in, bet, call, check, fold, raises
from src.core.game.evaluator import get_evaluator
from src.core.game.state import Card, GameState, Street

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel


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

    def _stacks_to_tuple(self, stacks: list[int]) -> tuple[int, int]:
        """Convert stacks list to fixed-size tuple for type safety."""
        return (stacks[0], stacks[1])

    def create_initial_state(
        self,
        starting_stack: int,
        hole_cards: tuple[tuple[Card, Card], tuple[Card, Card]],
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
            stacks=self._stacks_to_tuple(stacks),
            board=tuple(),
            hole_cards=hole_cards,
            betting_history=tuple(),
            button_position=button,
            current_player=current_player,
            is_terminal=False,
            to_call=to_call,
            last_aggressor=bb_player,  # BB is initial aggressor preflop
            blind_to_call=to_call,  # Preflop blind gap (BB - SB), constant for the hand
        )

    def is_action_valid(self, state: GameState, action: Action) -> bool:
        """
        Check if an action is valid for the current state without creating a new state.

        This is a lightweight validation method for use during MCCFR traversal
        where we need to filter stored actions against the current state's constraints
        (e.g., stack sizes may differ for states with the same InfoSetKey).

        Args:
            state: Current game state
            action: Action to validate

        Returns:
            True if the action is valid, False otherwise
        """
        if state.is_terminal:
            return False

        current_stack = state.stacks[state.current_player]
        to_call = state.to_call

        # Basic legality checks (same as apply_action but without state creation)
        if action.type == ActionType.FOLD:
            # Can only fold when facing a bet
            return to_call > 0

        elif action.type == ActionType.CHECK:
            # Can only check when not facing a bet
            return to_call == 0

        elif action.type == ActionType.CALL:
            # Can only call when facing a bet and have chips
            return to_call > 0 and current_stack > 0

        elif action.type == ActionType.BET:
            # Can only bet when not facing a bet and have enough chips
            if to_call != 0:
                return False
            return action.amount <= current_stack and action.amount > 0

        elif action.type == ActionType.RAISE:
            # Can only raise when facing a bet and have enough chips for call + raise
            if to_call == 0:
                return False
            total_needed = to_call + action.amount
            return total_needed <= current_stack and action.amount > 0

        elif action.type == ActionType.ALL_IN:
            # All-in is always valid if we have chips
            return current_stack > 0 and action.amount == current_stack

        return False

    def get_legal_actions(
        self, state: GameState, action_model: ActionModel | None = None
    ) -> list[Action]:
        """
        Get all legal actions for the current player.

        Args:
            state: Current game state
            action_model: Optional action model to discretize actions

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
            if action_model:
                bet_sizes = action_model.get_bet_sizes(state)
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
            if action_model:
                raise_sizes = action_model.get_raise_sizes(state)
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
        to_call = state.to_call

        # Basic action legality checks that don't require an abstraction
        if action.type == ActionType.FOLD and to_call == 0:
            raise ValueError("Cannot fold when no bet is faced")
        if action.type == ActionType.CALL and to_call == 0:
            raise ValueError("Cannot call when no bet is faced")
        if action.type == ActionType.BET and to_call != 0:
            raise ValueError("Cannot bet when facing a bet")
        if action.type == ActionType.RAISE and to_call == 0:
            raise ValueError("Cannot raise when no bet is faced")

        # Copy mutable state
        pot = state.pot
        stacks = list(state.stacks)
        betting_history: list[Action] = list(state.betting_history)
        betting_history.append(action)

        # Handle different action types
        if action.type == ActionType.FOLD:
            # Opponent wins (winner determined by last action being FOLD)
            return self._create_terminal_state(state, tuple(betting_history))

        elif action.type == ActionType.CHECK:
            # Check is only legal if to_call == 0
            if to_call != 0:
                raise ValueError("Cannot check when facing a bet")

            # Get actions on current street to check for check-check
            # Note: betting_history already includes the current action
            actions_this_street = self._get_actions_on_current_street(betting_history)

            # Check-check: both players have checked on this street
            if len(actions_this_street) >= 2:
                # At least 2 actions on this street, check if last two are both checks
                if (
                    actions_this_street[-1].type == ActionType.CHECK
                    and actions_this_street[-2].type == ActionType.CHECK
                ):
                    # Check-check: advance to next street
                    return self._advance_street(state, tuple(betting_history))

            # First check or not check-check: pass action to opponent
            return state.replace(
                betting_history=tuple(betting_history),
                current_player=opponent,
                to_call=0,
                last_aggressor=None,  # No aggression on this street
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
                return self._advance_to_showdown(
                    state, tuple(betting_history), pot, self._stacks_to_tuple(stacks)
                )
            else:
                # Move to next street
                return self._advance_street(
                    state, tuple(betting_history), pot=pot, stacks=self._stacks_to_tuple(stacks)
                )

        elif action.type in (ActionType.BET, ActionType.RAISE):
            # Add chips to pot
            if action.type == ActionType.BET:
                bet_amount = action.amount
            else:  # RAISE
                bet_amount = to_call + action.amount

            if bet_amount <= 0:
                raise ValueError("Bet amount must be positive")

            # Reject oversize bets instead of silently converting to all-in
            if bet_amount > stacks[current_player]:
                raise ValueError("Bet exceeds available stack")

            stacks[current_player] -= bet_amount
            pot += bet_amount

            # Opponent now needs to call this amount
            new_to_call = bet_amount if action.type == ActionType.BET else action.amount

            return state.replace(
                pot=pot,
                stacks=self._stacks_to_tuple(stacks),
                betting_history=tuple(betting_history),
                current_player=opponent,
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
                            state, tuple(betting_history), pot, self._stacks_to_tuple(stacks)
                        )
                    else:
                        # All-in raise
                        return state.replace(
                            pot=pot,
                            stacks=self._stacks_to_tuple(stacks),
                            betting_history=tuple(betting_history),
                            current_player=opponent,
                            to_call=new_to_call,
                            last_aggressor=current_player,
                        )
                else:
                    # All-in for less than call, treat as call
                    return self._advance_to_showdown(
                        state, tuple(betting_history), pot, self._stacks_to_tuple(stacks)
                    )
            else:
                # All-in bet
                return state.replace(
                    pot=pot,
                    stacks=self._stacks_to_tuple(stacks),
                    betting_history=tuple(betting_history),
                    current_player=opponent,
                    to_call=all_in_amount,
                    last_aggressor=current_player,
                )

        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def _get_actions_on_current_street(self, betting_history: list[Action]) -> list[Action]:
        """
        Extract actions that occurred on the current street.

        A street is closed by exactly one of the engine's advancement paths:
        - CALL (always advances the street / goes to showdown; see ``apply_action``)
        - Two consecutive CHECKs (check-check advances the street)

        We reconstruct the current street by replaying ``betting_history``
        *forward*, resetting the accumulator whenever one of those closers fires
        on a non-final action. Forward replay is consistent-by-construction with
        the engine's own advancement and, unlike a backward heuristic, can never
        straddle a boundary: a reset clears the accumulator, so a check-check
        pair is only ever matched within a single street. (A backward scan
        mismatched the boundary-straddling pair when two consecutive streets both
        ended check-check, leaving the second street unable to advance.)

        The final action is never treated as a closer here: it is the in-progress
        action whose closer status the caller is deciding.

        Returns actions in chronological order (oldest first).
        """
        actions_this_street: list[Action] = []
        last_index = len(betting_history) - 1

        for index, action in enumerate(betting_history):
            actions_this_street.append(action)

            closes_street = action.type == ActionType.CALL or (
                len(actions_this_street) >= 2
                and actions_this_street[-1].type == ActionType.CHECK
                and actions_this_street[-2].type == ActionType.CHECK
            )
            if closes_street and index != last_index:
                actions_this_street = []

        return actions_this_street

    def _advance_street(
        self,
        state: GameState,
        betting_history: tuple[Action, ...],
        pot: int | None = None,
        stacks: tuple[int, int] | None = None,
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

        return state.replace(
            street=next_street,
            pot=state.pot if pot is None else pot,
            stacks=state.stacks if stacks is None else stacks,
            betting_history=betting_history,
            current_player=first_to_act,
            to_call=0,
            last_aggressor=None,
            validate=False,  # Board is dealt by the caller (CFR) after the street advances
        )

    def _advance_to_showdown(
        self,
        state: GameState,
        betting_history: tuple[Action, ...],
        pot: int,
        stacks: tuple[int, int],
    ) -> GameState:
        """Advance directly to showdown (player all-in)."""
        # Run out remaining board cards and determine winner
        return self._create_showdown_state(state, betting_history, pot, stacks)

    def _create_showdown_state(
        self,
        state: GameState,
        betting_history: tuple[Action, ...],
        pot: int,
        stacks: tuple[int, int],
    ) -> GameState:
        """Create terminal showdown state and determine winner."""
        # Note: Board might not be complete (all-in before river)
        # CFR solver will deal remaining cards before evaluating
        # So we just create a terminal state without validating board size
        if len(state.board) < 5:
            # Return state signaling showdown needed but board incomplete
            # CFR will deal remaining cards
            return state.replace(
                pot=pot,
                stacks=stacks,
                betting_history=betting_history,
                is_terminal=True,
                to_call=0,
                validate=False,  # Board may be incomplete
            )

        # Winner computed on-demand via get_payoff() using hand evaluation
        return self._create_terminal_state(state, betting_history, pot, stacks)

    def _determine_winner(
        self,
        hole_cards: tuple[tuple[Card, Card], tuple[Card, Card]],
        board: tuple[Card, ...],
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
        betting_history: tuple[Action, ...],
        pot: int | None = None,
        stacks: tuple[int, int] | None = None,
    ) -> GameState:
        """
        Create terminal state.

        Note: Winner is not stored - it's computed on-demand from the state
        via get_payoff() using either the last action (for folds) or
        hand evaluation (for showdowns).
        """
        return state.replace(
            pot=state.pot if pot is None else pot,
            stacks=state.stacks if stacks is None else stacks,
            betting_history=betting_history,
            is_terminal=True,
            to_call=0,
        )

    @staticmethod
    def starting_stack(state: GameState) -> float:
        """Per-player starting stack, reconstructed from pot conservation
        (both players start the hand with the same stack)."""
        return (state.pot + state.stacks[0] + state.stacks[1]) / 2

    @staticmethod
    def invested_chips(state: GameState) -> tuple[float, float]:
        """Chips each player has committed this hand.

        Reconstructed from pot conservation: both players start with the same
        stack, so ``starting = (pot + s0 + s1) / 2`` and invested = starting - s_i.
        """
        starting = GameRules.starting_stack(state)
        return starting - state.stacks[0], starting - state.stacks[1]

    def get_payoff(self, state: GameState, player: int) -> float:
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

        starting_stack = self.starting_stack(state)

        # Determine winner if fold or showdown
        if state.ended_by_fold:
            # current_player is the folder at time of fold
            winner = state.opponent(state.current_player)
        else:
            winner = self._determine_winner(state.hole_cards, state.board)

        if winner == -1:
            # Split pot on tie
            pot_share = state.pot / 2
            total_chips = state.stacks[player] + pot_share
        elif player == winner:
            total_chips = state.stacks[player] + state.pot
        else:
            total_chips = state.stacks[player]

        return total_chips - starting_stack
