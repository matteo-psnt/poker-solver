"""
Action abstraction for poker betting.

This module implements action abstraction that discretizes the continuous
betting space into a small set of actions (e.g., bet 33%, 75%, all-in).
"""

from typing import Dict, List, Optional, Union

from src.game.actions import Action, ActionType, all_in, bet, raises
from src.game.state import GameState, Street


class ActionAbstraction:
    """
    Defines legal betting actions per street using abstraction.

    Action abstraction is critical for making poker tractable. Without it,
    the action space is continuous (any bet size from 1 chip to all-in).
    We discretize to a small set of bet/raise sizes.

    Example configuration:
        Preflop: fold, call, raise 2.5bb, raise 4bb, all-in
        Postflop: fold, call, check, bet 33% pot, bet 75% pot, all-in
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize action abstraction from configuration.

        Args:
            config: Dictionary with abstraction parameters
                   {
                       'preflop_raises': [2.5, 4.0, 'all-in'],  # BB units
                       'postflop_bets': [0.33, 0.75, 'all-in'],  # Pot fractions
                   }
                   If None, uses default abstraction.
        """
        if config is None:
            config = self._default_config()

        self.preflop_raises = config.get("preflop_raises", [2.5, 4.0, "all-in"])
        self.postflop_bets = config.get("postflop_bets", [0.33, 0.75, "all-in"])

        # Convert 'all-in' string to special marker
        self.preflop_raises = [r if r != "all-in" else float("inf") for r in self.preflop_raises]
        self.postflop_bets = [b if b != "all-in" else float("inf") for b in self.postflop_bets]

    @staticmethod
    def _default_config() -> Dict:
        """Get default action abstraction."""
        return {
            "preflop_raises": [2.5, 4.0, "all-in"],  # Raise to 2.5bb, 4bb, or all-in
            "postflop_bets": [0.33, 0.75, "all-in"],  # Bet 33%, 75% pot, or all-in
        }

    def get_bet_sizes(self, state: GameState) -> List[int]:
        """
        Get legal bet sizes for current state.

        Args:
            state: Current game state

        Returns:
            List of bet sizes in chips
        """
        if state.to_call > 0:
            # Cannot bet if facing a bet
            return []

        pot = state.pot
        stack = state.stacks[state.current_player]

        sizes = []

        if state.street.is_preflop():
            # Preflop: use BB-denominated sizes
            big_blind = 2  # TODO: Get from game rules
            for raise_bb in self.preflop_raises:
                if raise_bb == float("inf"):
                    # All-in
                    if stack not in sizes:
                        sizes.append(stack)
                else:
                    bet_size = int(raise_bb * big_blind)
                    if bet_size <= stack and bet_size not in sizes:
                        sizes.append(bet_size)
        else:
            # Postflop: use pot-fraction sizes
            for pot_frac in self.postflop_bets:
                if pot_frac == float("inf"):
                    # All-in
                    if stack not in sizes:
                        sizes.append(stack)
                else:
                    bet_size = int(pot * pot_frac)
                    if bet_size <= stack and bet_size > 0 and bet_size not in sizes:
                        sizes.append(bet_size)

        # Filter out bets that are too small (< 1 chip)
        sizes = [s for s in sizes if s > 0]

        return sorted(sizes)

    def get_raise_sizes(self, state: GameState) -> List[int]:
        """
        Get legal raise sizes for current state.

        Args:
            state: Current game state

        Returns:
            List of raise sizes in chips (on top of call amount)
        """
        if state.to_call == 0:
            # Cannot raise if not facing a bet
            return []

        pot = state.pot
        to_call = state.to_call
        stack = state.stacks[state.current_player]

        if stack <= to_call:
            # Cannot raise, only call or fold
            return []

        sizes = []

        if state.street.is_preflop():
            # Preflop: raise in BB units
            big_blind = 2  # TODO: Get from game rules
            for raise_bb in self.preflop_raises:
                if raise_bb == float("inf"):
                    # All-in raise
                    raise_amount = stack - to_call
                    if raise_amount > 0 and raise_amount not in sizes:
                        sizes.append(raise_amount)
                else:
                    # Standard raise size
                    total_bet = int(raise_bb * big_blind)
                    raise_amount = total_bet - to_call
                    if raise_amount > 0 and total_bet <= stack and raise_amount not in sizes:
                        sizes.append(raise_amount)
        else:
            # Postflop: raise as pot fractions
            for pot_frac in self.postflop_bets:
                if pot_frac == float("inf"):
                    # All-in raise
                    raise_amount = stack - to_call
                    if raise_amount > 0 and raise_amount not in sizes:
                        sizes.append(raise_amount)
                else:
                    # Pot-sized raise
                    raise_size = int(pot * pot_frac)
                    if raise_size > 0 and (to_call + raise_size) <= stack and raise_size not in sizes:
                        sizes.append(raise_size)

        return sorted(sizes)

    def get_legal_actions(self, state: GameState) -> List[Action]:
        """
        Get all legal abstracted actions for current player.

        Args:
            state: Current game state

        Returns:
            List of legal actions according to abstraction
        """
        from src.game.actions import call, check, fold

        if state.is_terminal:
            return []

        actions = []
        stack = state.stacks[state.current_player]

        # Can always fold unless can check for free
        if state.to_call > 0:
            actions.append(fold())

        # Can check if no bet to call
        if state.to_call == 0:
            actions.append(check())

        # Can call if facing a bet
        if state.to_call > 0:
            if state.to_call >= stack:
                # Calling is all-in
                actions.append(all_in(stack))
            else:
                actions.append(call())

        # Get abstracted bet/raise sizes
        if state.to_call == 0:
            # Can bet
            bet_sizes = self.get_bet_sizes(state)
            for size in bet_sizes:
                if size >= stack:
                    # Bet is all-in
                    if not any(a.type == ActionType.ALL_IN for a in actions):
                        actions.append(all_in(stack))
                else:
                    actions.append(bet(size))
        else:
            # Can raise
            raise_sizes = self.get_raise_sizes(state)
            for size in raise_sizes:
                total_needed = state.to_call + size
                if total_needed >= stack:
                    # Raise is all-in
                    if not any(a.type == ActionType.ALL_IN for a in actions):
                        actions.append(all_in(stack))
                else:
                    actions.append(raises(size))

        return actions

    def discretize_action(self, state: GameState, action: Action) -> Action:
        """
        Map a raw action to the nearest abstracted action.

        This is useful when you have a continuous action (e.g., from human play)
        and need to map it to the abstraction for strategy lookup.

        Args:
            state: Current game state
            action: Raw action

        Returns:
            Nearest action in abstraction
        """
        # If action is fold, check, or call, return as-is
        if action.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            return action

        # Get legal abstracted actions
        legal_actions = self.get_legal_actions(state)

        # Find nearest action by amount
        if action.type == ActionType.BET:
            bet_actions = [a for a in legal_actions if a.type == ActionType.BET]
            if not bet_actions:
                # No bet actions, return all-in
                return all_in(state.stacks[state.current_player])
            return min(bet_actions, key=lambda a: abs(a.amount - action.amount))

        elif action.type == ActionType.RAISE:
            raise_actions = [a for a in legal_actions if a.type == ActionType.RAISE]
            if not raise_actions:
                # No raise actions, return all-in
                return all_in(state.stacks[state.current_player])
            return min(raise_actions, key=lambda a: abs(a.amount - action.amount))

        elif action.type == ActionType.ALL_IN:
            # Return all-in action if exists
            all_in_actions = [a for a in legal_actions if a.type == ActionType.ALL_IN]
            if all_in_actions:
                return all_in_actions[0]
            # Otherwise return largest bet/raise
            aggressive = [a for a in legal_actions if a.is_aggressive()]
            if aggressive:
                return max(aggressive, key=lambda a: a.amount)
            return action

        return action

    def __str__(self) -> str:
        """String representation of abstraction."""
        preflop_str = ", ".join(
            ["all-in" if r == float("inf") else f"{r}bb" for r in self.preflop_raises]
        )
        postflop_str = ", ".join(
            ["all-in" if b == float("inf") else f"{int(b*100)}%" for b in self.postflop_bets]
        )
        return f"ActionAbstraction(preflop=[{preflop_str}], postflop=[{postflop_str}])"
