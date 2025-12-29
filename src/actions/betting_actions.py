"""
Action abstraction for poker betting.

This module implements action abstraction that discretizes the continuous
betting space into a small set of actions (e.g., bet 33%, 75%, all-in).
"""

import json

import xxhash

from src.game.actions import Action, ActionType, all_in, bet, call, check, fold, raises
from src.game.state import GameState


class BettingActions:
    """
    Defines legal betting actions per street using abstraction.

    Action abstraction is critical for making poker tractable. Without it,
    the action space is continuous (any bet size from 1 chip to all-in).
    We discretize to a small set of bet/raise sizes.

    Example configuration:
        Preflop: fold, call, raise 2.5bb, raise 4bb, all-in
        Postflop: fold, call, check, bet 33% pot, bet 75% pot, all-in
    """

    def __init__(self, config: dict | None = None, big_blind: int = 2):
        """
        Initialize action abstraction from configuration.

        Args:
            config: Dictionary with abstraction parameters
                   {
                       'preflop_raises': [2.5, 3.5, 5.0],  # BB units
                       'postflop': {  # Street-dependent
                           'flop': [0.33, 0.66, 1.25],
                           'turn': [0.50, 1.0, 1.5],
                           'river': [0.50, 1.0, 2.0],
                       },
                       'all_in_spr_threshold': 2.0,  # Only allow all-in if SPR < threshold
                   }
                   If None, uses default abstraction.
            big_blind: Big blind size for preflop raise calculations
        """
        if config is None:
            config = self._default_config()

        defaults = self._default_config()

        self.big_blind = big_blind

        # Preflop raises: new format only
        preflop_raises = config.get("preflop_raises")
        if preflop_raises is None:
            preflop_section = config.get("preflop", {})
            if isinstance(preflop_section, dict):
                preflop_raises = preflop_section.get("raises") or preflop_section.get("bets")

        self.preflop_raises = preflop_raises or defaults["preflop_raises"]

        # Street-dependent postflop bets (new format only)
        postflop_config = config.get("postflop")

        self.postflop_bets = {}
        default_postflop = defaults["postflop"]

        if postflop_config is None:
            self.postflop_bets = default_postflop.copy()
        elif isinstance(postflop_config, list):
            for street in ["flop", "turn", "river"]:
                self.postflop_bets[street] = postflop_config
        elif isinstance(postflop_config, dict):
            for street in ["flop", "turn", "river"]:
                street_config = postflop_config.get(street, default_postflop[street])
                if isinstance(street_config, dict):
                    self.postflop_bets[street] = street_config.get("bets") or street_config.get(
                        "sizes", default_postflop[street]
                    )
                else:
                    self.postflop_bets[street] = street_config
        else:
            raise ValueError("Invalid postflop configuration format.")

        # Fill any missing streets with defaults
        for street in ["flop", "turn", "river"]:
            if street not in self.postflop_bets or not self.postflop_bets[street]:
                self.postflop_bets[street] = default_postflop[street]

        # All-in threshold (SPR)
        self.all_in_spr_threshold = config.get(
            "all_in_spr_threshold", defaults["all_in_spr_threshold"]
        )

        # Max raises per street (caps action tree depth)
        self.max_raises_per_street = config.get(
            "max_raises_per_street", defaults["max_raises_per_street"]
        )

    @staticmethod
    def _default_config() -> dict:
        """Get default research-grade action abstraction."""
        return {
            "preflop_raises": [2.5, 3.5, 5.0],  # Standard raise sizes
            "postflop": {
                "flop": [0.33, 0.66, 1.25],  # Range, standard, overbet
                "turn": [0.50, 1.0, 1.5],  # Medium, pot, large
                "river": [0.50, 1.0, 2.0],  # Thin, pot, large overbet
            },
            "all_in_spr_threshold": 2.0,  # Only allow all-in when SPR < 2
            "max_raises_per_street": 4,  # Cap raises per street (prevents infinite action trees)
        }

    def get_bet_sizes(self, state: GameState) -> list[int]:
        """
        Get legal bet sizes for current state.

        Uses street-dependent sizing and SPR-conditional all-in.

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

        # Check raise cap (BET counts as first raise)
        raises_this_street = self._count_raises_on_current_street(state)
        if raises_this_street >= self.max_raises_per_street:
            # Hit raise cap, can only check
            return []

        # Calculate SPR (Stack-to-Pot Ratio)
        spr = stack / pot if pot > 0 else float("inf")

        sizes = []

        if state.street.is_preflop():
            # Preflop: use BB-denominated sizes
            for raise_bb in self.preflop_raises:
                bet_size = int(raise_bb * self.big_blind)
                if bet_size <= stack and bet_size not in sizes:
                    sizes.append(bet_size)
        else:
            # Postflop: use street-dependent pot-fraction sizes
            street_name = state.street.name.lower()
            pot_fractions = self.postflop_bets.get(street_name, [0.33, 0.66, 1.25])

            for pot_frac in pot_fractions:
                bet_size = int(pot * pot_frac)
                if bet_size <= stack and bet_size > 0 and bet_size not in sizes:
                    sizes.append(bet_size)

        # Add all-in only if SPR < threshold
        if spr < self.all_in_spr_threshold:
            if stack not in sizes and stack > 0:
                sizes.append(stack)

        # Filter out bets that are too small (< 1 chip)
        sizes = [s for s in sizes if s > 0]

        return sorted(sizes)

    def _count_raises_on_current_street(self, state: GameState) -> int:
        """
        Count the number of BET/RAISE actions on the current street.

        This is used to cap raises per street and prevent infinite action trees.

        Args:
            state: Current game state

        Returns:
            Number of bets/raises on current street
        """
        raises_count = 0
        betting_history = list(state.betting_history)

        # Walk backwards through history to find current street actions
        for i in range(len(betting_history) - 1, -1, -1):
            action = betting_history[i]

            # Stop if we hit a CALL (previous street ended)
            if action.type == ActionType.CALL:
                break

            # Stop if we hit check-check pattern (previous street ended)
            if (
                i >= 1
                and action.type == ActionType.CHECK
                and betting_history[i - 1].type == ActionType.CHECK
            ):
                break

            # Count BET and RAISE actions
            if action.type in (ActionType.BET, ActionType.RAISE):
                raises_count += 1

        return raises_count

    def get_raise_sizes(self, state: GameState) -> list[int]:
        """
        Get legal raise sizes for current state.

        Uses street-dependent sizing and SPR-conditional all-in.

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

        # Check raise cap (prevent infinite action trees)
        raises_this_street = self._count_raises_on_current_street(state)
        if raises_this_street >= self.max_raises_per_street:
            # Hit raise cap, can only call or fold
            return []

        # Calculate SPR (Stack-to-Pot Ratio)
        spr = stack / pot if pot > 0 else float("inf")

        sizes = []

        if state.street.is_preflop():
            # Preflop: raise in BB units
            for raise_bb in self.preflop_raises:
                # Standard raise size
                total_bet = int(raise_bb * self.big_blind)
                raise_amount = total_bet - to_call
                if raise_amount > 0 and total_bet <= stack and raise_amount not in sizes:
                    sizes.append(raise_amount)
        else:
            # Postflop: street-dependent raise as pot fractions
            street_name = state.street.name.lower()
            pot_fractions = self.postflop_bets.get(street_name, [0.50, 1.0, 1.5])

            for pot_frac in pot_fractions:
                # Pot-sized raise
                raise_size = int(pot * pot_frac)
                if raise_size > 0 and (to_call + raise_size) <= stack and raise_size not in sizes:
                    sizes.append(raise_size)

        # Add all-in raise only if SPR < threshold
        if spr < self.all_in_spr_threshold:
            raise_amount = stack - to_call
            if raise_amount > 0 and raise_amount not in sizes:
                sizes.append(raise_amount)

        return sorted(sizes)

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """
        Get all legal abstracted actions for current player.

        Args:
            state: Current game state

        Returns:
            List of legal actions according to abstraction
        """
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

    def get_config_hash(self) -> str:
        """
        Compute a stable hash of the action abstraction configuration.

        This hash is used to detect config changes between training runs.
        If you load a checkpoint with a different config hash, the strategies
        may be invalid because actions were abstracted differently.

        Returns:
            16-character hex string representing the config hash
        """
        # Build deterministic representation of config
        # Sort keys to ensure consistent ordering
        config_data = {
            "big_blind": self.big_blind,
            "preflop_raises": sorted(self.preflop_raises),
            "postflop_bets": {
                street: sorted(sizes) for street, sizes in sorted(self.postflop_bets.items())
            },
            "all_in_spr_threshold": self.all_in_spr_threshold,
            "max_raises_per_street": self.max_raises_per_street,
        }

        # Use JSON for deterministic serialization
        config_json = json.dumps(config_data, sort_keys=True, separators=(",", ":"))
        return xxhash.xxh64(config_json.encode()).hexdigest()

    def __str__(self) -> str:
        """String representation of abstraction."""
        preflop_str = ", ".join([f"{r}bb" for r in self.preflop_raises])

        # Show street-dependent postflop
        postflop_parts = []
        for street, bets in self.postflop_bets.items():
            bet_str = ", ".join([f"{int(b * 100)}%" for b in bets])
            postflop_parts.append(f"{street}=[{bet_str}]")
        postflop_str = ", ".join(postflop_parts)

        return f"BettingActions(preflop=[{preflop_str}], postflop={{{postflop_str}}}, all_in_spr<{self.all_in_spr_threshold})"
