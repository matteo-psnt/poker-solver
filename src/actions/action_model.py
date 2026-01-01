"""
Node-template action model for Heads-Up No-Limit Hold'em.

This module replaces fixed street-only abstractions with situation-dependent
templates (preflop node type, postflop raise depth, and SPR-gated all-in nodes).
"""

from __future__ import annotations

import json

import xxhash

from src.game.actions import Action, ActionType, all_in, bet, call, check, fold, raises
from src.game.state import GameState
from src.utils.config import Config


class ActionModel:
    """
    Legal-action generator backed by configurable node templates.

    The model is intentionally small and deterministic so it can be reused by:
    - blueprint training
    - evaluation
    - runtime subgame resolving
    """

    def __init__(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config")
        self.config = config

    def get_preflop_open_sizes_bb(self) -> list[float]:
        """Return configured SB first-in non-all-in open sizes in BB units."""
        options = self.config.action_model.preflop_templates["sb_first_in"]
        sizes: list[float] = []
        for token in options:
            if isinstance(token, (int, float)) and float(token) > 0:
                sizes.append(float(token))
        return sorted(set(sizes))

    def get_bet_sizes(self, state: GameState) -> list[int]:
        """Get legal bet sizes for the current player when no bet is faced."""
        if state.to_call > 0:
            return []

        raises_this_street = self._count_raises_on_current_street(state)
        if raises_this_street >= self.config.resolver.max_raises_per_street:
            return []

        if state.street.is_preflop():
            return self._preflop_bet_sizes(state)

        template_key = self._postflop_template_key(state, raises_this_street)
        tokens = self.config.action_model.postflop_templates[template_key]
        return self._postflop_bet_sizes_from_tokens(state, tokens)

    def get_raise_sizes(self, state: GameState) -> list[int]:
        """Get legal raise sizes for the current player when facing a bet."""
        if state.to_call == 0:
            return []

        stack = state.stacks[state.current_player]
        if stack <= state.to_call:
            return []

        raises_this_street = self._count_raises_on_current_street(state)
        if raises_this_street >= self.config.resolver.max_raises_per_street:
            return []

        if state.street.is_preflop():
            return self._preflop_raise_sizes(state)

        template_key = self._postflop_template_key(state, raises_this_street)
        tokens = self.config.action_model.postflop_templates[template_key]
        return self._postflop_raise_sizes_from_tokens(state, tokens)

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Get all legal actions according to the current abstraction model."""
        if state.is_terminal:
            return []

        actions: list[Action] = []
        stack = state.stacks[state.current_player]

        if state.to_call > 0:
            actions.append(fold())
            if state.to_call >= stack:
                actions.append(all_in(stack))
            else:
                actions.append(call())
        else:
            actions.append(check())

        if state.to_call == 0:
            for size in self.get_bet_sizes(state):
                if size >= stack:
                    if not any(a.type == ActionType.ALL_IN for a in actions):
                        actions.append(all_in(stack))
                else:
                    actions.append(bet(size))
        else:
            for size in self.get_raise_sizes(state):
                total_needed = state.to_call + size
                if total_needed >= stack:
                    if not any(a.type == ActionType.ALL_IN for a in actions):
                        actions.append(all_in(stack))
                else:
                    actions.append(raises(size))

        return actions

    def discretize_action(self, state: GameState, action: Action) -> Action:
        """Map a raw action to the nearest legal abstract action."""
        if action.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            return action

        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            return action

        if action.type == ActionType.ALL_IN:
            all_in_actions = [a for a in legal_actions if a.type == ActionType.ALL_IN]
            if all_in_actions:
                return all_in_actions[0]
            aggressive = [a for a in legal_actions if a.is_aggressive()]
            return max(aggressive, key=lambda a: a.amount) if aggressive else action

        target_type = ActionType.BET if action.type == ActionType.BET else ActionType.RAISE
        same_type = [a for a in legal_actions if a.type == target_type]
        if same_type:
            return min(same_type, key=lambda a: abs(a.amount - action.amount))

        aggressive = [a for a in legal_actions if a.is_aggressive()]
        if aggressive:
            return min(aggressive, key=lambda a: abs(a.amount - action.amount))
        return action

    def get_config_hash(self) -> str:
        """Compute a stable hash for action model compatibility checks."""
        config_data = {
            "big_blind": self.config.game.big_blind,
            "max_raises_per_street": self.config.resolver.max_raises_per_street,
            "action_model": self.config.action_model.model_dump(),
        }
        payload = json.dumps(config_data, sort_keys=True, separators=(",", ":"))
        return xxhash.xxh64(payload.encode()).hexdigest()

    def _preflop_template_key(self, state: GameState) -> str:
        aggressive = self._count_preflop_aggressions(state)
        current_is_sb = state.current_player == state.button_position

        if aggressive == 0:
            if current_is_sb:
                return "sb_first_in"
            return "bb_vs_limp"
        if aggressive == 1:
            return "bb_vs_open" if not current_is_sb else "sb_vs_3bet"
        if aggressive == 2:
            return "sb_vs_3bet" if current_is_sb else "bb_vs_4bet"
        if aggressive == 3:
            return "sb_vs_5bet" if current_is_sb else "bb_vs_4bet"
        return "sb_vs_5bet" if current_is_sb else "bb_vs_4bet"

    def _preflop_bet_sizes(self, state: GameState) -> list[int]:
        template_key = self._preflop_template_key(state)
        tokens = self.config.action_model.preflop_templates[template_key]
        stack = state.stacks[state.current_player]
        sizes: list[int] = []

        for token in tokens:
            if isinstance(token, (int, float)):
                total_bet = int(round(float(token) * self.config.game.big_blind))
                if 0 < total_bet <= stack:
                    sizes.append(total_bet)
            elif token in ("jam", "allin", "all_in"):
                if stack > 0:
                    sizes.append(stack)

        return sorted(set(sizes))

    def _preflop_raise_sizes(self, state: GameState) -> list[int]:
        template_key = self._preflop_template_key(state)
        tokens = self.config.action_model.preflop_templates[template_key]
        stack = state.stacks[state.current_player]
        to_call = state.to_call
        sizes: list[int] = []

        for token in tokens:
            if isinstance(token, (int, float)):
                total_bet = int(round(float(token) * self.config.game.big_blind))
                raise_amount = total_bet - to_call
                if raise_amount > 0 and total_bet <= stack:
                    sizes.append(raise_amount)
            elif isinstance(token, str):
                parsed = self._parse_preflop_raise_token(token, to_call=to_call, stack=stack)
                if parsed is not None:
                    sizes.append(parsed)

        return sorted(set(s for s in sizes if s > 0))

    def _parse_preflop_raise_token(self, token: str, *, to_call: int, stack: int) -> int | None:
        normalized = token.strip().lower()
        if normalized in {"fold", "call", "check", "limp"}:
            return None
        if normalized in {"jam", "allin", "all_in"}:
            raise_amount = stack - to_call
            return raise_amount if raise_amount > 0 else None
        if normalized.endswith("x_open"):
            mult = float(normalized[: -len("x_open")])
            total_bet = int(round(mult * max(1, to_call)))
            raise_amount = total_bet - to_call
            if raise_amount > 0 and total_bet <= stack:
                return raise_amount
            return None
        if normalized.endswith("x_last"):
            mult = float(normalized[: -len("x_last")])
            total_bet = int(round(mult * max(1, to_call)))
            raise_amount = total_bet - to_call
            if raise_amount > 0 and total_bet <= stack:
                return raise_amount
            return None

        raise ValueError(f"Unsupported preflop raise token: {token}")

    def _postflop_template_key(self, state: GameState, raises_this_street: int) -> str:
        templates = self.config.action_model.postflop_templates
        if state.to_call == 0 and raises_this_street == 0:
            if state.street.name == "TURN" and "first_aggressive_turn" in templates:
                return "first_aggressive_turn"
            if state.street.name == "RIVER" and "first_aggressive_river" in templates:
                return "first_aggressive_river"
            return "first_aggressive"

        raise_rules = self.config.action_model.raise_count_rules
        if raises_this_street <= 1:
            return raise_rules["facing_1"]
        if raises_this_street == 2:
            return raise_rules["facing_2"]
        return raise_rules["facing_3_plus"]

    def _postflop_bet_sizes_from_tokens(
        self, state: GameState, tokens: list[float | str]
    ) -> list[int]:
        pot = state.pot
        stack = state.stacks[state.current_player]
        sizes: list[int] = []
        jam_spr_cutoff = self._jam_spr_cutoff()
        spr = self._spr(state)

        for token in tokens:
            if isinstance(token, (int, float)):
                bet_size = int(pot * float(token))
                if 0 < bet_size <= stack:
                    sizes.append(bet_size)
            elif isinstance(token, str):
                t = token.lower()
                if t in {"jam", "allin", "all_in"} and stack > 0:
                    sizes.append(stack)
                elif t == "jam_low_spr" and stack > 0 and spr <= jam_spr_cutoff:
                    sizes.append(stack)

        return sorted(set(s for s in sizes if s > 0))

    def _postflop_raise_sizes_from_tokens(
        self, state: GameState, tokens: list[float | str]
    ) -> list[int]:
        pot = state.pot
        to_call = state.to_call
        stack = state.stacks[state.current_player]
        jam_spr_cutoff = self._jam_spr_cutoff()
        spr = self._spr(state)
        sizes: list[int] = []

        for token in tokens:
            if isinstance(token, (int, float)):
                raise_size = int(pot * float(token))
                if raise_size > 0 and (to_call + raise_size) <= stack:
                    sizes.append(raise_size)
                continue

            t = token.lower()
            if t == "min_raise":
                raise_size = to_call
                if raise_size > 0 and (to_call + raise_size) <= stack:
                    sizes.append(raise_size)
            elif t == "pot_raise":
                raise_size = pot
                if raise_size > 0 and (to_call + raise_size) <= stack:
                    sizes.append(raise_size)
            elif t in {"jam", "allin", "all_in"}:
                jam_raise = stack - to_call
                if jam_raise > 0:
                    sizes.append(jam_raise)
            elif t == "jam_low_spr" and spr <= jam_spr_cutoff:
                jam_raise = stack - to_call
                if jam_raise > 0:
                    sizes.append(jam_raise)

        return sorted(set(s for s in sizes if s > 0))

    def _count_raises_on_current_street(self, state: GameState) -> int:
        raises_count = 0
        betting_history = list(state.betting_history)

        for i in range(len(betting_history) - 1, -1, -1):
            action = betting_history[i]

            if action.type == ActionType.CALL:
                break

            if (
                i >= 1
                and action.type == ActionType.CHECK
                and betting_history[i - 1].type == ActionType.CHECK
            ):
                break

            if action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                raises_count += 1

        return raises_count

    def _count_preflop_aggressions(self, state: GameState) -> int:
        if not state.street.is_preflop():
            return 0
        return sum(
            1
            for a in state.betting_history
            if a.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
        )

    def _spr(self, state: GameState) -> float:
        stack = state.stacks[state.current_player]
        if state.pot <= 0:
            return float("inf")
        return stack / state.pot

    def _jam_spr_cutoff(self) -> float:
        return float(self.config.action_model.jam_spr_threshold)

    def __str__(self) -> str:
        return (
            "ActionModel("
            f"preflop_templates={sorted(self.config.action_model.preflop_templates.keys())}, "
            f"postflop_templates={sorted(self.config.action_model.postflop_templates.keys())}, "
            f"jam_spr_threshold={self.config.action_model.jam_spr_threshold}, "
            f"off_tree_mapping={self.config.action_model.off_tree_mapping})"
        )
