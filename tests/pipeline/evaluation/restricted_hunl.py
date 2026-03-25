"""Scalar reference game for validating the public-tree BR engine.

Implements, through the generic :class:`ExtensiveGame` protocol, exactly the
restricted game that :mod:`src.pipeline.evaluation.public_tree_br` evaluates:
the blueprint's betting tree, a fixed sampled board plan with public branch
weights, and annulment of deals incompatible with a sampled branch. The hero
holds one fixed combo and the opponent is dealt uniformly from a restricted
combo set, which keeps the scalar tree small enough for the exact
``best_response_value`` in :mod:`best_response` to serve as ground truth.

Chance actions are dealt-card tuples (the opponent combo at the root, board
extensions elsewhere); decision actions are engine :class:`Action` objects.
Information keys are ``(player, full GameState)`` for policy players — the
policy callable rebuilds the canonical infoset key from the state — and a
public-plus-own-cards projection for the hero, which makes the scalar best
responder exactly as informed as the vectorized one (each combo is its own
information set).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.game.state import Card
from src.engine.solver.policy_lookup import blueprint_action_distribution
from src.engine.solver.policy_source import policy_source_for
from src.pipeline.evaluation.game_tree import CHANCE, Policy

DEAL = "DEAL"
ANNULLED = "ANNULLED"


class RestrictedHUNL:
    """ExtensiveGame over the sampled-board annulled measure, single hero combo."""

    num_players = 2

    def __init__(
        self,
        blueprint,
        plan,
        *,
        hero_seat: int,
        hero_combo: tuple[Card, Card],
        opp_combos: Sequence[tuple[Card, Card]],
        button: int,
        starting_stack: int,
        full_state_keys: bool = False,
    ):
        self._rules = blueprint.rules
        self._action_model = blueprint.action_model
        self._plan = plan
        self.hero_seat = hero_seat
        self._hero_combo = hero_combo
        hero_mask = hero_combo[0].mask | hero_combo[1].mask
        self.opp_combos = [
            combo for combo in opp_combos if not ((combo[0].mask | combo[1].mask) & hero_mask)
        ]
        self._button = button
        self._starting_stack = starting_stack
        self._full_state_keys = full_state_keys

    def initial_state(self):
        return DEAL

    def is_terminal(self, state) -> bool:
        if state == ANNULLED:
            return True
        if state == DEAL:
            return False
        return state.is_terminal and (state.ended_by_fold or len(state.board) == 5)

    def returns(self, state):
        if state == ANNULLED:
            return (0.0, 0.0)
        return (self._rules.get_payoff(state, 0), self._rules.get_payoff(state, 1))

    def current_player(self, state) -> int:
        if state == DEAL or state.is_terminal:
            return CHANCE
        if len(state.board) < state.street.board_card_count:
            return CHANCE
        return state.current_player

    def chance_outcomes(self, state):
        if state == DEAL:
            probability = 1.0 / len(self.opp_combos)
            return [(combo, probability) for combo in self.opp_combos]
        return self._plan.deal_options(state.board)

    def legal_actions(self, state):
        return self._rules.get_legal_actions(state, self._action_model)

    def next_state(self, state, action):
        if state == DEAL:
            holes: list = [None, None]
            holes[self.hero_seat] = self._hero_combo
            holes[1 - self.hero_seat] = action
            return self._rules.create_initial_state(
                starting_stack=self._starting_stack,
                hole_cards=(holes[0], holes[1]),
                button=self._button,
            )
        if self.current_player(state) == CHANCE:
            dealt_mask = 0
            for card in action:
                dealt_mask |= card.mask
            for combo in state.hole_cards:
                if (combo[0].mask | combo[1].mask) & dealt_mask:
                    return ANNULLED
            if state.is_terminal:
                return state.replace(board=(*state.board, *action), validate=False)
            return state.replace(
                board=(*state.board, *action),
                current_player=1 - state.button_position,
                is_terminal=False,
                to_call=0,
                last_aggressor=None,
            )
        return self._rules.apply_action(state, action)

    def information_state_key(self, state, player: int):
        if self._full_state_keys or player != self.hero_seat:
            return (player, state)
        return (
            player,
            state.street,
            state.normalized_betting_sequence(),
            state.board,
            state.pot,
            state.stacks,
            state.to_call,
        )


def blueprint_policy(blueprint) -> Policy:
    """Policy callable replaying the deployed blueprint from full-state info keys.

    Resolves through the policy source rather than addressing storage directly:
    the source is what the real scorers use, so a test that reached past it
    would be validating a lookup path nothing in production takes.
    """
    source = policy_source_for(blueprint)

    def policy(info_key, legal_actions):
        player, state = info_key
        distribution = blueprint_action_distribution(
            source.infoset_for(state, player),
            state,
            blueprint.rules,
            tuple(legal_actions),
            use_average=True,
        )
        n = len(legal_actions)
        if distribution is None:
            return [1.0 / n] * n
        return [distribution.get(action, 0.0) for action in legal_actions]

    return policy
