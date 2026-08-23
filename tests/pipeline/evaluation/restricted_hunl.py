"""Scalar reference game for validating the public-tree BR engine.

Implements, through the generic :class:`ExtensiveGame` protocol, exactly the
restricted game that :mod:`src.pipeline.evaluation.estimators.public_tree_br` evaluates:
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
information set). ``hero_combos`` with ``hero_public_key`` deals the hero from
a set and hides its cards from its own key: a card-blind responder, which is
what the bucket-constrained engine becomes when every hero combo shares one
bucket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.engine.solver.policy.lookup import blueprint_action_distribution
from src.pipeline.evaluation.reference.game_tree import CHANCE, Policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.game.state import Card

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
        hero_combo: tuple[Card, Card] | None = None,
        opp_combos: Sequence[tuple[Card, Card]],
        button: int,
        starting_stack: int,
        full_state_keys: bool = False,
        hero_combos: Sequence[tuple[Card, Card]] | None = None,
        hero_public_key: bool = False,
    ):
        self._rules = blueprint.rules
        self._action_model = blueprint.action_model
        self._plan = plan
        self.hero_seat = hero_seat
        if hero_combos is None:
            assert hero_combo is not None, "one hero combo or a set of them"
            hero_combos = [hero_combo]
        self._hero_combos: list[tuple[Card, Card]] = list(hero_combos)
        self.opp_combos = [
            combo
            for combo in opp_combos
            if all(not (_mask(combo) & _mask(hero)) for hero in self._hero_combos)
        ]
        self._button = button
        self._starting_stack = starting_stack
        self._full_state_keys = full_state_keys
        self._hero_public_key = hero_public_key

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
            pairs = [(hero, opp) for hero in self._hero_combos for opp in self.opp_combos]
            return [(pair, 1.0 / len(pairs)) for pair in pairs]
        return self._plan.deal_options(state.board)

    def legal_actions(self, state):
        return self._rules.get_legal_actions(state, self._action_model)

    def next_state(self, state, action):
        if state == DEAL:
            hero, opp = action
            holes: list = [None, None]
            holes[self.hero_seat] = hero
            holes[1 - self.hero_seat] = opp
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
        public = (
            player,
            state.street,
            state.normalized_betting_sequence(),
            state.board,
            state.pot,
            state.stacks,
            state.to_call,
        )
        # The hero's own cards are in the full state, so a full-state key is
        # per-combo; the public projection below is card-blind.
        return public if self._hero_public_key else (*public, state.hole_cards[player])


def _mask(combo: tuple[Card, Card]) -> int:
    return combo[0].mask | combo[1].mask


def blueprint_policy(blueprint) -> Policy:
    """Policy callable replaying the deployed blueprint from full-state info keys.

    Resolves through the policy source rather than addressing storage directly:
    the source is what the real scorers use, so a test that reached past it
    would be validating a lookup path nothing in production takes.
    """
    source = blueprint.policy_source

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
