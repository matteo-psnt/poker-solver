"""Opponent models for the HUNL LBR evaluator.

The LBR engine measures a *strategy*: at every opponent decision it needs, for
each hole-card combo the opponent might hold, the probability of each legal
action — a per-combo action matrix. This module isolates WHO that opponent is
behind a small interface so the evaluator can measure either artifact:

- :class:`BlueprintOpponent` — the raw strategy table (what LBR measured
  historically). Stateless.
- :class:`ResolvedOpponent` — the deployed system: blueprint blended with the
  runtime subgame resolver, mirroring deployment's per-hand range bookkeeping.

Only the *realized path* (the actions the opponent actually takes, and terminal
branch integration) goes through the selected model. The LBR player's myopic
scorer keeps using the blueprint model regardless: the scorer only selects the
exploiter's actions, so approximations there loosen the reported lower bound
but never invalidate it — and backing it with resolver solves would multiply
eval cost roughly tenfold for no change in the bound's validity.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import GameState
from src.engine.search.action_translation import translate_action_distribution
from src.engine.search.range_inference import (
    ALL_COMBOS,
    COMBO_MASKS,
    NUM_COMBOS,
    PlayerRanges,
    infer_ranges,
    replace_actor_hole_cards,
    update_ranges,
)
from src.engine.search.resolver import HUResolver
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.shared.config import ResolverConfig

_EPS = 1e-12


def known_mask(state: GameState, actor: int) -> int:
    """Bitmask of cards the evaluator can see (the non-actor's holes + board)."""
    viewer = 1 - actor
    mask = 0
    for card in state.hole_cards[viewer]:
        mask |= card.mask
    for card in state.board:
        mask |= card.mask
    return mask


class OpponentModel(Protocol):
    """Per-combo action distributions for the strategy under measurement."""

    def reset(self, initial_state: GameState, actor: int) -> None:
        """Start of a hand; ``actor`` is the seat this model plays."""
        ...

    def action_matrix(
        self,
        state: GameState,
        actor: int,
        prev_state: GameState | None,
        prev_action: Action | None,
    ) -> tuple[list[Action], dict[Action, np.ndarray]]:
        """Legal actions and, per action, P(action | combo) over ``ALL_COMBOS``."""
        ...

    def observe(self, state: GameState, action: Action) -> None:
        """A (public) action was played at ``state`` by ``state.current_player``."""
        ...


class BlueprintOpponent:
    """The frozen strategy table, completed off-tree by action translation.

    Stateless: distributions are pure functions of the public state. Logic moved
    verbatim from the LBR engine; it must produce bit-identical results to the
    pre-seam evaluator.
    """

    def __init__(self, blueprint):
        self.blueprint = blueprint
        self.rules = blueprint.rules
        self.action_model = blueprint.action_model
        self.card_abstraction = blueprint.card_abstraction
        self.storage = blueprint.storage

    def reset(self, initial_state: GameState, actor: int) -> None:
        del initial_state, actor

    def observe(self, state: GameState, action: Action) -> None:
        del state, action

    def action_matrix(
        self,
        state: GameState,
        actor: int,
        prev_state: GameState | None,
        prev_action: Action | None,
    ) -> tuple[list[Action], dict[Action, np.ndarray]]:
        """Per-combo blueprint action probabilities at an opponent node.

        Returns the legal actions and, for each, a length-``NUM_COMBOS`` vector
        giving the probability the opponent takes it holding each combo. The
        opponent's bucket (hence its whole distribution) is action-independent
        per combo, so it is computed once and cached by infoset key across all
        combos that share a bucket.
        """
        legal = list(self.rules.get_legal_actions(state, action_model=self.action_model))
        vecs: dict[Action, np.ndarray] = {action: np.zeros(NUM_COMBOS) for action in legal}
        if not legal:
            return legal, vecs

        known = known_mask(state, actor)
        cache: dict[object, dict[Action, float]] = {}
        for idx in range(NUM_COMBOS):
            if COMBO_MASKS[idx] & known:
                continue
            opp_state = replace_actor_hole_cards(state, actor=actor, combo=ALL_COMBOS[idx])
            key = encode_infoset_key(opp_state, actor, self.card_abstraction)
            dist = cache.get(key)
            if dist is None:
                dist = self._opponent_distribution(opp_state, actor, prev_state, prev_action)
                cache[key] = dist
            for action, prob in dist.items():
                vec = vecs.get(action)
                if vec is not None:
                    vec[idx] += prob
        return legal, vecs

    def _opponent_distribution(
        self,
        state: GameState,
        opp: int,
        prev_state: GameState | None,
        prev_lbr_action: Action | None,
    ) -> dict[Action, float]:
        """Blueprint response distribution over legal actions at ``state``.

        On-tree nodes read the blueprint infoset directly. Off-tree nodes (the
        LBR player made an off-tree bet) map the bet to the nearest on-tree size
        and read the blueprint there, projecting the response back onto the real
        legal actions — the translation completion described in the LBR module docs.
        """
        legal = list(self.rules.get_legal_actions(state, action_model=self.action_model))
        if not legal:
            return {}

        direct = self._infoset_distribution(state, opp, legal)
        if direct is not None:
            return direct

        if prev_state is not None and prev_lbr_action is not None:
            translated = self._translated_distribution(
                state, opp, legal, prev_state, prev_lbr_action
            )
            if translated:
                return translated

        uniform = 1.0 / len(legal)
        return {action: uniform for action in legal}

    def _infoset_distribution(
        self, state: GameState, player: int, legal: list[Action]
    ) -> dict[Action, float] | None:
        """Blueprint strategy over ``legal`` at ``state``, or ``None`` if off-tree."""
        infoset_key = encode_infoset_key(state, player, self.card_abstraction)
        infoset = self.storage.get_infoset(infoset_key)
        if infoset is None:
            return None

        legal_set = set(legal)
        valid_indices: list[int] = []
        valid_actions: list[Action] = []
        for idx, action in enumerate(infoset.legal_actions):
            if action in legal_set and self.rules.is_action_valid(state, action):
                valid_indices.append(idx)
                valid_actions.append(action)
        if not valid_indices:
            return None

        strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=True)
        dist: dict[Action, float] = {}
        for action, prob in zip(valid_actions, strategy):
            dist[action] = dist.get(action, 0.0) + float(prob)
        return dist

    def _translated_distribution(
        self,
        state: GameState,
        opp: int,
        legal: list[Action],
        prev_state: GameState,
        prev_lbr_action: Action,
    ) -> dict[Action, float]:
        """Project the blueprint's on-tree response onto real legal actions."""
        on_tree = translate_action_distribution(
            prev_state, prev_lbr_action, self.action_model, self.rules
        )
        dist: dict[Action, float] = {}
        for proxy_action, weight in on_tree:
            proxy_state = prev_state.apply_action(proxy_action, self.rules)
            proxy_legal = list(
                self.rules.get_legal_actions(proxy_state, action_model=self.action_model)
            )
            proxy_dist = self._infoset_distribution(proxy_state, opp, proxy_legal)
            if proxy_dist is None:
                continue
            for action, prob in proxy_dist.items():
                for real_action, share in self._map_to_real(state, action, legal):
                    dist[real_action] = dist.get(real_action, 0.0) + weight * prob * share

        total = sum(dist.values())
        if total <= _EPS:
            return {}
        return {action: prob / total for action, prob in dist.items()}

    def _map_to_real(
        self, state: GameState, action: Action, legal: list[Action]
    ) -> list[tuple[Action, float]]:
        """Map a proxy-node response to the real legal actions at ``state``."""
        if action in legal:
            return [(action, 1.0)]
        # Fold/check/call carry across states unchanged; only bet/raise sizes
        # need re-discretizing to the real legal menu.
        return translate_action_distribution(state, action, self.action_model, self.rules)


class ResolvedOpponent:
    """The deployed system: blueprint + runtime subgame resolver.

    Faithful to deployment semantics (:class:`HUResolver` as driven by
    ``resolver_match``): fresh ranges each hand; the model's own range is
    Bayes-updated after each of its own actions using blueprint likelihoods;
    the other player's observed actions are NOT incorporated (the known
    uniform-opponent-range limitation of the deployed resolver — measuring it
    is the point).

    Determinism: requires ``resolver_config.max_iterations`` — budget-driven
    solves vary with wall clock, which would break reproducibility and paired
    (CRN) comparisons across checkpoints.

    Failures propagate. Deployment's ``act()`` falls back to the blueprint on
    resolver errors; an eval that silently measured the blueprint at some nodes
    while reporting a deployed number would be lying, so here a resolver error
    kills the eval loudly instead.
    """

    def __init__(self, blueprint, resolver_config: ResolverConfig):
        if resolver_config.max_iterations is None:
            raise ValueError(
                "Deployed-opponent LBR requires resolver max_iterations to be set: "
                "wall-clock-budgeted solves make the measured strategy load-dependent "
                "and irreproducible."
            )
        self.blueprint = blueprint
        self._resolver = HUResolver(
            blueprint=blueprint,
            action_model=blueprint.action_model,
            rules=blueprint.rules,
            config=resolver_config,
        )
        self._ranges: PlayerRanges | None = None
        self._seat: int | None = None
        # Realized-path subgame solves this model ran (cost/diagnostics).
        self.solve_count: int = 0

    def reset(self, initial_state: GameState, actor: int) -> None:
        del initial_state
        self._ranges = None
        self._seat = actor

    def action_matrix(
        self,
        state: GameState,
        actor: int,
        prev_state: GameState | None,
        prev_action: Action | None,
    ) -> tuple[list[Action], dict[Action, np.ndarray]]:
        # The resolver's off-tree response IS the re-solve of the actual state:
        # no translation proxy is needed, so prev_state/prev_action are unused.
        del prev_state, prev_action
        if self._ranges is None:
            self._ranges = infer_ranges(state, self.blueprint)

        actions, matrix = self._resolver.solve_strategy_matrix(state, ranges=self._ranges)
        self.solve_count += 1
        # Duplicate legal actions would silently drop matrix columns here; the
        # action model never produces them, so guard rather than handle.
        assert len(set(actions)) == len(actions), "duplicate root actions"
        del actor
        # The matrix carries uniform placeholder rows for board-blocked combos;
        # as an opponent model those rows must be ZERO (P(action | dead combo)),
        # or a mid-hand street deal leaves dead combos alive in the caller's
        # belief posterior. Board cards are public, so this masking is honest.
        board_mask = 0
        for card in state.board:
            board_mask |= card.mask
        matrix = np.where((COMBO_MASKS & board_mask)[:, None] == 0, matrix, 0.0)
        return actions, {action: matrix[:, i] for i, action in enumerate(actions)}

    def observe(self, state: GameState, action: Action) -> None:
        """Mirror deployment's range bookkeeping: self-update on own actions only."""
        if state.current_player != self._seat or self._ranges is None:
            return
        self._ranges = update_ranges(state, self._ranges, action, self.blueprint)
