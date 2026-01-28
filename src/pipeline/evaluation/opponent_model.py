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
    """Per-combo action distributions for the strategy under measurement.

    ``wants_translated_state`` selects which state the model's decisions are
    defined on: ``True`` routes the LBR engine's on-tree shadow state to
    ``action_matrix`` (translation-completed strategies, whose lookups must
    never see an off-tree history), ``False`` routes the real state (the
    deployed resolver re-solves reality directly).
    """

    wants_translated_state: bool

    def reset(self, initial_state: GameState, actor: int) -> None:
        """Start of a hand; ``actor`` is the seat this model plays."""
        ...

    def action_matrix(
        self, state: GameState, actor: int
    ) -> tuple[list[Action], dict[Action, np.ndarray]]:
        """Legal actions and, per action, P(action | combo) over ``ALL_COMBOS``."""
        ...

    def observe(self, state: GameState, action: Action) -> None:
        """A (public) action was played at ``state`` by ``state.current_player``."""
        ...


class BlueprintOpponent:
    """The frozen strategy table, queried only on on-tree states.

    Stateless: distributions are pure functions of the public state. The LBR
    engine guarantees every query stays inside the abstract tree by keying off
    its carried shadow state (see :mod:`shadow_state`), so a uniform fallback
    here means a genuinely untrained infoset — never an off-tree betting
    sequence. ``queries``/``uniform_fallbacks`` count unique infoset keys per
    node (the per-key cache dedupes identically in every eval arm, so rates
    are comparable across runs).
    """

    wants_translated_state = True

    def __init__(self, blueprint, dist_memo=None):
        self.blueprint = blueprint
        self.rules = blueprint.rules
        self.action_model = blueprint.action_model
        self.card_abstraction = blueprint.card_abstraction
        self.storage = blueprint.storage
        self.queries: int = 0
        self.uniform_fallbacks: int = 0
        # Optional cross-call BlueprintDistMemo (lookahead scorer): pure cache,
        # value-inert. None (the default) keeps this class's behavior and cost
        # exactly as before — the myopic path never constructs one.
        self.dist_memo = dist_memo

    def reset(self, initial_state: GameState, actor: int) -> None:
        del initial_state, actor

    def observe(self, state: GameState, action: Action) -> None:
        del state, action

    def action_matrix(
        self, state: GameState, actor: int
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
                dist = self._memoized_distribution(key, opp_state, actor)
                cache[key] = dist
            for action, prob in dist.items():
                vec = vecs.get(action)
                if vec is not None:
                    vec[idx] += prob
        return legal, vecs

    def _memoized_distribution(
        self, infoset_key: object, state: GameState, opp: int
    ) -> dict[Action, float]:
        """Cross-call memo wrapper around :meth:`_opponent_distribution`.

        The memo key extends the infoset key with the exact public chip
        configuration: two states can share an infoset key (same normalized
        sequence / bucket / SPR bucket) with different chips, which changes the
        legal menu and hence the distribution. Counters tick per logical query
        on hits too, so fallback-rate diagnostics are memo-invariant.
        """
        if self.dist_memo is None:
            return self._opponent_distribution(state, opp)
        memo_key = self.dist_memo.key(infoset_key, state)
        entry = self.dist_memo.get(memo_key)
        if entry is not None:
            dist, was_fallback = entry
            self.queries += 1
            if was_fallback:
                self.uniform_fallbacks += 1
            return dist
        fallbacks_before = self.uniform_fallbacks
        dist = self._opponent_distribution(state, opp)
        if dist:  # the empty-legal {} case never ticks counters; keep it un-memoized
            self.dist_memo.put(memo_key, dist, self.uniform_fallbacks > fallbacks_before)
        return dist

    def _opponent_distribution(self, state: GameState, opp: int) -> dict[Action, float]:
        """Blueprint response distribution over legal actions at ``state``.

        A direct infoset read; missing infosets fall back to uniform. The
        engine's shadow keeps every query on-tree, so a miss can only mean the
        infoset was never trained.
        """
        legal = list(self.rules.get_legal_actions(state, action_model=self.action_model))
        if not legal:
            return {}

        self.queries += 1
        direct = self._infoset_distribution(state, opp, legal)
        if direct is not None:
            return direct

        self.uniform_fallbacks += 1
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


class ResolvedOpponent:
    """The deployed system: blueprint + runtime subgame resolver.

    Faithful to deployment semantics (:class:`HUResolver` as driven by
    ``resolver_match``): fresh ranges each hand, then **history-replay range
    inference** — every realized action from BOTH seats Bayes-updates the
    acting player's range using blueprint likelihoods (translation-completed
    for off-tree sizes), so the resolver's next solve sees ranges shaped by
    the whole observed history instead of the uniform ranges that made the
    deployed system measurably exploitable.

    Determinism: requires ``resolver_config.max_iterations`` — budget-driven
    solves vary with wall clock, which would break reproducibility and paired
    (CRN) comparisons across checkpoints.

    Failures propagate. Deployment's ``act()`` falls back to the blueprint on
    resolver errors; an eval that silently measured the blueprint at some nodes
    while reporting a deployed number would be lying, so here a resolver error
    kills the eval loudly instead.

    Deliberately keeps the REAL state (``wants_translated_state = False``): the
    resolver's off-tree response IS the re-solve of the actual state, and its
    internal blueprint blend degrading toward uniform on off-tree histories is
    exactly what deployment does — measuring that is the point.
    """

    wants_translated_state = False

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
        # Ranges start fresh per hand and must exist before the first observe()
        # (which can precede the first action_matrix call).
        self._ranges = infer_ranges(initial_state, self.blueprint)
        self._seat = actor

    def action_matrix(
        self, state: GameState, actor: int
    ) -> tuple[list[Action], dict[Action, np.ndarray]]:
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
        """History-replay range inference: every realized action (both seats)
        Bayes-updates the acting player's range, mirroring deployment's
        ``HUResolver.observe`` bookkeeping."""
        if self._ranges is None:
            return
        self._ranges = update_ranges(state, self._ranges, action, self.blueprint)
