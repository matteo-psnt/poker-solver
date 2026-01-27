"""Simple range inference utilities for runtime resolving."""

from __future__ import annotations

from dataclasses import dataclass

import eval7
import numpy as np

from src.core.game.actions import Action
from src.core.game.state import Card, GameState
from src.engine.search.action_translation import translate_action_distribution
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.shared.numeric import NORMALIZE_EPS


@dataclass
class PlayerRanges:
    """Range weights for both players at the current public node."""

    p0: np.ndarray
    p1: np.ndarray


_ALL_COMBOS: list[tuple[Card, Card]] = []
_deck = eval7.Deck()
for i in range(len(_deck.cards)):
    for j in range(i + 1, len(_deck.cards)):
        _ALL_COMBOS.append((Card(_deck.cards[i]), Card(_deck.cards[j])))
_NUM_COMBOS = len(_ALL_COMBOS)

# Floor for a combo's action likelihood. Distinct from `NORMALIZE_EPS` despite the
# shared value: this one keeps a likelihood strictly positive so a Bayes update can
# never drive a combo to exactly zero, which would evict it from the range forever
# with no way back. It is a modeling choice about range support and may be retuned
# independently of the divide-by-zero guard.
_LIKELIHOOD_FLOOR = 1e-12

# Public, canonical hole-card combo enumeration. Range vectors produced by
# `infer_ranges`/`update_ranges` are indexed by this ordering; consumers that
# reason over individual combos (e.g. the LBR evaluator, the subgame CFR) must
# use it too. All derived lookup tables live HERE — don't rebuild them locally.
ALL_COMBOS: list[tuple[Card, Card]] = _ALL_COMBOS
NUM_COMBOS: int = _NUM_COMBOS

# Per-combo card bitmasks in ALL_COMBOS order, for fast dead-card filtering.
COMBO_MASKS: np.ndarray = np.array([c1.mask | c2.mask for c1, c2 in _ALL_COMBOS], dtype=np.int64)

# Per-combo (card_a, card_b) deck indices in ALL_COMBOS order.
_CARD_INDEX: dict[int, int] = {card.mask: i for i, card in enumerate(Card.get_full_deck())}
COMBO_CARDS: np.ndarray = np.array(
    [(_CARD_INDEX[c1.mask], _CARD_INDEX[c2.mask]) for c1, c2 in _ALL_COMBOS], dtype=np.int64
)

_COMBO_INDEX_BY_MASK: dict[int, int] = {int(mask): i for i, mask in enumerate(COMBO_MASKS)}


def combo_index_for(combo: tuple[Card, Card]) -> int:
    """Index of a hole-card pair in the canonical ALL_COMBOS enumeration."""
    return _COMBO_INDEX_BY_MASK[int(combo[0].mask | combo[1].mask)]


def blocked_combos(board: tuple[Card, ...]) -> np.ndarray:
    """Boolean vector over ALL_COMBOS: True where a combo shares a card with ``board``."""
    board_mask = 0
    for card in board:
        board_mask |= card.mask
    return (COMBO_MASKS & board_mask) != 0


def infer_ranges(state: GameState, blueprint) -> PlayerRanges:
    """
    Infer ranges from blueprint and action history.

    Current implementation is intentionally conservative: it returns uniform ranges
    and relies on local re-solving + blueprint leaf values. This keeps latency tight
    for the first resolver iteration and leaves room for richer Bayesian updates.
    """
    _ = blueprint
    blocked = blocked_combos(state.board)
    base = _masked_uniform(blocked)
    return PlayerRanges(p0=base.copy(), p1=base.copy())


def update_ranges(
    state: GameState,
    ranges: PlayerRanges,
    observed_action: Action,
    blueprint,
) -> PlayerRanges:
    """
    Update ranges after an observed action using Bayesian likelihood weighting.

    The resolver still adapts through re-solving each node; this keeps the opponent
    range tracking calibrated to observed actions.
    """

    actor = state.current_player
    blocked = blocked_combos(state.board)
    likelihood = _action_likelihood_vector(
        state=state,
        actor=actor,
        observed_action=observed_action,
        blueprint=blueprint,
        blocked=blocked,
    )

    p0 = _normalize_or_uniform(_mask_invalid_combos(ranges.p0, blocked), blocked)
    p1 = _normalize_or_uniform(_mask_invalid_combos(ranges.p1, blocked), blocked)

    if actor == 0:
        posterior = p0 * likelihood
        p0 = _normalize_or_uniform(posterior, blocked)
    else:
        posterior = p1 * likelihood
        p1 = _normalize_or_uniform(posterior, blocked)

    return PlayerRanges(p0=p0, p1=p1)


def _action_likelihood_vector(
    *,
    state: GameState,
    actor: int,
    observed_action: Action,
    blueprint,
    blocked: np.ndarray,
) -> np.ndarray:
    likelihood = np.full(_NUM_COMBOS, _LIKELIHOOD_FLOOR, dtype=np.float64)
    cache: dict = {}

    for idx, combo in enumerate(_ALL_COMBOS):
        if blocked[idx]:
            continue

        hypo_state = replace_actor_hole_cards(state, actor=actor, combo=combo)
        infoset_key = encode_infoset_key(hypo_state, actor, blueprint.card_abstraction)
        cached = cache.get(infoset_key)
        if cached is not None:
            likelihood[idx] = cached
            continue

        legal_actions = blueprint.rules.get_legal_actions(
            hypo_state, action_model=blueprint.action_model
        )
        if not legal_actions:
            cache[infoset_key] = _LIKELIHOOD_FLOOR
            likelihood[idx] = _LIKELIHOOD_FLOOR
            continue

        translated = translate_action_distribution(
            hypo_state,
            observed_action=observed_action,
            action_model=blueprint.action_model,
            rules=blueprint.rules,
        )
        infoset = blueprint.storage.get_infoset(infoset_key)
        if infoset is None:
            uniform = 1.0 / len(legal_actions)
            prob = sum(weight * uniform for _, weight in translated)
            prob = max(prob, _LIKELIHOOD_FLOOR)
            cache[infoset_key] = prob
            likelihood[idx] = prob
            continue

        valid_indices: list[int] = []
        valid_actions: list[Action] = []
        legal_set = set(legal_actions)
        for i, action in enumerate(infoset.legal_actions):
            if action in legal_set and blueprint.rules.is_action_valid(hypo_state, action):
                valid_indices.append(i)
                valid_actions.append(action)

        if not valid_indices:
            uniform = 1.0 / len(legal_actions)
            prob = sum(weight * uniform for _, weight in translated)
            prob = max(prob, _LIKELIHOOD_FLOOR)
            cache[infoset_key] = prob
            likelihood[idx] = prob
            continue

        strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=True)
        action_prob: dict[Action, float] = {}
        for action, prob in zip(valid_actions, strategy):
            action_prob[action] = action_prob.get(action, 0.0) + float(prob)

        total_prob = 0.0
        for mapped_action, weight in translated:
            total_prob += float(weight) * action_prob.get(mapped_action, 0.0)

        total_prob = max(total_prob, _LIKELIHOOD_FLOOR)
        cache[infoset_key] = total_prob
        likelihood[idx] = total_prob

    return likelihood


def replace_actor_hole_cards(
    state: GameState,
    *,
    actor: int,
    combo: tuple[Card, Card],
) -> GameState:
    """Copy of ``state`` with ``actor``'s hole cards swapped for ``combo``.

    Validation is skipped: the source state is valid by construction, and
    mid-transition states (e.g. a street advanced but its board not yet dealt)
    would fail the board-size check if re-validated.
    """
    hole_cards = list(state.hole_cards)
    hole_cards[actor] = combo
    return state.replace(hole_cards=(hole_cards[0], hole_cards[1]), validate=False)


def _masked_uniform(blocked: np.ndarray) -> np.ndarray:
    weights = np.where(blocked, 0.0, 1.0)
    total = weights.sum()
    if total <= NORMALIZE_EPS:
        return np.full(_NUM_COMBOS, 1.0 / _NUM_COMBOS, dtype=np.float64)
    return weights / total


def _mask_invalid_combos(weights: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    if len(weights) != _NUM_COMBOS:
        return _masked_uniform(blocked)
    masked = np.array(weights, dtype=np.float64, copy=True)
    masked[blocked] = 0.0
    return masked


def _normalize_or_uniform(weights: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= NORMALIZE_EPS:
        return _masked_uniform(blocked)
    return weights / total
