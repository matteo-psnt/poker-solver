"""Simple range inference utilities for runtime resolving."""

from __future__ import annotations

from dataclasses import dataclass

import eval7
import numpy as np

from src.core.game.actions import Action
from src.core.game.state import Card, GameState
from src.engine.search.action_translation import translate_action_distribution
from src.engine.solver.infoset_encoder import encode_infoset_key


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
_EPS = 1e-12


def uniform_ranges(num_combos: int = _NUM_COMBOS) -> PlayerRanges:
    """Create normalized uniform ranges for both players."""
    p0 = np.ones(num_combos, dtype=np.float64)
    p1 = np.ones(num_combos, dtype=np.float64)
    p0 /= p0.sum()
    p1 /= p1.sum()
    return PlayerRanges(p0=p0, p1=p1)


def infer_ranges(state: GameState, blueprint) -> PlayerRanges:
    """
    Infer ranges from blueprint and action history.

    Current implementation is intentionally conservative: it returns uniform ranges
    and relies on local re-solving + blueprint leaf values. This keeps latency tight
    for the first resolver iteration and leaves room for richer Bayesian updates.
    """
    _ = blueprint
    board_masks = {card.mask for card in state.board}
    base = _masked_uniform(board_masks)
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
    board_masks = {card.mask for card in state.board}
    likelihood = _action_likelihood_vector(
        state=state,
        actor=actor,
        observed_action=observed_action,
        blueprint=blueprint,
        board_masks=board_masks,
    )

    p0 = _normalize_or_uniform(_mask_invalid_combos(ranges.p0, board_masks), board_masks)
    p1 = _normalize_or_uniform(_mask_invalid_combos(ranges.p1, board_masks), board_masks)

    if actor == 0:
        posterior = p0 * likelihood
        p0 = _normalize_or_uniform(posterior, board_masks)
    else:
        posterior = p1 * likelihood
        p1 = _normalize_or_uniform(posterior, board_masks)

    return PlayerRanges(p0=p0, p1=p1)


def _action_likelihood_vector(
    *,
    state: GameState,
    actor: int,
    observed_action: Action,
    blueprint,
    board_masks: set[int],
) -> np.ndarray:
    likelihood = np.full(_NUM_COMBOS, _EPS, dtype=np.float64)
    cache: dict = {}

    for idx, combo in enumerate(_ALL_COMBOS):
        if combo[0].mask in board_masks or combo[1].mask in board_masks:
            continue

        hypo_state = _replace_actor_hole_cards(state, actor=actor, combo=combo)
        infoset_key = encode_infoset_key(hypo_state, actor, blueprint.card_abstraction)
        cached = cache.get(infoset_key)
        if cached is not None:
            likelihood[idx] = cached
            continue

        legal_actions = blueprint.action_model.get_legal_actions(hypo_state)
        if not legal_actions:
            cache[infoset_key] = _EPS
            likelihood[idx] = _EPS
            continue

        translated = translate_action_distribution(
            hypo_state,
            observed_action=observed_action,
            action_model=blueprint.action_model,
        )
        infoset = blueprint.storage.get_infoset(infoset_key)
        if infoset is None:
            uniform = 1.0 / len(legal_actions)
            prob = sum(weight * uniform for _, weight in translated)
            prob = max(prob, _EPS)
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
            prob = max(prob, _EPS)
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

        total_prob = max(total_prob, _EPS)
        cache[infoset_key] = total_prob
        likelihood[idx] = total_prob

    return likelihood


def _replace_actor_hole_cards(
    state: GameState,
    *,
    actor: int,
    combo: tuple[Card, Card],
) -> GameState:
    hole_cards = list(state.hole_cards)
    hole_cards[actor] = combo
    return GameState(
        street=state.street,
        pot=state.pot,
        stacks=state.stacks,
        board=state.board,
        hole_cards=(hole_cards[0], hole_cards[1]),
        betting_history=state.betting_history,
        button_position=state.button_position,
        current_player=state.current_player,
        is_terminal=state.is_terminal,
        to_call=state.to_call,
        last_aggressor=state.last_aggressor,
    )


def _masked_uniform(board_masks: set[int]) -> np.ndarray:
    weights = np.zeros(_NUM_COMBOS, dtype=np.float64)
    for idx, combo in enumerate(_ALL_COMBOS):
        if combo[0].mask in board_masks or combo[1].mask in board_masks:
            continue
        weights[idx] = 1.0
    total = weights.sum()
    if total <= _EPS:
        return np.full(_NUM_COMBOS, 1.0 / _NUM_COMBOS, dtype=np.float64)
    return weights / total


def _mask_invalid_combos(weights: np.ndarray, board_masks: set[int]) -> np.ndarray:
    if len(weights) != _NUM_COMBOS:
        return _masked_uniform(board_masks)
    masked = np.array(weights, dtype=np.float64, copy=True)
    for idx, combo in enumerate(_ALL_COMBOS):
        if combo[0].mask in board_masks or combo[1].mask in board_masks:
            masked[idx] = 0.0
    return masked


def _normalize_or_uniform(weights: np.ndarray, board_masks: set[int]) -> np.ndarray:
    total = float(weights.sum())
    if total <= _EPS:
        return _masked_uniform(board_masks)
    return weights / total
