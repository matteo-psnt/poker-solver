"""The single definition of "the blueprint's strategy at this state".

Historically the lookup pattern — encode the infoset key, fetch the stored
infoset, restrict its stored actions to the ones playable right now, normalize
the strategy over the survivors — was re-implemented at every consumer
(training traversal, blueprint sampling, the subgame resolver, range
inference, the LBR opponent model, the rollout diagnostic), and each copy made
its own choice of which restriction to apply. Those choices diverge exactly
when the stored action set drifts from today's menu (abstraction change,
chip-configuration edge cases), which is when the answer matters most.

This module fixes the predicate once: a stored action contributes iff it is
one of the caller's ``candidates`` (membership in the currently offered menu)
AND ``rules.is_action_valid(state, action)`` holds (the current chip
configuration can actually support it). Duplicate surviving stored actions
(e.g. placeholder rows reconstructed from a checkpoint without a legal-actions
cache) aggregate by summation rather than silently dropping mass.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import GameState
from src.engine.solver.infoset import InfoSet


def filter_stored_actions(
    infoset: InfoSet,
    state: GameState,
    rules: GameRules,
    candidates: Iterable[Action],
) -> tuple[list[int], list[Action]]:
    """Indices (into ``infoset.legal_actions``) and actions that survive the filter.

    Both lists may be empty; the caller owns its fallback. Use this form when
    the original storage indices are needed (e.g. regret updates); use
    :func:`blueprint_action_distribution` when only the distribution is.
    """
    candidate_set = set(candidates)
    valid_indices: list[int] = []
    valid_actions: list[Action] = []
    for index, action in enumerate(infoset.legal_actions):
        if action in candidate_set and rules.is_action_valid(state, action):
            valid_indices.append(index)
            valid_actions.append(action)
    return valid_indices, valid_actions


def blueprint_action_distribution(
    infoset: InfoSet | None,
    state: GameState,
    rules: GameRules,
    candidates: Iterable[Action],
    *,
    use_average: bool,
) -> dict[Action, float] | None:
    """Blueprint distribution over ``candidates`` at ``state``, or ``None``.

    ``None`` means the blueprint has no usable answer here — the infoset is
    missing (untrained/off-tree) or no stored action survives the filter — and
    the caller decides the fallback (uniform, resample, refuse). The returned
    probabilities are normalized over the surviving stored actions.
    """
    if infoset is None:
        return None
    valid_indices, valid_actions = filter_stored_actions(infoset, state, rules, candidates)
    if not valid_indices:
        return None
    strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=use_average)
    distribution: dict[Action, float] = {}
    for action, probability in zip(valid_actions, strategy):
        distribution[action] = distribution.get(action, 0.0) + float(probability)
    return distribution
