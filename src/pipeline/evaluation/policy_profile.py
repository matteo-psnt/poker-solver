"""What a checkpoint's strategy looks like, read straight off the arrays.

Not an exploitability: a sanity read beside one. Per street, how much of the
table training touched and how mixed the rows are -- under the average
strategy (what is fielded) AND the regret-matched current one (what training
is still moving toward) -- plus the preflop nodes as 169-class tables, which
is where a pathology is legible to a person.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.game.actions import ActionType
from src.core.game.state import FULL_DECK, GameState, Street
from src.engine.solver.infoset.index import preflop_hand_string_at

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.actions import Action
    from src.core.game.rules import GameRules
    from src.engine.solver.storage.static_array import StaticArrayStorage

PURE_CUTOFF = 0.99
"""A row whose top action carries at least this much is counted as pure."""

NAMED_HANDS = ("AA", "KK", "QQ", "AKs", "AKo", "JTs", "T9s", "55", "22", "A2o", "K2o", "72o")
_QUANTILES = (0.1, 0.5, 0.9)
_DUMMY_HOLES = ((FULL_DECK[0], FULL_DECK[1]), (FULL_DECK[2], FULL_DECK[3]))


def profile_policy(
    storage: StaticArrayStorage,
    rules: GameRules,
    action_model: ActionModel,
    starting_stack: int,
) -> dict[str, Any]:
    """Per-street coverage and entropy, plus the preflop nodes class by class."""
    return {
        "streets": _street_profiles(storage),
        "preflop": _preflop_nodes(storage, rules, action_model, starting_stack),
        "preflop_open_sizes_bb": list(action_model.get_preflop_open_sizes_bb()),
    }


def _street_profiles(storage: StaticArrayStorage) -> dict[str, dict[str, Any]]:
    tree = storage.tree
    rows_total: dict[Street, int] = dict.fromkeys(Street, 0)
    average_entropy: dict[Street, list[np.ndarray]] = {street: [] for street in Street}
    current_entropy: dict[Street, list[np.ndarray]] = {street: [] for street in Street}
    top_mass: dict[Street, list[np.ndarray]] = {street: [] for street in Street}
    no_positive_regret: dict[Street, int] = dict.fromkeys(Street, 0)
    for node in tree.nodes:
        width = node.num_actions
        count = int(tree.buckets_per_node[node.node_id])
        first_row = int(tree.row_offset[node.node_id])
        start = int(tree.slot_offset[node.node_id])
        visited = storage.visited[first_row : first_row + count].astype(bool)
        rows_total[node.street] += count
        if not visited.any():
            continue
        sums = storage.strategy_sum[start : start + count * width].reshape(count, width)
        regrets = storage.regrets[start : start + count * width].reshape(count, width)
        average = _normalised(sums[visited].astype(np.float64))
        positive = np.maximum(regrets[visited].astype(np.float64), 0.0)
        no_positive_regret[node.street] += int((positive.sum(axis=1) <= 0.0).sum())
        current = _normalised(positive)
        average_entropy[node.street].append(_entropy(average))
        current_entropy[node.street].append(_entropy(current))
        top_mass[node.street].append(average.max(axis=1))
    out: dict[str, dict[str, Any]] = {}
    for street in Street:
        visited_rows = sum(len(chunk) for chunk in top_mass[street])
        profile: dict[str, Any] = {
            "rows": rows_total[street],
            "visited_rows": visited_rows,
            "visited_fraction": visited_rows / rows_total[street] if rows_total[street] else 0.0,
        }
        if visited_rows:
            average = np.concatenate(average_entropy[street])
            current = np.concatenate(current_entropy[street])
            top = np.concatenate(top_mass[street])
            profile["average_entropy"] = _quantiles(average)
            profile["current_entropy"] = _quantiles(current)
            profile["pure_fraction"] = float((top >= PURE_CUTOFF).mean())
            profile["no_positive_regret_fraction"] = no_positive_regret[street] / visited_rows
        out[str(street)] = profile
    return out


def _normalised(rows: np.ndarray) -> np.ndarray:
    """Rows to distributions; a zero row is uniform, as ``average_strategy`` has it."""
    totals = rows.sum(axis=1, keepdims=True)
    uniform = np.full_like(rows, 1.0 / rows.shape[1])
    return np.divide(rows, totals, out=uniform, where=totals > 0)


def _entropy(rows: np.ndarray) -> np.ndarray:
    """Shannon entropy per row, normalised to [0, 1] by log(actions)."""
    width = rows.shape[1]
    if width < 2:
        return np.zeros(rows.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(rows > 0, rows * np.log(rows), 0.0)
    return -terms.sum(axis=1) / np.log(width)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, _QUANTILES)
    out = {f"p{int(q * 100)}": float(v) for q, v in zip(_QUANTILES, points, strict=True)}
    out["mean"] = float(values.mean())
    return out


def _preflop_nodes(
    storage: StaticArrayStorage,
    rules: GameRules,
    action_model: ActionModel,
    starting_stack: int,
) -> list[dict[str, Any]]:
    """The first decisions of a hand, as 169-class tables.

    Walks the real rules from the root so the node labels (open sizes in bb)
    come off the action model rather than a guess: SB first in, BB facing a
    limp and each open, the SB facing each raise of its limp, and the SB facing
    the smallest 3-bet.
    """
    root = rules.create_initial_state(
        starting_stack=starting_stack, hole_cards=_DUMMY_HOLES, button=0
    )
    nodes = [("SB first in", root)]
    for action in rules.get_legal_actions(root, action_model):
        child = rules.apply_action(root, action)
        if child.is_terminal or child.street != Street.PREFLOP:
            continue
        if action.type == ActionType.CALL:
            nodes.append(("BB vs limp", child))
            for reply in rules.get_legal_actions(child, action_model):
                raised = rules.apply_action(child, reply)
                if reply.type in (ActionType.BET, ActionType.RAISE) and not raised.is_terminal:
                    nodes.append((f"SB limp vs raise {_label(reply, child, rules)}", raised))
        elif action.type == ActionType.RAISE:
            nodes.append((f"BB vs open {_label(action, root, rules)}", child))
    opens = [(label, state) for label, state in nodes if label.startswith("BB vs open")]
    if opens:
        label, opened = opens[0]
        for action in rules.get_legal_actions(opened, action_model):
            if action.type == ActionType.RAISE:
                nodes.append(
                    (
                        f"SB {label[6:]} vs 3-bet {_label(action, opened, rules)}",
                        rules.apply_action(opened, action),
                    )
                )
                break
    return [_node_table(storage, rules, action_model, label, state) for label, state in nodes]


def _label(action: Action, state: GameState, rules: GameRules) -> str:
    """Human label for an action at ``state``: chips committed by it, in bb."""
    if action.type in (ActionType.RAISE, ActionType.BET):
        return f"{(action.amount + state.to_call) / rules.big_blind:g}bb"
    return action.normalize(state.pot)


def _node_table(
    storage: StaticArrayStorage,
    rules: GameRules,
    action_model: ActionModel,
    label: str,
    state: GameState,
) -> dict[str, Any]:
    tree = storage.tree
    node_id = tree.node_id(state)
    legal = rules.get_legal_actions(state, action_model)
    tokens = [_label(action, state, rules) for action in legal]
    width = int(tree.num_actions[node_id])
    count = int(tree.buckets_per_node[node_id])
    start = int(tree.slot_offset[node_id])
    first_row = int(tree.row_offset[node_id])
    rows = _normalised(
        storage.strategy_sum[start : start + count * width].reshape(count, width).astype(np.float64)
    )
    visited = storage.visited[first_row : first_row + count].astype(bool)
    classes = [preflop_hand_string_at(index) for index in range(count)]
    weights = np.array([_combos_in_class(name) for name in classes], dtype=np.float64)
    return {
        "label": label,
        "sequence": state.normalized_betting_sequence(),
        "actions": tokens,
        "visited_classes": int(visited.sum()),
        "mean_mix": dict(zip(tokens, rows.mean(axis=0).tolist(), strict=True)),
        "combo_weighted_mix": dict(
            zip(
                tokens,
                ((weights[:, None] * rows).sum(axis=0) / weights.sum()).tolist(),
                strict=True,
            )
        ),
        "named_hands": {
            name: dict(zip(tokens, rows[classes.index(name)].tolist(), strict=True))
            for name in NAMED_HANDS
            if name in classes
        },
        "classes": {name: rows[index].tolist() for index, name in enumerate(classes)},
    }


def _combos_in_class(name: str) -> int:
    """Pairs are 6 combos, suited 4, offsuit 12 -- the weights a range chart uses."""
    if len(name) == 2:
        return 6
    return 4 if name.endswith("s") else 12


__all__ = ("profile_policy",)
