"""Preflop chart data preparation for the UI server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage.in_memory import InMemoryStorage


@dataclass(frozen=True)
class ChartDataRuntime:
    """Runtime dependencies required to render chart data."""

    action_model: ActionModel
    rules: GameRules
    storage: InMemoryStorage
    starting_stack: int


POSITION_OPTIONS = [
    {"id": 0, "label": "Button (BTN)"},
    {"id": 1, "label": "Big Blind (BB)"},
]
_POSITION_LABELS = {option["id"]: option["label"] for option in POSITION_OPTIONS}


def build_chart_metadata(
    run_id: str,
    action_model: ActionModel,
) -> dict[str, Any]:
    situations: list[dict[str, Any]] = [{"id": "first_to_act", "label": "First to act"}]

    for raise_bb in action_model.get_preflop_open_sizes_bb():
        situations.append(
            {
                "id": f"facing_raise_{raise_bb}",
                "label": f"Facing raise to {_format_bb_size(raise_bb)}",
                "raiseBb": raise_bb,
            }
        )

    return {
        "runId": run_id,
        "positions": POSITION_OPTIONS,
        "situations": situations,
        "defaultPosition": 0,
        "defaultSituation": "first_to_act",
    }


def build_preflop_chart_data(
    source: ChartDataRuntime,
    position: int,
    situation_id: str,
    run_id: str,
) -> dict:
    ranks = "AKQJT98765432"
    situation_label = "First to act"
    state = _initial_state(source)
    if situation_id.startswith("facing_raise_"):
        raise_bb = float(situation_id.removeprefix("facing_raise_"))
        situation_label = f"Facing raise to {_format_bb_size(raise_bb)}"
        raised_state = _apply_open_raise(state, source, raise_bb)
        if raised_state is not None:
            state = raised_state

    betting_sequence = state.normalized_betting_sequence()
    to_call = state.to_call

    grid = []
    action_meta: dict[str, dict] = {}

    for i, r1 in enumerate(ranks):
        row = []
        for j, r2 in enumerate(ranks):
            if i == j:
                hand = f"{r1}{r2}"
                strategy = _get_hand_strategy(
                    source, r1, r2, False, True, position, betting_sequence, to_call
                )
            elif i < j:
                hand = f"{r1}{r2}s"
                strategy = _get_hand_strategy(
                    source, r1, r2, True, False, position, betting_sequence, to_call
                )
            else:
                hand = f"{r2}{r1}o"
                strategy = _get_hand_strategy(
                    source, r2, r1, False, False, position, betting_sequence, to_call
                )

            if strategy is None:
                row.append({"hand": hand, "actions": []})
                continue

            actions = []
            for action in strategy["actions"]:
                action_id = action["id"]
                action_meta[action_id] = {
                    "id": action_id,
                    "label": action["label"],
                    "kind": action["kind"],
                    "sizeBb": action.get("size_bb"),
                }
                actions.append({"id": action_id, "pct": round(action["prob"] * 100, 1)})

            row.append({"hand": hand, "actions": actions})

        grid.append(row)

    ordered_actions = _order_action_meta(action_meta)

    return {
        "runId": run_id,
        "position": position,
        "situation": situation_id,
        "positionLabel": _POSITION_LABELS[position],
        "situationLabel": situation_label,
        "bettingSequence": betting_sequence,
        "ranks": ranks,
        "actions": ordered_actions,
        "grid": grid,
    }


def _format_bb_size(size_bb: float) -> str:
    if abs(size_bb - round(size_bb)) < 1e-6:
        return f"{round(size_bb)}bb"
    return f"{size_bb:.1f}bb"


def _build_action_breakdown(
    legal_actions: list[Action],
    strategy: list[float],
    rules: GameRules,
    to_call: int,
) -> list[dict]:
    breakdown: list[dict] = []

    for action, prob in zip(legal_actions, strategy):
        if prob <= 0:
            continue

        if action.type == ActionType.FOLD:
            breakdown.append({"id": "fold", "label": "Fold", "kind": "fold", "prob": prob})
        elif action.type in (ActionType.CALL, ActionType.CHECK):
            breakdown.append({"id": "call", "label": "Call / Check", "kind": "call", "prob": prob})
        elif action.type == ActionType.ALL_IN:
            size_bb = action.amount / rules.big_blind if rules.big_blind else action.amount
            breakdown.append(
                {
                    "id": "allin",
                    "label": f"All-in ({_format_bb_size(size_bb)})",
                    "kind": "all-in",
                    "prob": prob,
                    "size_bb": size_bb,
                }
            )
        elif action.type in (ActionType.BET, ActionType.RAISE):
            total = action.amount if action.type == ActionType.BET else action.amount + to_call
            size_bb = total / rules.big_blind if rules.big_blind else total
            action_id = f"size_{size_bb:.2f}"
            verb = "Open to" if action.type == ActionType.BET and to_call == 0 else "Raise to"
            breakdown.append(
                {
                    "id": action_id,
                    "label": f"{verb} {_format_bb_size(size_bb)}",
                    "kind": "aggressive",
                    "prob": prob,
                    "size_bb": size_bb,
                }
            )

    return breakdown


# Hole cards do not affect the betting structure; any two disjoint hands work.
_DUMMY_HOLE_CARDS = (
    (Card.new("As"), Card.new("Kd")),
    (Card.new("Qc"), Card.new("Jh")),
)


def _initial_state(source: ChartDataRuntime) -> GameState:
    return source.rules.create_initial_state(
        starting_stack=source.starting_stack,
        hole_cards=_DUMMY_HOLE_CARDS,
        button=0,
    )


def _apply_open_raise(
    state: GameState, source: ChartDataRuntime, raise_bb: float
) -> GameState | None:
    """Apply the open raise to ``raise_bb`` big blinds, or None if not in the tree."""
    total_bet = int(raise_bb * source.rules.big_blind)
    legal_actions = source.rules.get_legal_actions(state, action_model=source.action_model)
    raise_action = next(
        (
            action
            for action in legal_actions
            if action.type == ActionType.RAISE and (action.amount + state.to_call) == total_bet
        ),
        None,
    )

    if raise_action is None:
        return None

    return state.apply_action(raise_action, source.rules)


def _get_hand_strategy(
    source: ChartDataRuntime,
    rank1: str,
    rank2: str,
    suited: bool,
    is_pair: bool,
    position: int,
    betting_sequence: str,
    to_call: int,
) -> dict | None:
    hand_string = _ranks_to_hand_string(rank1, rank2, suited, is_pair)

    spr_bucket = 2

    key = InfoSetKey(
        player_position=position,
        street=Street.PREFLOP,
        betting_sequence=betting_sequence,
        preflop_hand=hand_string,
        postflop_bucket=None,
        spr_bucket=spr_bucket,
    )

    infoset = source.storage.get_infoset(key)

    if infoset is None:
        return None
    if float(infoset.strategy_sum.sum()) == 0.0:
        return None

    strategy = [float(prob) for prob in infoset.get_filtered_strategy(use_average=True)]

    return {
        "actions": _build_action_breakdown(
            infoset.legal_actions,
            strategy,
            source.rules,
            to_call,
        )
    }


def _ranks_to_hand_string(rank1: str, rank2: str, suited: bool, is_pair: bool) -> str:
    ranks = "AKQJT98765432"

    if is_pair:
        return f"{rank1}{rank2}"

    r1_val = 14 - ranks.index(rank1)
    r2_val = 14 - ranks.index(rank2)

    if r1_val > r2_val:
        high, low = rank1, rank2
    else:
        high, low = rank2, rank1

    suffix = "s" if suited else "o"

    return f"{high}{low}{suffix}"


def _order_action_meta(action_meta: dict[str, dict]) -> list[dict]:
    aggressive = [v for v in action_meta.values() if v["kind"] == "aggressive"]
    aggressive.sort(key=lambda item: item.get("sizeBb") or 0)

    all_in = [v for v in action_meta.values() if v["kind"] == "all-in"]
    call = [v for v in action_meta.values() if v["kind"] == "call"]
    fold = [v for v in action_meta.values() if v["kind"] == "fold"]

    return aggressive + all_in + call + fold
