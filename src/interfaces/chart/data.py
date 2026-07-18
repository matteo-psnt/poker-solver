"""Preflop chart presentation for the UI server.

Pure formatting: grid assembly, labels, and payload shapes. All strategy
access (state construction, key encoding, storage reads) lives in
:mod:`src.pipeline.evaluation.preflop_chart`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.core.game.actions import Action, ActionType
from src.pipeline.evaluation.preflop_chart import RANKS, PreflopChartData

POSITION_OPTIONS = [
    {"id": 0, "label": "Button (BTN)"},
    {"id": 1, "label": "Big Blind (BB)"},
]
_POSITION_LABELS = {option["id"]: option["label"] for option in POSITION_OPTIONS}


def parse_situation(situation_id: str) -> float | None:
    """Open-raise size (bb) encoded in ``situation_id``, or None for first-to-act."""
    if situation_id.startswith("facing_raise_"):
        return float(situation_id.removeprefix("facing_raise_"))
    return None


def build_chart_metadata(run_id: str, open_sizes_bb: Sequence[float]) -> dict[str, Any]:
    situations: list[dict[str, Any]] = [{"id": "first_to_act", "label": "First to act"}]

    for raise_bb in open_sizes_bb:
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


def render_preflop_chart(
    chart: PreflopChartData,
    *,
    run_id: str,
    position: int,
    situation_id: str,
) -> dict:
    """Format one chart query's strategy data into the viewer payload."""
    raise_bb = parse_situation(situation_id)
    situation_label = (
        f"Facing raise to {_format_bb_size(raise_bb)}" if raise_bb is not None else "First to act"
    )

    grid = []
    action_meta: dict[str, dict] = {}

    for i, r1 in enumerate(RANKS):
        row = []
        for j, r2 in enumerate(RANKS):
            if i == j:
                hand = f"{r1}{r2}"
            elif i < j:
                hand = f"{r1}{r2}s"
            else:
                hand = f"{r2}{r1}o"

            strategy = chart.hands.get(hand)
            if strategy is None:
                row.append({"hand": hand, "actions": []})
                continue

            actions = []
            breakdown = _build_action_breakdown(
                strategy.actions, strategy.probabilities, chart.big_blind, chart.to_call
            )
            for action in breakdown:
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

    return {
        "runId": run_id,
        "position": position,
        "situation": situation_id,
        "positionLabel": _POSITION_LABELS[position],
        "situationLabel": situation_label,
        "bettingSequence": chart.betting_sequence,
        "ranks": RANKS,
        "actions": _order_action_meta(action_meta),
        "grid": grid,
    }


def _format_bb_size(size_bb: float) -> str:
    if abs(size_bb - round(size_bb)) < 1e-6:
        return f"{round(size_bb)}bb"
    return f"{size_bb:.1f}bb"


def _build_action_breakdown(
    legal_actions: Sequence[Action],
    strategy: Sequence[float],
    big_blind: int,
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
            size_bb = action.amount / big_blind if big_blind else action.amount
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
            size_bb = total / big_blind if big_blind else total
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


def _order_action_meta(action_meta: dict[str, dict]) -> list[dict]:
    aggressive = [v for v in action_meta.values() if v["kind"] == "aggressive"]
    aggressive.sort(key=lambda item: item.get("sizeBb") or 0)

    all_in = [v for v in action_meta.values() if v["kind"] == "all-in"]
    call = [v for v in action_meta.values() if v["kind"] == "call"]
    fold = [v for v in action_meta.values() if v["kind"] == "fold"]

    return aggressive + all_in + call + fold
