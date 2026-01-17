"""Preflop chart data preparation for the UI server."""

from __future__ import annotations

from typing import Optional

from src.actions.betting_actions import BettingActions
from src.bucketing.utils.infoset import InfoSetKey
from src.game.actions import Action, ActionType
from src.game.rules import GameRules
from src.game.state import Card, Street
from src.solver.mccfr import MCCFRSolver

POSITION_OPTIONS = [
    {"id": 0, "label": "Button (BTN)"},
    {"id": 1, "label": "Big Blind (BB)"},
]

SITUATION_OPTIONS = [
    {"id": "first_to_act", "label": "First to act"},
    {"id": "facing_raise", "label": "Facing raise"},
]


def build_chart_metadata(
    run_id: str,
    action_abstraction: BettingActions,
) -> dict:
    situations = [{"id": "first_to_act", "label": "First to act"}]

    for raise_bb in action_abstraction.preflop_raises:
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
    solver: MCCFRSolver,
    position: int,
    situation_id: str,
    run_id: str,
) -> dict:
    ranks = "AKQJT98765432"
    situation_label = "First to act"
    betting_sequence = ""
    if situation_id == "first_to_act":
        situation_label = "First to act"
        betting_sequence = ""
    elif situation_id.startswith("facing_raise_"):
        size_text = situation_id.replace("facing_raise_", "")
        raise_bb = float(size_text)
        situation_label = f"Facing raise to {_format_bb_size(raise_bb)}"
        betting_sequence = _generate_betting_sequence_for_raise(
            action_abstraction=solver.action_abstraction,
            rules=solver.rules,
            starting_stack=solver.config.game.starting_stack,
            raise_bb=raise_bb,
        )

    grid = []
    action_meta: dict[str, dict] = {}

    for i, r1 in enumerate(ranks):
        row = []
        for j, r2 in enumerate(ranks):
            if i == j:
                hand = f"{r1}{r2}"
                strategy = _get_hand_strategy(
                    solver, r1, r2, False, True, position, betting_sequence
                )
            elif i < j:
                hand = f"{r1}{r2}s"
                strategy = _get_hand_strategy(
                    solver, r1, r2, True, False, position, betting_sequence
                )
            else:
                hand = f"{r2}{r1}o"
                strategy = _get_hand_strategy(
                    solver, r2, r1, False, False, position, betting_sequence
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
        "positionLabel": "Button (BTN)" if position == 0 else "Big Blind (BB)",
        "situationLabel": situation_label,
        "bettingSequence": betting_sequence,
        "ranks": ranks,
        "actions": ordered_actions,
        "grid": grid,
    }


def _format_bb_size(size_bb: float) -> str:
    if abs(size_bb - round(size_bb)) < 1e-6:
        return f"{int(round(size_bb))}bb"
    return f"{size_bb:.1f}bb"


def _infer_to_call(
    legal_actions: list[Action],
    action_abstraction: BettingActions,
    rules: GameRules,
) -> int:
    if any(action.type == ActionType.BET for action in legal_actions):
        return 0

    raise_actions = [a for a in legal_actions if a.type == ActionType.RAISE]
    if not raise_actions:
        return 0

    total_bets = [int(bb * rules.big_blind) for bb in action_abstraction.preflop_raises]
    common_to_calls: set[int] | None = None

    for action in raise_actions:
        candidates = {total - action.amount for total in total_bets if total > action.amount}
        common_to_calls = candidates if common_to_calls is None else common_to_calls & candidates

    if common_to_calls:
        return min(common_to_calls)

    for action in raise_actions:
        for total in total_bets:
            to_call = total - action.amount
            if to_call > 0:
                return to_call

    return 0


def _build_action_breakdown(
    legal_actions: list[Action],
    strategy: list[float],
    action_abstraction: BettingActions,
    rules: GameRules,
) -> list[dict]:
    to_call = _infer_to_call(legal_actions, action_abstraction, rules)
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


def _generate_betting_sequence_for_situation(
    action_abstraction: BettingActions,
    rules: GameRules,
    situation: str,
    starting_stack: int = 200,
) -> str:
    if "First" in situation:
        return ""

    dummy_hole_cards = (
        (Card.new("As"), Card.new("Kd")),
        (Card.new("Qc"), Card.new("Jh")),
    )
    state = rules.create_initial_state(
        starting_stack=starting_stack * rules.big_blind,
        hole_cards=dummy_hole_cards,
        button=0,
    )

    legal_actions = action_abstraction.get_legal_actions(state)
    raise_action = next(
        (action for action in legal_actions if action.type == ActionType.RAISE), None
    )

    if raise_action is None:
        return ""

    new_state = state.apply_action(raise_action, rules)
    return new_state._normalize_betting_sequence()


def _generate_betting_sequence_for_raise(
    action_abstraction: BettingActions,
    rules: GameRules,
    starting_stack: int,
    raise_bb: float,
) -> str:
    dummy_hole_cards = (
        (Card.new("As"), Card.new("Kd")),
        (Card.new("Qc"), Card.new("Jh")),
    )
    state = rules.create_initial_state(
        starting_stack=starting_stack,
        hole_cards=dummy_hole_cards,
        button=0,
    )

    total_bet = int(raise_bb * rules.big_blind)
    legal_actions = action_abstraction.get_legal_actions(state)
    raise_action = next(
        (
            action
            for action in legal_actions
            if action.type == ActionType.RAISE and (action.amount + state.to_call) == total_bet
        ),
        None,
    )

    if raise_action is None:
        return ""

    new_state = state.apply_action(raise_action, rules)
    return new_state._normalize_betting_sequence()


def _get_hand_strategy(
    solver: MCCFRSolver,
    rank1: str,
    rank2: str,
    suited: bool,
    is_pair: bool,
    position: int,
    betting_sequence: str,
) -> Optional[dict]:
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

    infoset = solver.storage.get_infoset(key)

    if infoset is None:
        return None
    if float(infoset.strategy_sum.sum()) == 0.0:
        return None

    strategy = [float(prob) for prob in infoset.get_filtered_strategy(use_average=True)]

    return {
        "actions": _build_action_breakdown(
            infoset.legal_actions,
            strategy,
            solver.action_abstraction,
            solver.rules,
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
