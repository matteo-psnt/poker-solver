"""Preflop strategy extraction for chart rendering.

Chart presentation (grid assembly, labels, payload shapes) lives in
``src.interfaces.chart``; everything that touches solver internals lives here.
Query nodes are real ``GameState``s and keys go through the canonical
:func:`~src.engine.solver.infoset_encoder.encode_infoset_key` — the previous
renderer hand-built an ``InfoSetKey`` with an assumed SPR bucket, which
silently rendered blank charts whenever key encoding, SPR bucketing, or the
stack configuration drifted from that assumption. Routed through the encoder,
the chart follows the solver automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.game.actions import Action, ActionType
from src.core.game.state import Card, GameState
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.engine.solver.protocols import Blueprint

RANKS = "AKQJT98765432"
_SUITS = "shdc"

# Betting structure is card-independent; any two disjoint dummy hands work.
# The queried seat's cards are replaced per hand class.
_DUMMY_HOLE_CARDS = (
    (Card.new("As"), Card.new("Kd")),
    (Card.new("Qc"), Card.new("Jh")),
)


@dataclass(frozen=True)
class HandStrategy:
    """Stored strategy for one preflop hand class at a query node."""

    actions: tuple[Action, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class PreflopChartData:
    """Strategy data for one (position, situation) chart query, presentation-free.

    ``hands`` maps canonical class strings ("AA", "AKs", "AKo") to trained
    strategies; untrained classes are absent. ``applied_raise`` is False when an
    ``open_raise_bb`` was requested but is not in the blueprint's action tree —
    the data then describes the unraised node instead (the renderer's historical
    fallback).
    """

    betting_sequence: str
    to_call: int
    big_blind: int
    applied_raise: bool
    hands: dict[str, HandStrategy]


def preflop_open_sizes_bb(blueprint: Blueprint) -> list[float]:
    """Open-raise sizes (in bb) the blueprint trained on, for situation menus."""
    return blueprint.action_model.get_preflop_open_sizes_bb()


def preflop_chart_data(
    blueprint: Blueprint,
    position: int,
    open_raise_bb: float | None = None,
) -> PreflopChartData:
    """Read the blueprint's preflop strategy for every hand class at one node.

    ``position`` is the queried seat (0 = button, 1 = big blind);
    ``open_raise_bb`` selects the facing-a-raise node when given.
    """
    rules = blueprint.rules
    state = rules.create_initial_state(
        starting_stack=blueprint.config.game.starting_stack,
        hole_cards=_DUMMY_HOLE_CARDS,
        button=0,
    )
    applied_raise = False
    if open_raise_bb is not None:
        raised = _apply_open_raise(blueprint, state, open_raise_bb)
        if raised is not None:
            state = raised
            applied_raise = True

    blocked = frozenset(state.hole_cards[1 - position])
    hands: dict[str, HandStrategy] = {}
    for hand in _hand_classes():
        combo = _combo_for_class(hand, blocked)
        hole_cards = list(state.hole_cards)
        hole_cards[position] = combo
        query_state = state.replace(hole_cards=(hole_cards[0], hole_cards[1]), validate=False)
        key = encode_infoset_key(query_state, position, blueprint.card_abstraction)
        infoset = blueprint.storage.get_infoset(key)
        if infoset is None or float(infoset.strategy_sum.sum()) == 0.0:
            continue
        probabilities = infoset.get_filtered_strategy(use_average=True)
        hands[hand] = HandStrategy(
            actions=tuple(infoset.legal_actions),
            probabilities=tuple(float(p) for p in probabilities),
        )

    return PreflopChartData(
        betting_sequence=state.normalized_betting_sequence(),
        to_call=state.to_call,
        big_blind=rules.big_blind,
        applied_raise=applied_raise,
        hands=hands,
    )


def _apply_open_raise(blueprint: Blueprint, state: GameState, raise_bb: float) -> GameState | None:
    """Apply the open raise to ``raise_bb`` big blinds, or None if not in the tree."""
    total_bet = int(raise_bb * blueprint.rules.big_blind)
    legal_actions = blueprint.rules.get_legal_actions(state, action_model=blueprint.action_model)
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
    return state.apply_action(raise_action, blueprint.rules)


def _hand_classes() -> list[str]:
    """All 169 canonical preflop hand classes, high rank first."""
    classes: list[str] = []
    for i, high in enumerate(RANKS):
        classes.append(f"{high}{high}")
        for low in RANKS[i + 1 :]:
            classes.append(f"{high}{low}s")
            classes.append(f"{high}{low}o")
    return classes


def _combo_for_class(hand: str, blocked: frozenset[Card]) -> tuple[Card, Card]:
    """A representative combo for ``hand`` avoiding ``blocked`` cards.

    The encoder collapses combos of a class to the same key, so any live
    representative works; avoiding the opponent's dummy cards keeps the query
    state physically consistent. At most two cards are blocked, so a valid
    suit assignment always exists.
    """
    rank_high, rank_low = hand[0], hand[1]
    suited = len(hand) == 3 and hand[2] == "s"
    for suit_high in _SUITS:
        card_high = Card.new(f"{rank_high}{suit_high}")
        if card_high in blocked:
            continue
        for suit_low in _SUITS:
            # Suited classes share the suit; offsuit classes and pairs must not.
            if suited != (suit_low == suit_high):
                continue
            card_low = Card.new(f"{rank_low}{suit_low}")
            if card_low not in blocked:
                return card_high, card_low
    raise ValueError(f"No unblocked combo for hand class {hand!r}")
