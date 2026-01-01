"""Parity tests for infoset-key encoding extraction."""

from dataclasses import replace

from src.core.game.actions import bet, call
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.infoset_encoder import encode_infoset_key
from tests.test_helpers import DummyCardAbstraction

_RANKS = "AKQJT98765432"
_RANK_VALUES = {rank: 14 - idx for idx, rank in enumerate(_RANKS)}


def _reference_get_spr_bucket(spr: float) -> int:
    if spr < 4.0:
        return 0
    if spr < 13.0:
        return 1
    return 2


def _reference_rank_char(card: Card) -> str:
    card_repr = repr(card)
    if card_repr.startswith("10"):
        return "T"
    return card_repr[0]


def _reference_preflop_hand_string(hole_cards: tuple[Card, Card]) -> str:
    c1, c2 = hole_cards
    r1 = _reference_rank_char(c1)
    r2 = _reference_rank_char(c2)

    if r1 == r2:
        return f"{r1}{r2}"

    if _RANK_VALUES[r1] > _RANK_VALUES[r2]:
        high, low = r1, r2
    else:
        high, low = r2, r1
    suited = repr(c1)[-1] == repr(c2)[-1]
    return f"{high}{low}{'s' if suited else 'o'}"


def _reference_infoset_key(state, player: int, card_abstraction) -> InfoSetKey:
    effective_stack = min(state.stacks)
    spr = effective_stack / state.pot if state.pot > 0 else 0
    spr_bucket = _reference_get_spr_bucket(spr)
    betting_sequence = state._normalize_betting_sequence()

    if state.street == Street.PREFLOP:
        return InfoSetKey(
            player_position=player,
            street=state.street,
            betting_sequence=betting_sequence,
            preflop_hand=_reference_preflop_hand_string(state.hole_cards[player]),
            postflop_bucket=None,
            spr_bucket=spr_bucket,
        )

    return InfoSetKey(
        player_position=player,
        street=state.street,
        betting_sequence=betting_sequence,
        preflop_hand=None,
        postflop_bucket=card_abstraction.get_bucket(
            state.hole_cards[player], state.board, state.street
        ),
        spr_bucket=spr_bucket,
    )


def test_encoder_matches_reference_preflop():
    rules = GameRules(1, 2)
    state = rules.create_initial_state(
        starting_stack=200,
        hole_cards=((Card.new("As"), Card.new("Kd")), (Card.new("Qc"), Card.new("Jh"))),
        button=0,
    )
    card_abstraction = DummyCardAbstraction()

    assert encode_infoset_key(state, 0, card_abstraction) == _reference_infoset_key(
        state, 0, card_abstraction
    )
    assert encode_infoset_key(state, 1, card_abstraction) == _reference_infoset_key(
        state, 1, card_abstraction
    )


def test_encoder_matches_reference_postflop():
    rules = GameRules(1, 2)
    base_state = rules.create_initial_state(
        starting_stack=200,
        hole_cards=((Card.new("Ah"), Card.new("Kh")), (Card.new("Qs"), Card.new("Jd"))),
        button=0,
    )
    state = replace(
        base_state,
        street=Street.FLOP,
        pot=20,
        stacks=(190, 190),
        board=(Card.new("2c"), Card.new("7d"), Card.new("Ts")),
        betting_history=(bet(6), call()),
        to_call=0,
        current_player=0,
        street_start_pot=8,
        _skip_validation=True,
    )
    card_abstraction = DummyCardAbstraction()

    assert encode_infoset_key(state, 0, card_abstraction) == _reference_infoset_key(
        state, 0, card_abstraction
    )
    assert encode_infoset_key(state, 1, card_abstraction) == _reference_infoset_key(
        state, 1, card_abstraction
    )


def test_encoder_matches_reference_spr_boundaries():
    rules = GameRules(1, 2)
    base_state = rules.create_initial_state(
        starting_stack=200,
        hole_cards=((Card.new("Ac"), Card.new("Kc")), (Card.new("Qh"), Card.new("Js"))),
        button=1,
    )
    card_abstraction = DummyCardAbstraction()

    shallow = replace(base_state, pot=40, stacks=(120, 120), _skip_validation=True)
    medium = replace(base_state, pot=20, stacks=(120, 120), _skip_validation=True)
    deep = replace(base_state, pot=8, stacks=(120, 120), _skip_validation=True)

    assert encode_infoset_key(shallow, 0, card_abstraction) == _reference_infoset_key(
        shallow, 0, card_abstraction
    )
    assert encode_infoset_key(medium, 0, card_abstraction) == _reference_infoset_key(
        medium, 0, card_abstraction
    )
    assert encode_infoset_key(deep, 0, card_abstraction) == _reference_infoset_key(
        deep, 0, card_abstraction
    )
