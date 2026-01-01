"""Information-set key encoding for solver and evaluation code paths."""

from __future__ import annotations

from src.core.game.state import Card, GameState, Street
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.protocols import BucketingStrategy

# SPR (Stack-to-Pot Ratio) thresholds for bucketing
# Shallow: SPR < 4 (push/fold decisions)
# Medium: 4 <= SPR < 13 (standard play)
# Deep: SPR >= 13 (complex postflop play)
SPR_SHALLOW_THRESHOLD = 4.0
SPR_DEEP_THRESHOLD = 13.0

_RANKS = "AKQJT98765432"
_RANK_VALUES = {rank: 14 - idx for idx, rank in enumerate(_RANKS)}


def _rank_char(card: Card) -> str:
    card_repr = repr(card)
    if card_repr.startswith("10"):
        return "T"
    return card_repr[0]


def _same_suit(card1: Card, card2: Card) -> bool:
    return repr(card1)[-1] == repr(card2)[-1]


def _preflop_hand_string(hole_cards: tuple[Card, Card]) -> str:
    c1, c2 = hole_cards
    r1 = _rank_char(c1)
    r2 = _rank_char(c2)

    if r1 == r2:
        return f"{r1}{r2}"

    if _RANK_VALUES[r1] > _RANK_VALUES[r2]:
        high, low = r1, r2
    else:
        high, low = r2, r1
    suffix = "s" if _same_suit(c1, c2) else "o"
    return f"{high}{low}{suffix}"


def _get_spr_bucket(spr: float) -> int:
    """Get SPR bucket (0=shallow, 1=medium, 2=deep)."""
    if spr < SPR_SHALLOW_THRESHOLD:
        return 0
    if spr < SPR_DEEP_THRESHOLD:
        return 1
    return 2


def encode_infoset_key(
    state: GameState,
    player: int,
    card_abstraction: BucketingStrategy,
) -> InfoSetKey:
    """
    Build canonical infoset key for the current state and player.

    This is the single source of truth for infoset-key encoding.
    """
    effective_stack = min(state.stacks)
    spr = effective_stack / state.pot if state.pot > 0 else 0
    spr_bucket = _get_spr_bucket(spr)
    betting_sequence = state._normalize_betting_sequence()

    if state.street == Street.PREFLOP:
        preflop_hand = _preflop_hand_string(state.hole_cards[player])
        return InfoSetKey(
            player_position=player,
            street=state.street,
            betting_sequence=betting_sequence,
            preflop_hand=preflop_hand,
            postflop_bucket=None,
            spr_bucket=spr_bucket,
        )

    postflop_bucket = card_abstraction.get_bucket(
        state.hole_cards[player], state.board, state.street
    )
    return InfoSetKey(
        player_position=player,
        street=state.street,
        betting_sequence=betting_sequence,
        preflop_hand=None,
        postflop_bucket=postflop_bucket,
        spr_bucket=spr_bucket,
    )
