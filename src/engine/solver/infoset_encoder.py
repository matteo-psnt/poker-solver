"""Information-set key encoding for solver and evaluation code paths."""

from __future__ import annotations

from src.core.game.state import GameState, Street
from src.pipeline.abstraction.base import BucketingStrategy
from src.pipeline.abstraction.preflop.hand_classes import PreflopHandClasses
from src.pipeline.abstraction.utils.infoset import InfoSetKey

# SPR (Stack-to-Pot Ratio) thresholds for bucketing
# Shallow: SPR < 4 (push/fold decisions)
# Medium: 4 <= SPR < 13 (standard play)
# Deep: SPR >= 13 (complex postflop play)
SPR_SHALLOW_THRESHOLD = 4.0
SPR_DEEP_THRESHOLD = 13.0

_PREFLOP_HAND_CLASSES = PreflopHandClasses()


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
        preflop_hand = _PREFLOP_HAND_CLASSES.get_hand_string(state.hole_cards[player])
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
