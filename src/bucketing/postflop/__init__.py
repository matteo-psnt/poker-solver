"""
Postflop bucketing with suit isomorphism.

This module provides the foundation for combo-level abstraction under suit isomorphism,
which is required for theoretically sound postflop bucketing in poker solvers.

Key concepts:
- Suits are interchangeable until the board creates distinctions
- Canonical representation assigns suits to labels (0,1,2,3) in order of appearance
- Hands are canonicalized relative to the board's suit mapping
- This eliminates duplicate states and ensures deterministic bucket mapping
"""

from src.bucketing.config import PrecomputeConfig
from src.bucketing.postflop.board_enumeration import (
    CanonicalBoardEnumerator,
    CanonicalBoardInfo,
)
from src.bucketing.postflop.hand_bucketing import (
    CanonicalHand,
    PostflopBucketer,
    get_all_canonical_hands,
)
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.bucketing.postflop.suit_isomorphism import (
    CanonicalCard,
    SuitMapping,
    canonicalize_board,
    canonicalize_hand,
    get_canonical_board_id,
    get_canonical_hand_id,
)

__all__ = [
    "SuitMapping",
    "CanonicalCard",
    "canonicalize_board",
    "canonicalize_hand",
    "get_canonical_hand_id",
    "get_canonical_board_id",
    "CanonicalHand",
    "PostflopBucketer",
    "get_all_canonical_hands",
    "CanonicalBoardEnumerator",
    "CanonicalBoardInfo",
    "PrecomputeConfig",
    "PostflopPrecomputer",
]
