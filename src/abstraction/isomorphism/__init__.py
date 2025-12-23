"""
Suit isomorphism module for canonical hand/board representation.

This module provides the foundation for combo-level abstraction under suit isomorphism,
which is required for theoretically sound postflop bucketing in poker solvers.

Key concepts:
- Suits are interchangeable until the board creates distinctions
- Canonical representation assigns suits to labels (0,1,2,3) in order of appearance
- Hands are canonicalized relative to the board's suit mapping
- This eliminates duplicate states and ensures deterministic bucket mapping
"""

from src.abstraction.isomorphism.canonical_boards import (
    CanonicalBoardEnumerator,
    CanonicalBoardInfo,
)
from src.abstraction.isomorphism.combo_abstraction import (
    CanonicalCombo,
    ComboAbstraction,
    get_all_canonical_combos,
)
from src.abstraction.isomorphism.precompute import (
    ComboPrecomputer,
    PrecomputeConfig,
)
from src.abstraction.isomorphism.suit_canonicalization import (
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
    "CanonicalCombo",
    "ComboAbstraction",
    "get_all_canonical_combos",
    "CanonicalBoardEnumerator",
    "CanonicalBoardInfo",
    "PrecomputeConfig",
    "ComboPrecomputer",
]
