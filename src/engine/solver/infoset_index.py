"""Integer infoset indexing: the single source of truth for infoset identity.

An infoset is ``(public betting node, card bucket)`` and nothing else. This
module turns a live ``GameState`` into the flat row index that pair denotes,
replacing the string-keyed ``InfoSetKey`` hash lookup on the hot path.

Two fields of the old key are deliberately absent, both verified redundant
against the production enumeration:

``spr_bucket``
    A pure function of ``(street, betting_sequence)`` — pot and effective stack
    are both determined by the normalized betting sequence at a fixed starting
    stack. Probed over 876 distinct decision nodes: zero ambiguous. It split
    nothing and cost a divide plus two compares per node visit.

``player_position``
    The absolute seat. The betting tree is exactly button-symmetric, so the
    acting seat relative to the button is already implied by the node, and the
    absolute seat carries no strategic information. Including it duplicated the
    entire infoset space and split the update budget across two identical
    halves — see ``betting_tree`` for the symmetry proof.
"""

from __future__ import annotations

from src.core.game.state import Card, GameState, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.protocols import BucketingStrategy

_RANKS = "AKQJT98765432"
_RANK_VALUES = {rank: 14 - idx for idx, rank in enumerate(_RANKS)}


def _rank_char(card: Card) -> str:
    card_repr = repr(card)
    if card_repr.startswith("10"):
        return "T"
    return card_repr[0]


def _same_suit(card1: Card, card2: Card) -> bool:
    return repr(card1)[-1] == repr(card2)[-1]


def _build_preflop_index() -> dict[tuple[str, str, bool], int]:
    """Canonical 0-168 ordering of starting hands.

    Ordered by high rank, then low rank, then suited-before-offsuit, so the index
    is stable across processes and runs. Pairs are their own category (suitedness
    is meaningless), giving 13 pairs + 78 suited + 78 offsuit = 169.
    """
    index: dict[tuple[str, str, bool], int] = {}
    for hi_pos, high in enumerate(_RANKS):
        for low in _RANKS[hi_pos:]:
            if high == low:
                index[(high, low, False)] = len(index)
            else:
                index[(high, low, True)] = len(index)
                index[(high, low, False)] = len(index)
    return index


_PREFLOP_INDEX = _build_preflop_index()
NUM_PREFLOP_HANDS = len(_PREFLOP_INDEX)


def _build_preflop_strings() -> list[str]:
    """Index -> canonical hand string, the exact inverse of :data:`_PREFLOP_INDEX`."""
    strings = [""] * NUM_PREFLOP_HANDS
    for (high, low, suited), index in _PREFLOP_INDEX.items():
        strings[index] = f"{high}{low}" if high == low else f"{high}{low}{'s' if suited else 'o'}"
    return strings


_PREFLOP_STRINGS = _build_preflop_strings()


def preflop_hand_string_at(index: int) -> str:
    """Canonical hand string for a preflop index — inverse of :func:`preflop_hand_index`.

    Exists so consumers that enumerate preflop buckets by index (the exact-BR
    engine builds a policy row per index) resolve them through the SAME ordering
    the solver stores them under. A second, independently-derived ordering would
    attach every policy row to the wrong hand, silently.
    """
    return _PREFLOP_STRINGS[index]


def preflop_hand_index(hole_cards: tuple[Card, Card]) -> int:
    """Map hole cards to their canonical starting-hand index (0-168)."""
    card1, card2 = hole_cards
    rank1 = _rank_char(card1)
    rank2 = _rank_char(card2)

    if rank1 == rank2:
        return _PREFLOP_INDEX[(rank1, rank2, False)]

    if _RANK_VALUES[rank1] > _RANK_VALUES[rank2]:
        high, low = rank1, rank2
    else:
        high, low = rank2, rank1
    return _PREFLOP_INDEX[(high, low, _same_suit(card1, card2))]


def bucket_of(
    state: GameState,
    player: int,
    card_abstraction: BucketingStrategy,
) -> int:
    """Card bucket for ``player`` at ``state`` — hand index preflop, equity bucket after."""
    if state.street == Street.PREFLOP:
        return preflop_hand_index(state.hole_cards[player])
    return card_abstraction.get_bucket(state.hole_cards[player], state.board, state.street)


def infoset_row(
    state: GameState,
    player: int,
    tree: BettingTree,
    card_abstraction: BucketingStrategy,
) -> int:
    """Flat infoset row for ``player`` at ``state``.

    ``player`` is the acting seat. Its absolute value never enters the index —
    it selects whose hole cards to bucket, and nothing more.
    """
    node_id = tree.node_id(state)
    return tree.row(node_id, bucket_of(state, player, card_abstraction))


def infoset_location(
    state: GameState,
    player: int,
    tree: BettingTree,
    card_abstraction: BucketingStrategy,
) -> tuple[int, int, int]:
    """``(node_id, row, bucket)`` in one lookup, for callers that need all three."""
    node_id = tree.node_id(state)
    bucket = bucket_of(state, player, card_abstraction)
    return node_id, tree.row(node_id, bucket), bucket


__all__ = (
    "NUM_PREFLOP_HANDS",
    "bucket_of",
    "infoset_location",
    "infoset_row",
    "preflop_hand_index",
)
