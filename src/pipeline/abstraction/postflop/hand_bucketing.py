"""
Combo-level abstraction for postflop poker.

This module provides the abstraction layer that operates on concrete hand combos
rather than 169 rank classes. It uses suit isomorphism to reduce the state space
while preserving all strategically relevant suit information.

Key difference from 169-class abstraction:
- 169 classes treat AKs as a single entity, ignoring which suits
- Combo-level tracks actual suits relative to the board
- A♠K♠ on T♠9♠8♣ is different from A♥K♥ on T♠9♠8♣ (flush vs no flush)
- But A♠K♠ on T♠9♠8♣ is equivalent to A♥K♥ on T♥9♥8♣ (suit isomorphism)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.game.state import Card, Street
from src.pipeline.abstraction.base import BucketingStrategy
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    RANKS,
    SUITS,
    CanonicalCard,
    SuitMapping,
    canonicalize_board,
    canonicalize_hand,
    get_canonical_board_id,
    get_canonical_hand_id,
)
from src.pipeline.abstraction.preflop.hand_classes import PreflopHandClasses

if TYPE_CHECKING:
    from src.pipeline.abstraction.postflop.board_clustering import BoardClusterer


@dataclass(frozen=True)
class CanonicalHand:
    """
    A canonical (hand, board) pair.

    Represents a postflop situation in canonical form, where:
    - Board suits are assigned labels 0,1,2,3 in order of appearance
    - Hand suits use the board's mapping, extending for new suits

    This is the fundamental unit for combo-level bucketing.
    """

    hand: tuple[CanonicalCard, CanonicalCard]
    board: tuple[CanonicalCard, ...]

    @property
    def hand_id(self) -> int:
        """Unique ID for the canonical hand."""
        return get_canonical_hand_id(self.hand)

    @property
    def board_id(self) -> int:
        """Unique ID for the canonical board."""
        return get_canonical_board_id(self.board)

    def to_key(self) -> tuple[int, int]:
        """Get (board_id, hand_id) tuple for dictionary keys."""
        return (self.board_id, self.hand_id)

    def __repr__(self) -> str:
        hand_str = f"({self.hand[0]}, {self.hand[1]})"
        board_str = " ".join(str(c) for c in self.board)
        return f"CanonicalHand(hand={hand_str}, board=[{board_str}])"


class PostflopBucketer(BucketingStrategy):
    """
    Combo-level abstraction for postflop bucket lookup.

    Uses board clustering for tractable precomputation:
    - Boards are clustered by texture (30-300 clusters per street)
    - Equity computed only for representative boards per cluster
    - Runtime lookup: canonicalize board -> predict cluster -> lookup bucket

    Storage format (sparse, cluster-based):
        {street: {cluster_id: {canonical_hand_id: bucket_id}}}
    """

    def __init__(self):
        """Initialize empty abstraction."""
        # Bucket assignments: street -> cluster_id -> hand_id -> bucket
        self._buckets: dict[Street, dict[int, dict[int, int]]] = {
            Street.FLOP: {},
            Street.TURN: {},
            Street.RIVER: {},
        }

        # Bucket counts per street (set during precomputation)
        self._num_buckets: dict[Street, int] = {}

        # Board clusterer for runtime cluster prediction
        # Set during precomputation
        self._board_clusterer: BoardClusterer | None = None

        # Preflop buckets (169 hand classes)
        self._preflop_buckets: dict[int, int] | None = None
        self._num_preflop_buckets: int = 169  # One bucket per hand class by default

        # Fallback statistics (for monitoring abstraction coverage)
        self._fallback_count: int = 0
        self._total_lookups: int = 0

    def __getstate__(self) -> dict:
        """Pickle support."""
        return dict(self.__dict__)

    def __setstate__(self, state: dict) -> None:
        """Unpickle support."""
        self.__dict__.update(state)

    def canonicalize(self, hole_cards: tuple[Card, Card], board: tuple[Card, ...]) -> CanonicalHand:
        """
        Canonicalize a (hand, board) pair.

        Args:
            hole_cards: Player's hole cards
            board: Current board

        Returns:
            CanonicalHand representing the canonical form
        """
        # First canonicalize the board to establish suit mapping
        canonical_board, suit_mapping = canonicalize_board(board)

        # Then canonicalize hand relative to board's suit mapping
        canonical_hand = canonicalize_hand(hole_cards, suit_mapping)

        return CanonicalHand(hand=canonical_hand, board=canonical_board)

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        """
        Get bucket ID for a (hand, board) pair.

        Pipeline:
        1. Canonicalize (hand, board) pair
        2. Predict board cluster from canonical board
        3. Lookup bucket from (cluster_id, hand_id)

        Args:
            hole_cards: Player's hole cards
            board: Current board
            street: Current street

        Returns:
            Bucket ID

        Raises:
            KeyError: If bucket not found (abstraction not fitted for this state)
        """
        # Handle preflop separately (uses 169 hand classes)
        if street == Street.PREFLOP:
            mapper = PreflopHandClasses()
            hand_index = mapper.get_hand_index(hole_cards)
            # For now, each hand class is its own bucket
            return hand_index

        # Canonicalize (hand, board) pair
        combo = self.canonicalize(hole_cards, board)

        # Predict board cluster
        if self._board_clusterer is None:
            raise ValueError("Board clusterer not initialized. Load precomputed abstraction first.")

        # Prefer fast lookup via canonical board ID; fall back to model prediction
        cluster_id = self._board_clusterer.get_cluster(combo.board_id, street)
        if cluster_id is None:
            # Predict cluster using the ACTUAL board cards (not canonical form)
            # BoardClusterer.predict() handles canonicalization via feature extraction
            cluster_id = self._board_clusterer.predict(board, street)

        # Lookup bucket from (cluster_id, hand_id)
        street_buckets = self._buckets.get(street, {})
        cluster_buckets = street_buckets.get(cluster_id, {})

        # Track lookup statistics
        self._total_lookups += 1

        if combo.hand_id not in cluster_buckets:
            # Fallback: Use nearest hand from this cluster
            # This happens when a hand wasn't seen with the representative boards
            # during precomputation, but should still get a reasonable bucket
            self._fallback_count += 1

            if not cluster_buckets:
                raise KeyError(
                    f"No buckets computed for cluster={cluster_id}, street={street.name}. "
                    "This cluster may not have had any valid hand combinations during precomputation."
                )

            # Find the nearest hand_id that exists
            # Simple approach: use median bucket for this cluster
            available_buckets = list(cluster_buckets.values())
            median_bucket = sorted(available_buckets)[len(available_buckets) // 2]

            return median_bucket

        return cluster_buckets[combo.hand_id]

    def get_fallback_stats(self) -> dict:
        """Get fallback statistics for monitoring abstraction coverage."""
        if self._total_lookups == 0:
            return {"total_lookups": 0, "fallback_count": 0, "fallback_rate": 0.0}
        return {
            "total_lookups": self._total_lookups,
            "fallback_count": self._fallback_count,
            "fallback_rate": self._fallback_count / self._total_lookups,
        }

    def reset_stats(self):
        """Reset lookup statistics."""
        self._fallback_count = 0
        self._total_lookups = 0

    def get_bucket_distribution(self, street: Street) -> dict[int, int]:
        """Get bucket_id -> combo_count distribution for a given street."""
        if street == Street.PREFLOP:
            return {}

        counts: dict[int, int] = {}
        for cluster_buckets in self._buckets.get(street, {}).values():
            for bucket_id in cluster_buckets.values():
                counts[bucket_id] = counts.get(bucket_id, 0) + 1
        return counts

    def num_buckets(self, street: Street) -> int:
        """Get number of buckets for a street."""
        if street == Street.PREFLOP:
            return self._num_preflop_buckets
        return self._num_buckets.get(street, 0)

    def set_bucket(self, canonical_combo: CanonicalHand, street: Street, bucket_id: int):
        """
        Set bucket for a canonical combo.

        Used during precomputation.
        """
        board_id = canonical_combo.board_id
        hand_id = canonical_combo.hand_id

        if board_id not in self._buckets[street]:
            self._buckets[street][board_id] = {}

        self._buckets[street][board_id][hand_id] = bucket_id

    def set_num_buckets(self, street: Street, num: int):
        """Set number of buckets for a street."""
        if street == Street.PREFLOP:
            self._num_preflop_buckets = num
        else:
            self._num_buckets[street] = num

    def __str__(self) -> str:
        """String representation."""
        buckets_str = []
        for street in [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]:
            n = self.num_buckets(street)
            if n > 0:
                buckets_str.append(f"{street.name}={n}")
        return f"PostflopBucketer({', '.join(buckets_str)})"


def generate_all_cards() -> list[Card]:
    """Generate all 52 cards."""
    cards = []
    for rank in RANKS:
        for suit in SUITS:
            cards.append(Card.new(f"{rank}{suit}"))
    return cards


def generate_all_combos() -> list[tuple[Card, Card]]:
    """
    Generate all 1326 unique 2-card combinations.

    Returns:
        List of (card1, card2) tuples where card1 < card2
    """
    cards = generate_all_cards()
    combos = []

    for i, c1 in enumerate(cards):
        for c2 in cards[i + 1 :]:
            combos.append((c1, c2))

    return combos


def get_all_canonical_hands(
    board: tuple[Card, ...], exclude_board_cards: bool = True
) -> Iterator[CanonicalHand]:
    """
    Generate all canonical combos for a given board.

    Args:
        board: The board cards
        exclude_board_cards: If True, exclude hands that share cards with board

    Yields:
        CanonicalHand objects
    """
    # Canonicalize board first
    canonical_board, suit_mapping = canonicalize_board(board)
    board_card_set = set(board)

    # Track seen canonical hands to avoid duplicates
    seen_canonical: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    # Generate all valid hole card combinations
    cards = generate_all_cards()

    for i, c1 in enumerate(cards):
        if exclude_board_cards and c1 in board_card_set:
            continue

        for c2 in cards[i + 1 :]:
            if exclude_board_cards and c2 in board_card_set:
                continue

            # Canonicalize this hand relative to the board
            canonical_hand = canonicalize_hand((c1, c2), suit_mapping)
            canonical_key = (canonical_hand[0].to_tuple(), canonical_hand[1].to_tuple())

            # Skip if we've already seen this canonical hand
            if canonical_key in seen_canonical:
                continue

            seen_canonical.add(canonical_key)

            yield CanonicalHand(hand=canonical_hand, board=canonical_board)


def count_canonical_hands_for_board(board: tuple[Card, ...]) -> int:
    """
    Count unique canonical combos for a board.

    Useful for estimating storage requirements.
    """
    return sum(1 for _ in get_all_canonical_hands(board))


def get_representative_hand(
    canonical_hand: tuple[CanonicalCard, CanonicalCard], suit_mapping: SuitMapping
) -> tuple[Card, Card]:
    """
    Convert a canonical hand back to a concrete hand.

    Uses the inverse of the suit mapping to get real suits.

    Args:
        canonical_hand: Canonical hand representation
        suit_mapping: Mapping used to canonicalize

    Returns:
        Tuple of two Card objects
    """
    # Invert the mapping
    inv_mapping = {v: k for k, v in suit_mapping.mapping.items()}

    # Assign remaining suits for any canonical labels not in mapping
    available_suits = [s for s in SUITS if s not in suit_mapping.mapping]
    for label in range(4):
        if label not in inv_mapping:
            if available_suits:
                inv_mapping[label] = available_suits.pop(0)
            else:
                inv_mapping[label] = "s"  # Fallback

    cards = []
    for cc in canonical_hand:
        rank_char = RANKS[cc.rank_idx]
        suit_char = inv_mapping[cc.suit_label]
        cards.append(Card.new(f"{rank_char}{suit_char}"))

    return (cards[0], cards[1])
