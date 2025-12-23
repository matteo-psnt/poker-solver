#!/usr/bin/env python3
"""Debug canonical flop count."""

from collections import Counter

from src.abstraction.isomorphism.canonical_boards import CanonicalBoardEnumerator
from src.abstraction.isomorphism.suit_canonicalization import (
    canonicalize_board,
    get_canonical_board_id,
)
from src.game.state import Card, Street

flop_enum = CanonicalBoardEnumerator(Street.FLOP)
flop_enum.enumerate()

# Count by number of distinct suits
suit_counts: Counter[int] = Counter()
for info in flop_enum.iterate():
    num_suits = max(c.suit_label for c in info.canonical_board) + 1
    suit_counts[num_suits] += 1

print("Canonical flops by suit pattern:")
for num_suits in sorted(suit_counts.keys()):
    print(f"  {num_suits} suits: {suit_counts[num_suits]} canonical boards")

print(f"Total canonical flops: {flop_enum.count()}")

# Two rainbow boards with same ranks but different suit orderings
b1 = (Card.new("As"), Card.new("Kh"), Card.new("Qd"))
b2 = (Card.new("Ah"), Card.new("Ks"), Card.new("Qd"))  # Different suit pattern

c1, _ = canonicalize_board(b1)
c2, _ = canonicalize_board(b2)

print()
print("Rainbow AKQ boards:")
print(f"  b1 AKQ in s/h/d: {[str(c) for c in c1]} ID: {get_canonical_board_id(c1)}")
print(f"  b2 AKQ in h/s/d: {[str(c) for c in c2]} ID: {get_canonical_board_id(c2)}")
print(f"  Same? {c1 == c2}")
