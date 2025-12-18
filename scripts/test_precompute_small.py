#!/usr/bin/env python3
"""
Quick test of equity bucket precomputation with small sample sizes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.abstraction.equity_bucketing import EquityBucketing
from src.abstraction.equity_calculator import EquityCalculator
from src.game.state import Card, Street


def generate_sample_boards(street: Street, num_samples: int) -> list:
    """Generate random sample boards."""
    np.random.seed(42)
    boards = []
    deck = Card.get_full_deck()

    board_size = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]

    for _ in range(num_samples):
        shuffled = np.random.permutation(deck).tolist()
        board = tuple(shuffled[:board_size])
        boards.append(board)

    return boards


def main():
    print("Testing equity bucket precomputation (small sample)...")
    print()

    # Small test configuration
    sample_boards = {
        Street.FLOP: generate_sample_boards(Street.FLOP, 20),  # Very small
    }

    print(f"Generated {len(sample_boards[Street.FLOP])} flop boards")

    # Create bucketing with small sizes
    equity_calc = EquityCalculator(num_samples=50, seed=42)  # Fast MC
    bucketing = EquityBucketing(
        num_buckets={Street.FLOP: 5},  # Only 5 buckets
        num_board_clusters={Street.FLOP: 3},  # Only 3 clusters
        equity_calculator=equity_calc,
    )

    print(f"Bucketing config: {bucketing}")
    print()

    # Fit
    print("Fitting (169 hands × 3 clusters × 2 boards = 1,014 equity calculations)...")
    bucketing.fit(sample_boards, num_samples_per_cluster=2)
    print("Fitting complete!")
    print()

    # Save
    output_file = Path("data/abstractions/test_equity_buckets.pkl")
    bucketing.save(output_file)
    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size} bytes")
    print()

    # Test
    print("Testing bucket assignments...")
    test_cases = [
        ((Card.new("As"), Card.new("Ah")), (Card.new("Ks"), Card.new("Qs"), Card.new("Jh"))),
        ((Card.new("7d"), Card.new("2c")), (Card.new("Ks"), Card.new("Qs"), Card.new("Jh"))),
    ]

    for hole_cards, board in test_cases:
        bucket = bucketing.get_bucket(hole_cards, board, Street.FLOP)
        print(
            f"  {hole_cards[0]} {hole_cards[1]} on {board[0]} {board[1]} {board[2]}: Bucket {bucket}"
        )

    print()
    print("Success! Precomputation works.")


if __name__ == "__main__":
    main()
