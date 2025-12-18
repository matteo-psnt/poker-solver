#!/usr/bin/env python3
"""
Precompute equity buckets for the poker solver.

This script:
1. Generates representative sample boards for each street
2. Fits board clusterer on sample boards
3. Computes equity for all (hand, board_cluster) pairs
4. Runs K-means clustering to assign buckets
5. Saves results to disk for fast runtime lookup

Runtime: ~1-2 hours on a single core (can be parallelized)
Storage: ~288 KB total
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.abstraction.equity_bucketing import EquityBucketing
from src.abstraction.equity_calculator import EquityCalculator
from src.game.state import Card, Street


def generate_sample_boards(street: Street, num_samples: int, seed: int = 42) -> list:
    """
    Generate random sample boards for a street.

    Args:
        street: Which street to generate for
        num_samples: How many boards to sample
        seed: Random seed for reproducibility

    Returns:
        List of board tuples
    """
    np.random.seed(seed)
    boards = []

    # Get full deck
    deck = Card.get_full_deck()

    # Board size based on street
    board_size = {
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
    }[street]

    print(f"Generating {num_samples} {street.name} boards...")
    for _ in tqdm(range(num_samples), desc="Sampling boards"):
        # Shuffle and take first N cards
        shuffled = np.random.permutation(deck).tolist()
        board = tuple(shuffled[:board_size])
        boards.append(board)

    return boards


def main():
    """Main precomputation pipeline."""
    print("=" * 80)
    print("POKER SOLVER - EQUITY BUCKET PRECOMPUTATION")
    print("=" * 80)
    print()

    # Configuration
    config = {
        "num_samples_per_street": {
            Street.FLOP: 5000,  # 5K flop boards
            Street.TURN: 3000,  # 3K turn boards
            Street.RIVER: 2000,  # 2K river boards
        },
        "num_buckets": {
            Street.FLOP: 50,
            Street.TURN: 100,
            Street.RIVER: 200,
        },
        "num_board_clusters": {
            Street.FLOP: 200,
            Street.TURN: 500,
            Street.RIVER: 1000,
        },
        "num_equity_samples": 1000,  # MC rollouts per equity calculation
        "num_samples_per_cluster": 5,  # Boards to sample per cluster for equity
        "output_file": Path("data/abstractions/equity_buckets.pkl"),
        "seed": 42,
    }

    print("Configuration:")
    print(
        f"  Flop: {config['num_samples_per_street'][Street.FLOP]} boards → "
        f"{config['num_board_clusters'][Street.FLOP]} clusters → "
        f"{config['num_buckets'][Street.FLOP]} buckets"
    )
    print(
        f"  Turn: {config['num_samples_per_street'][Street.TURN]} boards → "
        f"{config['num_board_clusters'][Street.TURN]} clusters → "
        f"{config['num_buckets'][Street.TURN]} buckets"
    )
    print(
        f"  River: {config['num_samples_per_street'][Street.RIVER]} boards → "
        f"{config['num_board_clusters'][Street.RIVER]} clusters → "
        f"{config['num_buckets'][Street.RIVER]} buckets"
    )
    print(f"  Equity samples: {config['num_equity_samples']} rollouts per calculation")
    print()

    # Step 1: Generate sample boards
    print("STEP 1: Generating sample boards")
    print("-" * 80)
    sample_boards = {}
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_samples = config["num_samples_per_street"][street]
        sample_boards[street] = generate_sample_boards(street, num_samples, config["seed"])
    print()

    # Step 2: Create equity bucketing system
    print("STEP 2: Creating equity bucketing system")
    print("-" * 80)
    start_time = time.time()

    equity_calc = EquityCalculator(num_samples=config["num_equity_samples"], seed=config["seed"])
    bucketing = EquityBucketing(
        num_buckets=config["num_buckets"],
        num_board_clusters=config["num_board_clusters"],
        equity_calculator=equity_calc,
    )

    print(f"Created bucketing system in {time.time() - start_time:.2f}s")
    print(f"  {bucketing}")
    print()

    # Step 3: Fit bucketing system
    print("STEP 3: Fitting bucketing system (this will take a while...)")
    print("-" * 80)
    print()
    print("This computes equity for:")
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_clusters = config["num_board_clusters"][street]
        num_samples_per_cluster = config["num_samples_per_cluster"]
        total_equity_calcs = 169 * num_clusters * num_samples_per_cluster
        print(
            f"  {street.name}: 169 hands × {num_clusters} clusters × {num_samples_per_cluster} boards = "
            f"{total_equity_calcs:,} equity calculations"
        )

    total_calcs = sum(
        169 * config["num_board_clusters"][street] * config["num_samples_per_cluster"]
        for street in [Street.FLOP, Street.TURN, Street.RIVER]
    )
    print(
        f"  TOTAL: {total_calcs:,} equity calculations "
        f"({total_calcs * config['num_equity_samples']:,} MC rollouts)"
    )
    print()

    fit_start = time.time()
    bucketing.fit(sample_boards, num_samples_per_cluster=config["num_samples_per_cluster"])
    fit_time = time.time() - fit_start

    print()
    print(f"Fitting completed in {fit_time:.2f}s ({fit_time / 60:.1f} minutes)")
    print()

    # Step 4: Save to disk
    print("STEP 4: Saving to disk")
    print("-" * 80)
    output_file = config["output_file"]
    bucketing.save(output_file)

    # Check file size
    file_size = output_file.stat().st_size
    print(f"Saved to: {output_file}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print()

    # Step 5: Validation
    print("STEP 5: Validation")
    print("-" * 80)
    print("Testing bucket assignments...")

    # Test a few hands
    test_cases = [
        # (hand, board, street, expected_range)
        (
            (Card.new("As"), Card.new("Ah")),
            (Card.new("Ks"), Card.new("Qs"), Card.new("Jh")),
            Street.FLOP,
        ),
        (
            (Card.new("7d"), Card.new("2c")),
            (Card.new("Ks"), Card.new("Qs"), Card.new("Jh")),
            Street.FLOP,
        ),
        (
            (Card.new("Ts"), Card.new("9s")),
            (Card.new("8s"), Card.new("7s"), Card.new("2h")),
            Street.FLOP,
        ),
    ]

    for hole_cards, board, street in test_cases:
        bucket = bucketing.get_bucket(hole_cards, board, street)
        hand_str = f"{hole_cards[0]} {hole_cards[1]}"
        board_str = " ".join(str(c) for c in board)
        print(f"  {hand_str} on {board_str}: Bucket {bucket}")

    print()

    # Summary
    print("=" * 80)
    print("PRECOMPUTATION COMPLETE!")
    print("=" * 80)
    print()
    print(
        f"Total time: {time.time() - start_time:.2f}s ({(time.time() - start_time) / 60:.1f} minutes)"
    )
    print(f"Output file: {output_file.absolute()}")
    print()
    print("The solver can now use this precomputed abstraction for training.")
    print()


if __name__ == "__main__":
    main()
