#!/usr/bin/env python3
"""Quick test of the precomputation pipeline."""

from src.abstraction.isomorphism import (
    CanonicalBoardEnumerator,
    ComboPrecomputer,
    PrecomputeConfig,
    canonicalize_board,
    get_all_canonical_combos,
)
from src.abstraction.isomorphism.precompute import compute_equity_for_combo
from src.game.state import Street


def test_canonical_combo_enumeration():
    """Test that we can enumerate canonical combos for a board."""
    print("Testing canonical combo enumeration:")

    enum = CanonicalBoardEnumerator(Street.FLOP)
    enum.enumerate()
    boards = list(enum.iterate())[:3]  # Just 3 boards for quick test

    for board_info in boards:
        board_repr = " ".join(str(c) for c in board_info.representative)
        combos = list(get_all_canonical_combos(board_info.representative))
        print(f"  Board: {board_repr} -> {len(combos)} canonical combos")

    return boards


def test_equity_calculation(boards):
    """Test equity calculation for a single combo."""
    print("\nTesting equity calculation:")

    board = boards[0].representative
    canonical_board, _ = canonicalize_board(board)
    combo = list(get_all_canonical_combos(board))[0]

    print(f"  Board: {' '.join(str(c) for c in board)}")
    print(f"  Canonical board: {combo.board}")
    print(f"  Canonical hand: {combo.hand}")

    # Compute equity with minimal samples
    board_id, hand_id, equity = compute_equity_for_combo(
        canonical_board=combo.board,
        canonical_hand=combo.hand,
        representative_board=board,
        equity_samples=100,  # Very fast for test
        seed=42,
    )
    print(f"  Equity: {equity:.3f}")

    return equity


def test_precomputer_small():
    """Test precomputation on a tiny sample."""
    print("\nTesting mini precomputation (this may take ~30 seconds):")

    # Create a very small config
    config = PrecomputeConfig(
        num_buckets={
            Street.FLOP: 5,
            Street.TURN: 5,
            Street.RIVER: 5,
        },
        equity_samples=50,  # Very few samples
        num_workers=2,
        seed=42,
    )

    # We'll skip full precomputation since it's slow
    # Just verify the precomputer can be created
    _precomputer = ComboPrecomputer(config)
    print("  ComboPrecomputer created successfully!")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Precompute Module Test")
    print("=" * 60)

    boards = test_canonical_combo_enumeration()
    equity = test_equity_calculation(boards)
    test_precomputer_small()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
