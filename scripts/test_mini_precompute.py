#!/usr/bin/env python3
"""
Mini end-to-end test of combo-level abstraction.

This runs a tiny precomputation (just a few boards) to verify
the full pipeline works and produces conflict-free buckets.
"""

from src.abstraction.isomorphism import (
    CanonicalBoardEnumerator,
    canonicalize_board,
    canonicalize_hand,
)
from src.abstraction.isomorphism.precompute import _worker_compute_board_equities
from src.game.state import Card, Street


def test_mini_precompute():
    """Run a mini precomputation on just a few boards."""
    print("=" * 60)
    print("Mini Precomputation Test")
    print("=" * 60)

    # Get just 5 boards for quick test
    enum = CanonicalBoardEnumerator(Street.FLOP)
    enum.enumerate()
    boards = list(enum.iterate())[:5]

    print(f"\nProcessing {len(boards)} canonical boards:")

    all_results = []
    for i, board_info in enumerate(boards):
        board_str = " ".join(str(c) for c in board_info.representative)

        # Compute equities for this board
        results = _worker_compute_board_equities((board_info, 100, 42))
        all_results.extend(results)

        print(f"  [{i + 1}] {board_str}: {len(results)} combos")

    print(f"\nTotal combos computed: {len(all_results)}")

    # Check for conflicts
    print("\nChecking for conflicts...")

    # Group by board_id
    by_board = {}
    for board_id, hand_id, equity in all_results:
        if board_id not in by_board:
            by_board[board_id] = {}
        if hand_id in by_board[board_id]:
            print(f"  CONFLICT: board={board_id}, hand={hand_id}")
            print(f"    Old equity: {by_board[board_id][hand_id]:.3f}")
            print(f"    New equity: {equity:.3f}")
            return False
        by_board[board_id][hand_id] = equity

    print("  No conflicts! Each (board, hand) pair has exactly one equity value.")

    # Show equity distribution
    all_equities = [e for _, _, e in all_results if e >= 0]
    print("\nEquity distribution:")
    print(f"  Min: {min(all_equities):.3f}")
    print(f"  Max: {max(all_equities):.3f}")
    print(f"  Mean: {sum(all_equities) / len(all_equities):.3f}")

    return True


def test_bucket_assignment():
    """Test that K-means bucketing works correctly."""
    print("\n" + "=" * 60)
    print("K-means Bucketing Test")
    print("=" * 60)

    import numpy as np
    from sklearn.cluster import KMeans

    # Simulate some equity values
    np.random.seed(42)
    n_combos = 500
    equities = np.random.beta(2, 2, size=n_combos)  # Bell-shaped around 0.5

    # Cluster into 10 buckets
    n_buckets = 10
    features_matrix = equities.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_buckets, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_matrix)

    # Sort buckets by center
    centers = kmeans.cluster_centers_.flatten()
    center_order = np.argsort(centers)
    label_map = {old: new for new, old in enumerate(center_order)}
    sorted_labels = [label_map[label] for label in labels]

    print(f"\nClustered {n_combos} combos into {n_buckets} buckets:")
    for bucket in range(n_buckets):
        count = sum(1 for label in sorted_labels if label == bucket)
        bucket_equities = [e for e, label in zip(equities, sorted_labels) if label == bucket]
        min_eq = min(bucket_equities)
        max_eq = max(bucket_equities)
        print(f"  Bucket {bucket}: {count:3d} combos, equity range [{min_eq:.3f}, {max_eq:.3f}]")

    return True


def test_isomorphic_hands_same_bucket():
    """Test that isomorphic hands map to the same bucket."""
    print("\n" + "=" * 60)
    print("Isomorphism Consistency Test")
    print("=" * 60)

    # Two isomorphic boards
    board1 = (Card.new("As"), Card.new("Ks"), Card.new("Qs"))
    board2 = (Card.new("Ah"), Card.new("Kh"), Card.new("Qh"))

    # Canonicalize both
    cb1, sm1 = canonicalize_board(board1)
    cb2, sm2 = canonicalize_board(board2)

    print(f"\nBoard 1: {' '.join(str(c) for c in board1)}")
    print(f"  Canonical: {cb1}")
    print(f"Board 2: {' '.join(str(c) for c in board2)}")
    print(f"  Canonical: {cb2}")

    # Check they're the same
    if cb1 == cb2:
        print("\n  PASS: Isomorphic boards have same canonical form!")
    else:
        print("\n  FAIL: Isomorphic boards have different canonical forms!")
        return False

    # Now test isomorphic hands
    hand1 = (Card.new("Js"), Card.new("Ts"))  # JTs in spades
    hand2 = (Card.new("Jh"), Card.new("Th"))  # JTh in hearts

    ch1 = canonicalize_hand(hand1, sm1)
    ch2 = canonicalize_hand(hand2, sm2)

    print(f"\nHand 1: {hand1[0]} {hand1[1]} (on board 1)")
    print(f"  Canonical: {ch1}")
    print(f"Hand 2: {hand2[0]} {hand2[1]} (on board 2)")
    print(f"  Canonical: {ch2}")

    if ch1 == ch2:
        print("\n  PASS: Isomorphic hands have same canonical form!")
        return True
    else:
        print("\n  FAIL: Isomorphic hands have different canonical forms!")
        return False


if __name__ == "__main__":
    ok = True

    ok = test_isomorphic_hands_same_bucket() and ok
    ok = test_bucket_assignment() and ok
    ok = test_mini_precompute() and ok

    print("\n" + "=" * 60)
    if ok:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
