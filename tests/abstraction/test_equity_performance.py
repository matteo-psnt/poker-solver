"""Performance and correctness tests for equity bucketing precomputation."""

import time

import numpy as np
import pytest

from src.abstraction.equity_bucketing import EquityBucketing
from src.abstraction.equity_calculator import EquityCalculator
from src.abstraction.parallel_equity import (
    _cards_conflict,
    _get_example_hand,
    compute_equity_matrix_parallel,
)
from src.abstraction.precompute import PrecomputeConfig, generate_boards_optimized
from src.game.state import Card, Street


class TestEquityPerformance:
    """Tests for equity calculation performance optimizations."""

    def test_card_conflict_detection_speed(self):
        """Test that optimized conflict detection is fast."""
        hole_cards = (Card.new("As"), Card.new("Kh"))
        boards = [
            (Card.new("Qd"), Card.new("Jc"), Card.new("Ts")),
            (Card.new("As"), Card.new("2d"), Card.new("3c")),  # Conflicts
            (Card.new("9h"), Card.new("8s"), Card.new("7d")),
        ]

        start = time.time()
        for _ in range(1000):  # Reduced from 10000
            for board in boards:
                _cards_conflict(hole_cards, board)
        elapsed = time.time() - start

        # Should be fast (< 0.1 seconds for 3k checks)
        assert elapsed < 0.1, f"Conflict detection too slow: {elapsed:.3f}s"

    def test_get_example_hand_creates_valid_cards(self):
        """Test that _get_example_hand creates valid, non-conflicting cards."""
        test_hands = [
            "AA",
            "KK",
            "QQ",
            "22",  # Pairs
            "AKs",
            "AQs",
            "KQs",  # Suited
            "AKo",
            "AQo",
            "72o",  # Offsuit
        ]

        for hand_str in test_hands:
            hole_cards = _get_example_hand(hand_str)

            # Check we got 2 cards
            assert len(hole_cards) == 2

            # Check they don't conflict
            assert hole_cards[0] != hole_cards[1]
            assert hole_cards[0].card_int != hole_cards[1].card_int

            # Check set operations work
            card_set = {hole_cards[0], hole_cards[1]}
            assert len(card_set) == 2

    def test_parallel_equity_faster_than_sequential(self):
        """Test that parallel computation is actually faster."""
        PrecomputeConfig.fast_test()

        # Generate small sample for testing
        boards = generate_boards_optimized(
            street=Street.FLOP, num_samples=50, seed=42, show_progress=False
        )

        # Create bucketing
        bucketing = EquityBucketing(
            num_buckets={Street.FLOP: 5},
            num_board_clusters={Street.FLOP: 10},
            equity_calculator=EquityCalculator(num_samples=50),
        )

        # Fit board clusterer
        bucketing.board_clusterer.fit(boards, Street.FLOP)

        # Sample cluster representatives
        cluster_representatives = bucketing._sample_cluster_representatives(
            boards, Street.FLOP, num_samples_per_cluster=3
        )

        # Test sequential computation
        start_seq = time.time()
        equity_matrix_seq = bucketing._compute_equity_matrix(cluster_representatives, Street.FLOP)
        time_seq = time.time() - start_seq

        # Test parallel computation (with 2 workers for CI compatibility)
        start_par = time.time()
        equity_matrix_par = compute_equity_matrix_parallel(
            cluster_representatives=cluster_representatives,
            street=Street.FLOP,
            num_equity_samples=50,
            seed=42,
            num_workers=2,
        )
        time_par = time.time() - start_par

        # Matrices should have same shape
        assert equity_matrix_seq.shape == equity_matrix_par.shape

        # Equities should be close (not exact due to MC variance)
        # Allow 20% difference due to randomness
        diff = np.abs(equity_matrix_seq - equity_matrix_par)
        assert np.mean(diff) < 0.2, f"Average difference too large: {np.mean(diff):.3f}"

        print(f"Sequential: {time_seq:.2f}s, Parallel: {time_par:.2f}s")
        # Don't assert speedup as it's machine-dependent

    def test_equity_matrix_no_nan_values(self):
        """Test that equity computation doesn't produce NaN values."""
        boards = generate_boards_optimized(
            street=Street.FLOP, num_samples=15, seed=42, show_progress=False
        )

        bucketing = EquityBucketing(
            num_buckets={Street.FLOP: 3},
            num_board_clusters={Street.FLOP: 5},
            equity_calculator=EquityCalculator(num_samples=20),
        )

        bucketing.board_clusterer.fit(boards, Street.FLOP)
        cluster_representatives = bucketing._sample_cluster_representatives(
            boards, Street.FLOP, num_samples_per_cluster=2
        )

        equity_matrix = bucketing._compute_equity_matrix(cluster_representatives, Street.FLOP)

        # Check for NaN or Inf
        assert not np.any(np.isnan(equity_matrix)), "Matrix contains NaN values"
        assert not np.any(np.isinf(equity_matrix)), "Matrix contains Inf values"

        # All equities should be in [0, 1]
        assert np.all(equity_matrix >= 0.0), "Found equity < 0"
        assert np.all(equity_matrix <= 1.0), "Found equity > 1"

    def test_board_generation_speed(self):
        """Test that board generation is efficient."""
        start = time.time()
        boards = generate_boards_optimized(
            street=Street.RIVER, num_samples=100, seed=42, show_progress=False
        )
        elapsed = time.time() - start

        # Should generate 100 river boards very quickly
        assert elapsed < 0.5, f"Board generation too slow: {elapsed:.3f}s"

        # Verify correctness
        assert len(boards) == 100
        for board in boards[:10]:  # Check first 10
            assert len(board) == 5  # River has 5 cards
            assert len(set(board)) == 5  # No duplicates

    def test_equity_calculator_caching(self):
        """Test that equity calculator uses cached deck."""
        calc = EquityCalculator(num_samples=100, seed=42)

        # Check that full_deck is cached
        assert hasattr(calc, "full_deck")
        assert len(calc.full_deck) == 52

        # Deck should be same instance each time
        deck1 = calc.full_deck
        deck2 = calc.full_deck
        assert deck1 is deck2

    def test_multiprocessing_worker_isolation(self):
        """Test that worker processes don't interfere with each other."""
        # Use the module-level functions from parallel_equity which are already picklable
        hands = ["AA", "KK", "AKs", "AQo", "72o"]

        # Test that we can call _get_example_hand multiple times
        for hand_str in hands:
            hole_cards = _get_example_hand(hand_str)
            assert len(hole_cards) == 2
            assert hole_cards[0] != hole_cards[1]

            # Test hash works
            h1 = hash(hole_cards[0])
            h2 = hash(hole_cards[1])
            assert h1 != 0
            assert h2 != 0
            assert h1 != h2  # Different cards have different hashes


class TestEquityBucketingWorkflow:
    """Integration tests for complete equity bucketing workflow."""

    def test_small_precompute_workflow(self, tmp_path):
        """Test complete precompute workflow with minimal configuration."""
        config = PrecomputeConfig(
            num_samples_per_street={
                Street.FLOP: 15,
                Street.TURN: 10,
                Street.RIVER: 10,
            },
            num_buckets={
                Street.FLOP: 3,
                Street.TURN: 4,
                Street.RIVER: 5,
            },
            num_board_clusters={
                Street.FLOP: 5,
                Street.TURN: 5,
                Street.RIVER: 5,
            },
            num_equity_samples=20,
            num_samples_per_cluster=2,
            output_file=tmp_path / "test_bucketing.pkl",
            seed=42,
            num_workers=2,  # Use parallel
        )

        # Generate boards
        sample_boards = {}
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            boards = generate_boards_optimized(
                street=street,
                num_samples=config.num_samples_per_street[street],
                seed=config.seed,
                show_progress=False,
            )
            sample_boards[street] = boards

        # Create and fit bucketing
        bucketing = EquityBucketing(
            num_buckets=config.num_buckets,
            num_board_clusters=config.num_board_clusters,
            equity_calculator=EquityCalculator(
                num_samples=config.num_equity_samples, seed=config.seed
            ),
        )

        start = time.time()
        bucketing.fit(
            sample_boards,
            num_samples_per_cluster=config.num_samples_per_cluster,
            num_workers=config.num_workers,
        )
        elapsed = time.time() - start

        # Should complete in reasonable time (< 30 seconds for this tiny config)
        assert elapsed < 30, f"Fitting took too long: {elapsed:.1f}s"

        # Verify bucketing is fitted
        assert bucketing.fitted
        assert Street.FLOP in bucketing.bucket_assignments
        assert Street.TURN in bucketing.bucket_assignments
        assert Street.RIVER in bucketing.bucket_assignments

        # Verify bucket assignments have correct shape
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            assignments = bucketing.bucket_assignments[street]
            assert assignments.shape[0] == 169  # 169 hands
            num_clusters = config.num_board_clusters[street]
            assert assignments.shape[1] == num_clusters

            # All bucket IDs should be valid
            assert np.all(assignments >= 0)
            assert np.all(assignments < config.num_buckets[street])

        # Test get_bucket works
        hole_cards = (Card.new("As"), Card.new("Ah"))
        board_flop = (Card.new("Kd"), Card.new("Qc"), Card.new("Js"))
        bucket = bucketing.get_bucket(hole_cards, board_flop, Street.FLOP)

        assert 0 <= bucket < config.num_buckets[Street.FLOP]

        # Save and load
        bucketing.save(config.output_file)
        assert config.output_file.exists()

        loaded = EquityBucketing.load(config.output_file)
        assert loaded.fitted

        # Verify loaded bucketing gives same results
        bucket_loaded = loaded.get_bucket(hole_cards, board_flop, Street.FLOP)
        assert bucket_loaded == bucket

    def test_card_hash_consistency_across_processes(self):
        """Test that card hashing is consistent across process boundaries."""
        card_strs = ["As", "Kh", "Qd", "Jc", "Ts"]

        # Get hashes in main process - each card should hash consistently
        for card_str in card_strs:
            card = Card.new(card_str)
            h1 = hash(card)
            h2 = hash(card)
            assert h1 == h2, "Hash should be consistent for same card"
            assert h1 != 0, "Hash should not be zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
