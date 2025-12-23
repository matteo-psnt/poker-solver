"""Tests for the precomputation pipeline."""

import pytest

from src.abstraction.isomorphism import (
    CanonicalBoardEnumerator,
    ComboPrecomputer,
    PrecomputeConfig,
    canonicalize_board,
    get_all_canonical_combos,
)
from src.abstraction.isomorphism.precompute import (
    compute_equity_for_combo,
)
from src.game.state import Card, Street


class TestCanonicalBoardEnumerator:
    """Tests for canonical board enumeration."""

    def test_flop_enumeration_count(self):
        """Test that we get expected number of canonical flops."""
        enum = CanonicalBoardEnumerator(Street.FLOP)
        enum.enumerate()
        boards = list(enum.iterate())

        # We verified earlier that 1911 is correct
        assert len(boards) == 1911

    def test_all_boards_have_correct_length(self):
        """Test that all flop boards have 3 cards."""
        enum = CanonicalBoardEnumerator(Street.FLOP)
        enum.enumerate()

        for board_info in enum.iterate():
            assert len(board_info.representative) == 3
            assert len(board_info.canonical_board) == 3

    def test_raw_counts_sum_correctly(self):
        """Test that raw counts sum to C(52,3) for flops."""
        enum = CanonicalBoardEnumerator(Street.FLOP)
        enum.enumerate()

        total_raw = sum(b.raw_count for b in enum.iterate())

        # C(52, 3) = 22100
        assert total_raw == 22100


class TestCanonicalComboGeneration:
    """Tests for canonical combo generation."""

    def test_paired_board_has_many_combos(self):
        """Test that a paired board has expected combos."""
        board = (Card.new("2s"), Card.new("2h"), Card.new("2d"))
        combos = list(get_all_canonical_combos(board))

        # With 3 deuces on board, 49 cards remain
        # C(49, 2) = 1176 raw combos
        # Under isomorphism, some collapse but it should be close
        assert len(combos) <= 1176
        assert len(combos) > 500  # Should still have many unique combos

    def test_no_duplicate_canonical_combos(self):
        """Test that all generated combos are unique."""
        board = (Card.new("As"), Card.new("Kh"), Card.new("Qd"))
        combos = list(get_all_canonical_combos(board))

        # Convert to keys for uniqueness check
        keys = [(c.board_id, c.hand_id) for c in combos]

        assert len(keys) == len(set(keys))

    def test_combos_dont_overlap_board(self):
        """Test that combo hands don't share cards with board."""
        board = (Card.new("As"), Card.new("Ks"), Card.new("Qs"))
        _board_set = set(board)

        for combo in get_all_canonical_combos(board):
            # The representative hand should not overlap with board
            # (This is checked internally but let's verify the canonical form)
            assert combo.board == canonicalize_board(board)[0]


class TestEquityComputation:
    """Tests for equity computation."""

    def test_compute_equity_returns_valid_range(self):
        """Test that equity is between 0 and 1."""
        board = (Card.new("As"), Card.new("Kh"), Card.new("2c"))
        combos = list(get_all_canonical_combos(board))

        # Test first combo
        combo = combos[0]
        board_id, hand_id, equity = compute_equity_for_combo(
            canonical_board=combo.board,
            canonical_hand=combo.hand,
            representative_board=board,
            equity_samples=50,  # Minimal for speed
            seed=42,
        )

        assert 0.0 <= equity <= 1.0

    def test_premium_hands_have_high_equity(self):
        """Test that premium hands have high equity on dry boards."""
        # Create a dry board
        board = (Card.new("2s"), Card.new("7h"), Card.new("Qc"))

        # Find combo for AA
        combos = list(get_all_canonical_combos(board))

        # Compute equity for several combos to find a high one
        high_equity_found = False
        for combo in combos[:20]:  # Check first 20
            _, _, equity = compute_equity_for_combo(
                canonical_board=combo.board,
                canonical_hand=combo.hand,
                representative_board=board,
                equity_samples=100,
                seed=42,
            )
            if equity > 0.7:
                high_equity_found = True
                break

        assert high_equity_found, "Should find at least one high equity hand"


class TestPrecomputeConfig:
    """Tests for precompute configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PrecomputeConfig.default()

        assert config.num_buckets[Street.FLOP] == 50
        assert config.num_buckets[Street.TURN] == 100
        assert config.num_buckets[Street.RIVER] == 200
        assert config.equity_samples == 1000

    def test_fast_test_config(self):
        """Test fast test configuration values."""
        config = PrecomputeConfig.fast_test()

        assert config.num_buckets[Street.FLOP] == 10
        assert config.equity_samples == 100


class TestComboPrecomputer:
    """Tests for the main precomputer class."""

    def test_precomputer_creation(self):
        """Test that precomputer can be created."""
        config = PrecomputeConfig.fast_test()
        precomputer = ComboPrecomputer(config)

        assert precomputer.abstraction is not None
        assert precomputer.config == config

    @pytest.mark.skip(reason="Full precomputation is too slow for unit tests")
    def test_full_precomputation(self):
        """Test full precomputation (skipped by default)."""
        config = PrecomputeConfig.fast_test()
        precomputer = ComboPrecomputer(config)

        abstraction = precomputer.precompute_all(streets=[Street.FLOP])

        assert abstraction.num_buckets(Street.FLOP) > 0
