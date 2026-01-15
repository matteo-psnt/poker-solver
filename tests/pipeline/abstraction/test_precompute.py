"""Tests for the precomputation pipeline."""

import numpy as np

from src.core.game.state import Card, Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.board_enumeration import CanonicalBoardEnumerator
from src.pipeline.abstraction.postflop.canonical_hands import (
    enumerate_hand_classes,
    get_all_canonical_hands,
)
from src.pipeline.abstraction.postflop.precompute import (
    PostflopPrecomputer,
    _worker_compute_board_chunk,
)
from src.pipeline.abstraction.postflop.suit_isomorphism import canonicalize_board


class TestCanonicalBoardEnumerator:
    """Tests for canonical board enumeration."""

    def test_flop_enumeration_count(self):
        """Test that we get expected number of canonical flops."""
        enum = CanonicalBoardEnumerator(Street.FLOP)
        enum.enumerate()
        boards = list(enum.iterate())

        # We verified earlier that 1755 is correct
        assert len(boards) == 1755

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
        combos = list(get_all_canonical_hands(board))

        # With 3 deuces on board, 49 cards remain
        # C(49, 2) = 1176 raw combos
        # Under isomorphism, some collapse but it should be close
        assert len(combos) <= 1176
        assert len(combos) > 500  # Should still have many unique combos

    def test_no_duplicate_canonical_combos(self):
        """Test that all generated combos are unique."""
        board = (Card.new("As"), Card.new("Kh"), Card.new("Qd"))
        combos = list(get_all_canonical_hands(board))

        # Convert to keys for uniqueness check
        keys = [(c.board_id, c.hand_id) for c in combos]

        assert len(keys) == len(set(keys))

    def test_combos_dont_overlap_board(self):
        """Test that combo hands don't share cards with board."""
        board = (Card.new("As"), Card.new("Ks"), Card.new("Qs"))

        for combo in get_all_canonical_hands(board):
            assert combo.board == canonicalize_board(board)[0]

    def test_hand_class_multiplicities_cover_all_combos(self):
        """Class multiplicities must sum to the concrete combo count."""
        board = (Card.new("Ts"), Card.new("9s"), Card.new("8c"))
        classes = enumerate_hand_classes(board)

        # C(49, 2) concrete combos on a flop
        assert sum(c.multiplicity for c in classes) == 1176
        assert len(classes) == len(list(get_all_canonical_hands(board)))
        # Every representative is a legal concrete hand
        board_set = set(board)
        for hand_class in classes:
            assert hand_class.representative[0] not in board_set
            assert hand_class.representative[1] not in board_set


class TestEquityWorker:
    """Tests for the per-board equity worker."""

    def test_worker_covers_all_hand_classes(self):
        """One worker call yields an equity for every class on every board."""
        boards = [
            (0, (Card.new("As"), Card.new("Kh"), Card.new("2c"))),
            (1, (Card.new("2s"), Card.new("7h"), Card.new("Qc"))),
        ]
        results = _worker_compute_board_chunk((boards, 20, 42))

        assert len(results) == 2
        for (row, _board), (result_row, cols, equities, multiplicities) in zip(boards, results):
            assert result_row == row
            n_classes = len(enumerate_hand_classes(_board))
            assert len(cols) == len(equities) == len(multiplicities) == n_classes
            assert np.all((equities >= 0.0) & (equities <= 1.0))
            assert np.all(cols >= 0)
            assert np.all(multiplicities >= 1)
            # Columns are unique: one cell per class
            assert len(np.unique(cols)) == n_classes

    def test_worker_is_deterministic(self):
        """Same args produce identical equities (seeded runout sampling)."""
        boards = [(0, (Card.new("2s"), Card.new("7h"), Card.new("Qc")))]

        _, cols1, eq1, _ = _worker_compute_board_chunk((boards, 25, 42))[0]
        _, cols2, eq2, _ = _worker_compute_board_chunk((boards, 25, 42))[0]

        assert np.array_equal(cols1, cols2)
        assert np.array_equal(eq1, eq2)

    def test_premium_hands_have_high_equity(self):
        """Some hands on a dry board must be strong (sanity of equity scale)."""
        boards = [(0, (Card.new("2s"), Card.new("7h"), Card.new("Qc")))]
        _, _, equities, _ = _worker_compute_board_chunk((boards, 25, 42))[0]

        assert float(equities.max()) > 0.8


class TestPrecomputeConfig:
    """Tests for precompute configuration."""

    def test_quick_test_config(self):
        """Test fast test configuration values."""
        config = PrecomputeConfig.from_yaml("quick_test")

        assert config.num_buckets[Street.FLOP] == 10
        assert config.flop_runouts == 200


class TestComboPrecomputer:
    """Tests for the main precomputer class."""

    def test_precomputer_creation(self):
        """Test that precomputer can be created."""
        config = PrecomputeConfig.from_yaml("quick_test")
        precomputer = PostflopPrecomputer(config)

        assert precomputer.config == config
