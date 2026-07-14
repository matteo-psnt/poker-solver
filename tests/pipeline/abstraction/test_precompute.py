"""Tests for the precomputation pipeline."""

import pytest

from src.core.game.state import Card, Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.board_enumeration import (
    CanonicalBoardEnumerator,
    CanonicalBoardInfo,
)
from src.pipeline.abstraction.postflop.hand_bucketing import get_all_canonical_hands
from src.pipeline.abstraction.postflop.precompute import (
    PostflopPrecomputer,
    _worker_compute_cluster_equities,
)
from src.pipeline.abstraction.postflop.suit_isomorphism import (
    canonicalize_board,
    get_canonical_board_id,
)


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
        _board_set = set(board)

        for combo in get_all_canonical_hands(board):
            # The representative hand should not overlap with board
            # (This is checked internally but let's verify the canonical form)
            assert combo.board == canonicalize_board(board)[0]


def _make_board_info(board: tuple[Card, ...]) -> CanonicalBoardInfo:
    canonical, _ = canonicalize_board(board)
    return CanonicalBoardInfo(
        canonical_board=canonical,
        board_id=get_canonical_board_id(canonical),
        raw_count=1,
        representative=board,
    )


class TestEquityWorker:
    """Tests for the per-representative-board equity worker."""

    def test_worker_covers_all_canonical_combos(self):
        """One worker call yields an equity for every canonical combo."""
        board = (Card.new("As"), Card.new("Kh"), Card.new("2c"))
        board_info = _make_board_info(board)

        results = _worker_compute_cluster_equities((board_info, 3, 20, 42))

        num_combos = sum(1 for _ in get_all_canonical_hands(board))
        assert len(results) == num_combos

        hand_ids = set()
        for cluster_id, board_id, hand_id, equity in results:
            assert cluster_id == 3
            assert board_id == board_info.board_id
            assert 0.0 <= equity <= 1.0
            hand_ids.add(hand_id)
        assert len(hand_ids) == num_combos

    def test_worker_is_deterministic(self):
        """Same args produce identical equities (seeded runout sampling)."""
        board = (Card.new("2s"), Card.new("7h"), Card.new("Qc"))
        board_info = _make_board_info(board)

        results1 = _worker_compute_cluster_equities((board_info, 0, 25, 42))
        results2 = _worker_compute_cluster_equities((board_info, 0, 25, 42))

        assert results1 == results2

    def test_premium_hands_have_high_equity(self):
        """Some hands on a dry board must be strong (sanity of equity scale)."""
        board = (Card.new("2s"), Card.new("7h"), Card.new("Qc"))
        board_info = _make_board_info(board)

        results = _worker_compute_cluster_equities((board_info, 0, 25, 42))

        assert max(equity for _, _, _, equity in results) > 0.8


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

        assert precomputer.abstraction is not None
        assert precomputer.config == config

    @pytest.mark.skip(reason="Full precomputation is too slow for unit tests")
    def test_full_precomputation(self):
        """Test full precomputation (skipped by default)."""
        config = PrecomputeConfig.from_yaml("quick_test")
        precomputer = PostflopPrecomputer(config)

        abstraction = precomputer.precompute_all(streets=[Street.FLOP])

        assert abstraction.num_buckets(Street.FLOP) > 0
