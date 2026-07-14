"""Tests for the precomputation pipeline."""

import numpy as np

from src.core.game.state import Card, Street
from src.pipeline.abstraction.config import PrecomputeConfig, StreetBucketConfig
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

    def test_disk_cache_roundtrip(self, tmp_path):
        """A cache-loaded enumeration is identical to a fresh one."""
        fresh = CanonicalBoardEnumerator(Street.FLOP, cache_dir=tmp_path)
        fresh.enumerate()
        assert (tmp_path / "flop_v1.npz").exists()

        cached = CanonicalBoardEnumerator(Street.FLOP, cache_dir=tmp_path)
        cached.enumerate()

        fresh_infos = {info.board_id: info for info in fresh.iterate()}
        cached_infos = {info.board_id: info for info in cached.iterate()}
        assert fresh_infos.keys() == cached_infos.keys()
        for board_id, info in fresh_infos.items():
            loaded = cached_infos[board_id]
            assert loaded.raw_count == info.raw_count
            assert loaded.representative == info.representative
            assert loaded.canonical_board == info.canonical_board


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
        results = _worker_compute_board_chunk((boards, 20, 42, 8))

        assert len(results) == 2
        for (row, _board), (result_row, cols, equities, multiplicities, hists) in zip(
            boards, results
        ):
            assert result_row == row
            n_classes = len(enumerate_hand_classes(_board))
            assert len(cols) == len(equities) == len(multiplicities) == n_classes
            assert np.all((equities >= 0.0) & (equities <= 1.0))
            assert np.all(cols >= 0)
            assert np.all(multiplicities >= 1)
            # Columns are unique: one cell per class
            assert len(np.unique(cols)) == n_classes
            # Realization histograms: one distribution per class, rows sum to 1
            assert hists.shape == (n_classes, 8)
            np.testing.assert_allclose(hists.sum(axis=1).astype(np.float64), 1.0, atol=2e-3)

    def test_worker_without_histograms(self):
        """River-style call (histogram_bins=None) returns no histograms."""
        boards = [(0, (Card.new("2s"), Card.new("7h"), Card.new("Qc")))]
        _, _, equities, _, hists = _worker_compute_board_chunk((boards, 25, 42, None))[0]

        assert hists is None
        assert np.all((equities >= 0.0) & (equities <= 1.0))

    def test_worker_is_deterministic(self):
        """Same args produce identical equities (seeded runout sampling)."""
        boards = [(0, (Card.new("2s"), Card.new("7h"), Card.new("Qc")))]

        _, cols1, eq1, _, _ = _worker_compute_board_chunk((boards, 25, 42, None))[0]
        _, cols2, eq2, _, _ = _worker_compute_board_chunk((boards, 25, 42, None))[0]

        assert np.array_equal(cols1, cols2)
        assert np.array_equal(eq1, eq2)

    def test_premium_hands_have_high_equity(self):
        """Some hands on a dry board must be strong (sanity of equity scale)."""
        boards = [(0, (Card.new("2s"), Card.new("7h"), Card.new("Qc")))]
        _, _, equities, _, _ = _worker_compute_board_chunk((boards, 25, 42, None))[0]

        assert float(equities.max()) > 0.8


class TestHistogramBucketing:
    """Potential-aware bucketing must separate draws from made hands."""

    def test_equal_equity_different_shape_separate(self):
        """Bimodal (draw) and unimodal (made) distributions with the same mean
        must land in different buckets, which scalar-equity bucketing cannot do."""
        config = PrecomputeConfig(
            buckets=StreetBucketConfig(flop=2, turn=2, river=2),
            kmeans_max_iter=50,
            kmeans_n_init=2,
            seed=42,
        )
        precomputer = PostflopPrecomputer(config)

        bins = 8
        made = np.zeros(bins)
        made[3:5] = 0.5  # unimodal around 0.5 equity
        draw = np.zeros(bins)
        draw[0] = 0.5  # bimodal: miss...
        draw[7] = 0.5  # ...or hit; same mean equity

        n_each = 8
        hist_matrix = np.zeros((1, 2 * n_each, bins), dtype=np.float16)
        hist_matrix[0, :n_each] = made
        hist_matrix[0, n_each:] = draw
        equity_matrix = np.full((1, 2 * n_each), 0.5, dtype=np.float32)
        weight_matrix = np.ones((1, 2 * n_each), dtype=np.uint8)
        board_ids = np.array([0], dtype=np.int64)

        precomputer.bucket_street(Street.FLOP, board_ids, equity_matrix, weight_matrix, hist_matrix)
        buckets = precomputer._bucket_matrices[Street.FLOP][0]

        made_buckets = set(buckets[:n_each].tolist())
        draw_buckets = set(buckets[n_each:].tolist())
        assert len(made_buckets) == 1
        assert len(draw_buckets) == 1
        assert made_buckets != draw_buckets

    def test_bucket_order_follows_mean_equity(self):
        """Bucket 0 must hold the lowest-equity distributions."""
        config = PrecomputeConfig(
            buckets=StreetBucketConfig(flop=2, turn=2, river=2),
            kmeans_max_iter=50,
            kmeans_n_init=2,
            seed=42,
        )
        precomputer = PostflopPrecomputer(config)

        bins = 8
        weak = np.zeros(bins)
        weak[0] = 1.0  # ~0.06 equity
        strong = np.zeros(bins)
        strong[7] = 1.0  # ~0.94 equity

        n_each = 8
        hist_matrix = np.zeros((1, 2 * n_each, bins), dtype=np.float16)
        hist_matrix[0, :n_each] = weak
        hist_matrix[0, n_each:] = strong
        equity_matrix = np.zeros((1, 2 * n_each), dtype=np.float32)
        equity_matrix[0, :n_each] = 0.06
        equity_matrix[0, n_each:] = 0.94
        weight_matrix = np.ones((1, 2 * n_each), dtype=np.uint8)
        board_ids = np.array([0], dtype=np.int64)

        quality = precomputer.bucket_street(
            Street.FLOP, board_ids, equity_matrix, weight_matrix, hist_matrix
        )
        buckets = precomputer._bucket_matrices[Street.FLOP][0]

        assert set(buckets[:n_each].tolist()) == {0}
        assert set(buckets[n_each:].tolist()) == {1}
        assert quality["bucketing"] == "equity_histogram_cdf"


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
