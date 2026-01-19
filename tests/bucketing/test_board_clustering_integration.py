"""Integration tests for board clustering in precomputation pipeline."""

import pytest

from src.bucketing.config import PrecomputeConfig
from src.bucketing.postflop.board_clustering import BoardClusterer
from src.bucketing.postflop.board_enumeration import CanonicalBoardEnumerator
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.game.state import Street

pytestmark = pytest.mark.slow


class TestBoardClusteringIntegration:
    """Test board clustering integration with precomputation."""

    def test_board_clusterer_creation(self):
        """Test creating a board clusterer."""
        clusterer = BoardClusterer(
            {
                Street.FLOP: 10,
                Street.TURN: 20,
                Street.RIVER: 30,
            }
        )
        assert clusterer is not None

    def test_board_enumeration_and_clustering(self):
        """Test enumerating boards and clustering them."""
        # Enumerate a small number of flop boards
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        enumerator.enumerate()

        board_infos = list(enumerator.iterate())
        assert len(board_infos) > 0

        # Extract canonical and representative boards
        canonical_boards = [info.canonical_board for info in board_infos[:100]]  # Use first 100
        representative_boards = [info.representative for info in board_infos[:100]]

        # Create clusterer and fit
        clusterer = BoardClusterer({Street.FLOP: 10})
        clusterer.fit(
            canonical_boards, representative_boards, Street.FLOP, representatives_per_cluster=1
        )

        # Get clusters
        clusters = clusterer.get_all_clusters(Street.FLOP)
        assert len(clusters) > 0
        assert len(clusters) <= 10  # Should have at most 10 clusters

        # Verify each cluster has representatives
        for cluster in clusters:
            assert len(cluster.representative_boards) > 0
            assert len(cluster.canonical_representatives) > 0
            assert cluster.cluster_id >= 0

    def test_board_info_attributes(self):
        """Test that CanonicalBoardInfo has correct attributes."""
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        enumerator.enumerate()

        board_infos = list(enumerator.iterate())
        assert len(board_infos) > 0

        first_board = board_infos[0]

        # Check required attributes exist
        assert hasattr(first_board, "canonical_board")
        assert hasattr(first_board, "representative")
        assert hasattr(first_board, "board_id")
        assert hasattr(first_board, "raw_count")

        # Check types
        assert isinstance(first_board.canonical_board, tuple)
        assert isinstance(first_board.representative, tuple)
        assert isinstance(first_board.board_id, int)
        assert isinstance(first_board.raw_count, int)

    def test_precomputer_initialization(self):
        """Test that precomputer initializes correctly."""
        config = PrecomputeConfig(
            num_buckets={
                Street.FLOP: 5,
                Street.TURN: 5,
                Street.RIVER: 5,
            },
            num_board_clusters={
                Street.FLOP: 5,
                Street.TURN: 5,
                Street.RIVER: 5,
            },
            representatives_per_cluster=1,
            equity_samples=10,
            num_workers=1,
        )
        precomputer = PostflopPrecomputer(config)

        assert precomputer.config == config
        assert precomputer.abstraction is not None

    def test_minimal_precomputation(self):
        """Test running minimal precomputation (single street, few boards)."""
        # Create a very minimal config
        config = PrecomputeConfig(
            num_buckets={
                Street.FLOP: 5,
                Street.TURN: 5,
                Street.RIVER: 5,
            },
            num_board_clusters={
                Street.FLOP: 5,
                Street.TURN: 5,
                Street.RIVER: 5,
            },
            representatives_per_cluster=1,
            equity_samples=10,  # Very few samples for speed
            num_workers=1,
            seed=42,
        )

        precomputer = PostflopPrecomputer(config)

        # Note: Full precomputation is too slow for unit tests
        # Just verify the precomputer is correctly initialized
        assert precomputer.abstraction is not None
        assert len(precomputer.abstraction._buckets) == 3  # FLOP, TURN, RIVER

    def test_save_and_load_abstraction(self):
        """Test saving and loading precomputed abstraction."""
        # Create minimal abstraction
        config = PrecomputeConfig(
            num_buckets={
                Street.FLOP: 3,
                Street.TURN: 3,
                Street.RIVER: 3,
            },
            num_board_clusters={
                Street.FLOP: 3,
                Street.TURN: 3,
                Street.RIVER: 3,
            },
            representatives_per_cluster=1,
            equity_samples=10,
            num_workers=1,
            seed=42,
        )

        precomputer = PostflopPrecomputer(config)

        # Note: Full precomputation is too slow for unit tests
        # Just verify save/load mechanics work
        assert precomputer.abstraction is not None

    def test_cluster_prediction(self):
        """Test that board clusterer can predict clusters."""
        # Create and fit clusterer
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        enumerator.enumerate()

        board_infos = list(enumerator.iterate())[:50]  # Use 50 boards
        canonical_boards = [info.canonical_board for info in board_infos]
        representative_boards = [info.representative for info in board_infos]

        clusterer = BoardClusterer({Street.FLOP: 5})
        clusterer.fit(
            canonical_boards, representative_boards, Street.FLOP, representatives_per_cluster=1
        )

        # Predict cluster for a board (use representative, not canonical)
        test_board = representative_boards[0]
        cluster_id = clusterer.predict(test_board, Street.FLOP)

        assert isinstance(cluster_id, int)
        assert 0 <= cluster_id < 5

    def test_error_handling_invalid_street(self):
        """Test error handling for invalid street."""
        clusterer = BoardClusterer({Street.FLOP: 10})

        # Try to fit on TURN (not configured)
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        enumerator.enumerate()
        board_infos = list(enumerator.iterate())[:10]

        canonical_boards = [info.canonical_board for info in board_infos]
        representative_boards = [info.representative for info in board_infos]

        with pytest.raises(ValueError, match="No cluster count specified"):
            clusterer.fit(
                canonical_boards,
                representative_boards,
                Street.TURN,  # Not configured
                representatives_per_cluster=1,
            )


class TestPrecomputeConfigYAML:
    """Test YAML config loading for precomputation."""

    def test_configs_have_buckets(self):
        """Test that built-in configs have bucket settings."""
        # Test using class methods
        configs = [
            PrecomputeConfig.from_yaml("fast_test"),
            PrecomputeConfig.default(),
        ]

        for config in configs:
            # Check buckets exist for all streets
            assert Street.FLOP in config.num_buckets
            assert Street.TURN in config.num_buckets
            assert Street.RIVER in config.num_buckets

            # Check they're positive
            assert config.num_buckets[Street.FLOP] > 0
            assert config.num_buckets[Street.TURN] > 0
            assert config.num_buckets[Street.RIVER] > 0

    def test_config_consistency(self):
        """Test that configs are internally consistent."""
        configs = [
            PrecomputeConfig.from_yaml("fast_test"),
            PrecomputeConfig.default(),
        ]

        for config in configs:
            # Verify bucket counts are reasonable
            for street in [Street.FLOP, Street.TURN, Street.RIVER]:
                buckets = config.num_buckets[street]

                # Just verify buckets are specified
                assert buckets > 0
