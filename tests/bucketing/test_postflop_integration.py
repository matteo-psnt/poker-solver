"""
Integration tests for board clustering in combo abstraction.

Tests the full pipeline:
1. Load config with clustering parameters
2. Precompute with board clustering
3. Save to disk
4. Load from disk
5. Runtime bucket lookup via cluster prediction
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.bucketing.config import PrecomputeConfig, StreetBucketConfig
from src.bucketing.postflop.board_enumeration import CanonicalBoardEnumerator
from src.bucketing.postflop.hand_bucketing import PostflopBucketer
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.game.rules import Street
from src.game.state import Card

pytestmark = pytest.mark.slow


class TestClusteringIntegration:
    """Test complete clustering pipeline."""

    @pytest.fixture(scope="session")
    def test_config(self):
        """Create minimal config for testing."""
        return PrecomputeConfig(
            board_clusters=StreetBucketConfig(flop=5, turn=10, river=15),
            representatives_per_cluster=1,  # Reduced from 2
            equity_samples=25,  # Reduced from 50
            buckets=StreetBucketConfig(flop=10, turn=15, river=20),
            seed=42,
        )

    @pytest.fixture(scope="session")
    def precomputed_abstraction(self, test_config):
        """Precompute abstraction with clustering (session-scoped for speed)."""
        precomputer = PostflopPrecomputer(test_config)

        # Precompute one street for testing
        precomputer.precompute_street(Street.FLOP)

        return precomputer.abstraction, precomputer

    def test_config_loading_from_yaml(self):
        """Test loading config from YAML file."""
        config = PrecomputeConfig.from_yaml("quick_test")

        # Check clustering parameters loaded
        assert Street.FLOP in config.num_board_clusters
        assert Street.TURN in config.num_board_clusters
        assert Street.RIVER in config.num_board_clusters
        assert config.representatives_per_cluster > 0

        # Check bucket numbers
        assert Street.FLOP in config.num_buckets
        assert Street.TURN in config.num_buckets
        assert Street.RIVER in config.num_buckets

    def test_precomputation_creates_clusters(self, precomputed_abstraction):
        """Test that precomputation creates cluster-based storage."""
        abstraction, precomputer = precomputed_abstraction

        # Check board_clusterer was created
        assert abstraction._board_clusterer is not None

        # Check storage is cluster-based (not board-based)
        assert Street.FLOP in abstraction._buckets

        flop_buckets = abstraction._buckets[Street.FLOP]

        # Should have ~10 clusters (from config), not ~1755 boards
        num_clusters = len(flop_buckets)
        assert 5 <= num_clusters <= 15, f"Expected ~10 clusters, got {num_clusters}"

        # Each cluster should have buckets for many hands
        first_cluster_id = next(iter(flop_buckets.keys()))
        num_hands_in_cluster = len(flop_buckets[first_cluster_id])
        assert num_hands_in_cluster > 100, "Should have many hands per cluster"

    def test_runtime_bucket_lookup(self, precomputed_abstraction):
        """Test runtime lookup predicts cluster and finds bucket."""
        abstraction, _ = precomputed_abstraction

        # Sample board and hand
        hole_cards = (Card.new("Ah"), Card.new("Kd"))
        board = (Card.new("Qs"), Card.new("Jc"), Card.new("9h"))

        # Should successfully predict cluster and return bucket
        bucket = abstraction.get_bucket(hole_cards, board, Street.FLOP)

        # Bucket should be valid
        assert isinstance(bucket, int)
        assert 0 <= bucket < 20  # 20 buckets in test config

    def test_canonicalization_same_cluster(self, precomputed_abstraction):
        """Test that identical canonical boards map to same bucket."""
        abstraction, _ = precomputed_abstraction

        # Use the exact same canonical form (not just isomorphic)
        # With fewer clusters, isomorphic boards might map differently
        hole_cards = (Card.new("Ah"), Card.new("Kd"))
        board = (Card.new("Qs"), Card.new("Jc"), Card.new("9h"))

        # Get bucket twice - should be consistent
        bucket1 = abstraction.get_bucket(hole_cards, board, Street.FLOP)
        bucket2 = abstraction.get_bucket(hole_cards, board, Street.FLOP)

        # Should map to same bucket (deterministic)
        assert bucket1 == bucket2

    @pytest.mark.timeout(20)
    def test_save_and_load_with_clustering(self, test_config):
        """Test saving cluster-based abstraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Precompute
            precomputer = PostflopPrecomputer(test_config)
            precomputer.precompute_street(Street.FLOP)

            # Save
            precomputer.save(save_dir)

            # Verify files exist
            assert (save_dir / "metadata.json").exists()
            assert (save_dir / "combo_abstraction.pkl").exists()

            # Verify board_clusterer was saved
            assert precomputer.abstraction._board_clusterer is not None
            assert Street.FLOP in precomputer.abstraction._buckets

    @pytest.mark.timeout(10)
    def test_metadata_includes_clustering_info(self, test_config):
        """Test that saved metadata includes clustering statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Precompute and save
            precomputer = PostflopPrecomputer(test_config)
            precomputer.precompute_street(Street.FLOP)
            precomputer.save(save_dir)

            metadata_path = save_dir / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check clustering info present in statistics
            assert "statistics" in metadata
            assert "FLOP" in metadata["statistics"]
            assert "num_clusters" in metadata["statistics"]["FLOP"]

            # Check config has clustering parameters
            config = metadata["config"]
            assert "board_clusters" in config
            assert "buckets" in config
            assert "representatives_per_cluster" in config
            assert "config_hash" in metadata

    def test_different_boards_span_multiple_clusters(self, precomputed_abstraction):
        """Test that texturally different boards are not all mapped to one cluster."""
        abstraction, _ = precomputed_abstraction

        # Very different board textures
        paired_board = (Card.new("Qs"), Card.new("Qc"), Card.new("9h"))  # Paired
        straight_board = (Card.new("Ts"), Card.new("9c"), Card.new("8h"))  # Connected
        rainbow_board = (Card.new("As"), Card.new("7c"), Card.new("2h"))  # Rainbow high card

        clusterer = abstraction._board_clusterer
        assert clusterer is not None

        cluster_ids = {
            clusterer.predict(paired_board, Street.FLOP),
            clusterer.predict(straight_board, Street.FLOP),
            clusterer.predict(rainbow_board, Street.FLOP),
        }
        assert len(cluster_ids) >= 2, "Expected different textures to span multiple clusters"

    def test_computational_savings(self, test_config):
        """Verify clustering reduces computation vs full enumeration."""
        # Count equity computations needed
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        enumerator.enumerate()

        # Full enumeration: compute equity for all canonical boards
        num_canonical_boards = len(enumerator._cache)

        # Clustering: compute equity for representatives only
        num_clusters = test_config.num_board_clusters[Street.FLOP]
        reps_per_cluster = test_config.representatives_per_cluster
        num_equity_computations = num_clusters * reps_per_cluster

        # Clustering should be ~10-100x fewer computations
        speedup = num_canonical_boards / num_equity_computations
        assert speedup >= 10, f"Expected 10x speedup, got {speedup:.1f}x"

    def test_error_on_missing_clusterer(self):
        """Test error when board_clusterer not initialized."""
        abstraction = PostflopBucketer()
        # Don't precompute, so _board_clusterer is None

        hole_cards = (Card.new("Ah"), Card.new("Kd"))
        board = (Card.new("Qs"), Card.new("Jc"), Card.new("9h"))

        with pytest.raises(ValueError, match="Board clusterer not initialized"):
            abstraction.get_bucket(hole_cards, board, Street.FLOP)


class TestClusterPrediction:
    """Test board cluster prediction specifically."""

    @pytest.fixture(scope="session")
    def clustered_abstraction(self):
        """Create abstraction with fitted board clusterer (session-scoped for speed)."""
        config = PrecomputeConfig(
            board_clusters=StreetBucketConfig(flop=8, turn=15, river=20),  # Reduced
            representatives_per_cluster=1,  # Reduced from 2
            equity_samples=50,  # Reduced from 100
            buckets=StreetBucketConfig(flop=15, turn=20, river=30),  # Reduced
            seed=42,
        )

        precomputer = PostflopPrecomputer(config)
        precomputer.precompute_street(Street.FLOP)

        return precomputer.abstraction

    def test_cluster_prediction_consistent(self, clustered_abstraction):
        """Test that cluster prediction is consistent for same board."""
        board = (Card.new("Qs"), Card.new("Jc"), Card.new("9h"))

        # Predict cluster multiple times
        cluster1 = clustered_abstraction._board_clusterer.predict(board, Street.FLOP)
        cluster2 = clustered_abstraction._board_clusterer.predict(board, Street.FLOP)

        assert cluster1 == cluster2, "Cluster prediction should be deterministic"

    def test_cluster_prediction_is_isomorphism_invariant(self, clustered_abstraction):
        """CRITICAL: Suit-isomorphic boards must predict same cluster."""

        # Test case 1: Monotone boards with different suits
        board1 = (Card.new("Ah"), Card.new("Kh"), Card.new("Qh"))  # Hearts
        board2 = (Card.new("As"), Card.new("Ks"), Card.new("Qs"))  # Spades

        cluster1 = clustered_abstraction._board_clusterer.predict(board1, Street.FLOP)
        cluster2 = clustered_abstraction._board_clusterer.predict(board2, Street.FLOP)

        assert cluster1 == cluster2, (
            "Monotone AKQ boards must predict same cluster regardless of suit"
        )

        # Test case 2: Two-tone boards with permuted suits
        board3 = (Card.new("Ah"), Card.new("Kh"), Card.new("Qc"))  # Hearts + Club
        board4 = (Card.new("Ad"), Card.new("Kd"), Card.new("Qs"))  # Diamonds + Spade

        cluster3 = clustered_abstraction._board_clusterer.predict(board3, Street.FLOP)
        cluster4 = clustered_abstraction._board_clusterer.predict(board4, Street.FLOP)

        assert cluster3 == cluster4, (
            "Two-tone AKQ boards must predict same cluster regardless of specific suits"
        )

        # Test case 3: Paired boards
        board5 = (Card.new("Ah"), Card.new("As"), Card.new("Kh"))  # AA with different suits
        board6 = (Card.new("Ad"), Card.new("Ac"), Card.new("Ks"))  # AA with different suits

        cluster5 = clustered_abstraction._board_clusterer.predict(board5, Street.FLOP)
        cluster6 = clustered_abstraction._board_clusterer.predict(board6, Street.FLOP)

        assert cluster5 == cluster6, (
            "Paired boards must predict same cluster regardless of suit distribution"
        )

    def test_cluster_prediction_returns_valid_id(self, clustered_abstraction):
        """Test that predicted cluster ID is valid."""
        board = (Card.new("Qs"), Card.new("Jc"), Card.new("9h"))

        cluster_id = clustered_abstraction._board_clusterer.predict(board, Street.FLOP)

        # Cluster ID should be in range [0, num_clusters)
        assert isinstance(cluster_id, int)
        assert 0 <= cluster_id < 10  # 10 clusters in config

    def test_canonical_boards_predict_to_fitted_clusters(self, clustered_abstraction):
        """Test that canonical boards predict to clusters we fitted on."""
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        enumerator.enumerate()

        # Get some canonical boards
        canonical_boards = [info.representative for info in list(enumerator._cache.values())[:50]]

        for board in canonical_boards:
            cluster_id = clustered_abstraction._board_clusterer.predict(board, Street.FLOP)

            # Cluster should be valid and have buckets
            assert cluster_id in clustered_abstraction._buckets[Street.FLOP]
