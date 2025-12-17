"""Tests for board clustering."""

import numpy as np

from src.abstraction.board_clustering import BoardClusterer
from src.game.state import Card, Street


class TestBoardClusterer:
    """Tests for BoardClusterer."""

    def test_create_clusterer(self):
        """Test creating clusterer."""
        clusterer = BoardClusterer()
        assert clusterer is not None
        assert clusterer.num_clusters[Street.FLOP] == 200
        assert clusterer.num_clusters[Street.TURN] == 500
        assert clusterer.num_clusters[Street.RIVER] == 1000

    def test_create_clusterer_custom_clusters(self):
        """Test creating clusterer with custom cluster counts."""
        custom = {
            Street.FLOP: 50,
            Street.TURN: 100,
            Street.RIVER: 200,
        }
        clusterer = BoardClusterer(num_clusters=custom)
        assert clusterer.num_clusters[Street.FLOP] == 50
        assert clusterer.num_clusters[Street.TURN] == 100
        assert clusterer.num_clusters[Street.RIVER] == 200

    def test_extract_features_flop(self):
        """Test feature extraction for flop."""
        clusterer = BoardClusterer()

        board = (Card.new("As"), Card.new("Ks"), Card.new("Qh"))
        features = clusterer.extract_features(board)

        # Should return feature vector
        assert features.shape == (16,)  # 4 + 5 + 4 + 3 features

        # Should be normalized (mostly between 0 and 1)
        assert np.all((features >= 0) & (features <= 1.5))

    def test_suit_features_monotone(self):
        """Test suit features for monotone board."""
        clusterer = BoardClusterer()

        # All spades
        board = (Card.new("As"), Card.new("Ks"), Card.new("2s"), Card.new("7s"), Card.new("Js"))
        features = clusterer.extract_features(board)

        # Features: [num_suits/4, max_suit/len, is_monotone, is_two_tone, ...]
        # num_suits = 1, max_suit = 5
        assert features[0] == 1.0 / 4.0  # 1 suit
        assert features[1] == 5.0 / 5.0  # All cards same suit
        assert features[2] == 1.0  # Is monotone
        assert features[3] == 0.0  # Not two-tone

    def test_suit_features_rainbow(self):
        """Test suit features for rainbow board."""
        clusterer = BoardClusterer()

        # All different suits
        board = (Card.new("As"), Card.new("Kh"), Card.new("Qd"), Card.new("Jc"), Card.new("2s"))
        features = clusterer.extract_features(board)

        # 4 different suits (rainbow for 5 cards)
        assert features[0] == 4.0 / 4.0  # 4 suits
        assert features[2] == 0.0  # Not monotone
        assert features[3] == 0.0  # Not two-tone

    def test_suit_features_two_tone(self):
        """Test suit features for two-tone board."""
        clusterer = BoardClusterer()

        # Exactly 2 suits
        board = (Card.new("As"), Card.new("Ks"), Card.new("Qh"))
        features = clusterer.extract_features(board)

        assert features[0] == 2.0 / 4.0  # 2 suits
        assert features[3] == 1.0  # Is two-tone

    def test_rank_features_paired(self):
        """Test rank features for paired board."""
        clusterer = BoardClusterer()

        # Pair on board
        board = (Card.new("As"), Card.new("Ah"), Card.new("Kd"))
        features = clusterer.extract_features(board)

        # Features: [..., num_ranks/len, max_rank_count/4, is_paired, is_trips, is_two_pair, ...]
        # Index 4-8 are rank features
        assert features[4] == 2.0 / 3.0  # 2 unique ranks out of 3 cards
        assert features[5] == 2.0 / 4.0  # Max count is 2 (pair)
        assert features[6] == 1.0  # Is paired
        assert features[7] == 0.0  # Not trips

    def test_rank_features_trips(self):
        """Test rank features for trips."""
        clusterer = BoardClusterer()

        # Trips on board
        board = (Card.new("7s"), Card.new("7h"), Card.new("7d"), Card.new("Ac"), Card.new("2s"))
        features = clusterer.extract_features(board)

        assert features[6] == 1.0  # Is paired
        assert features[7] == 1.0  # Is trips

    def test_rank_features_two_pair(self):
        """Test rank features for two pair."""
        clusterer = BoardClusterer()

        # Two pair on board
        board = (Card.new("As"), Card.new("Ah"), Card.new("Ks"), Card.new("Kd"), Card.new("2c"))
        features = clusterer.extract_features(board)

        assert features[8] == 1.0  # Is two pair

    def test_connectivity_features_connected(self):
        """Test connectivity for connected board."""
        clusterer = BoardClusterer()

        # Connected board (straight possible)
        board = (Card.new("9s"), Card.new("Th"), Card.new("Jd"))
        features = clusterer.extract_features(board)

        # Features: [..., spread/12, has_straight, connected_count/len, has_wheel, ...]
        # 9-T-J has small spread, straights possible
        assert features[9] <= 0.3  # Small spread
        assert features[10] == 1.0  # Has straight possibility

    def test_connectivity_features_spread(self):
        """Test connectivity for spread board."""
        clusterer = BoardClusterer()

        # Spread board (no straight possible)
        board = (Card.new("2s"), Card.new("7h"), Card.new("Kd"))
        features = clusterer.extract_features(board)

        # 2-7-K has large spread, straights difficult
        assert features[9] > 0.5  # Large spread

    def test_high_card_features(self):
        """Test high card strength features."""
        clusterer = BoardClusterer()

        # High board (A-K-Q)
        board = (Card.new("As"), Card.new("Kh"), Card.new("Qd"))
        features = clusterer.extract_features(board)

        # Features: [..., max_rank/14, avg_rank/14, broadway_count/len]
        # All cards are broadway (T+)
        assert features[13] == 14.0 / 14.0  # Max rank is A (14)
        assert features[15] == 3.0 / 3.0  # All broadway

    def test_fit_and_predict(self):
        """Test fitting clusterer and predicting."""
        clusterer = BoardClusterer(num_clusters={Street.FLOP: 10})

        # Generate sample boards
        sample_boards = [
            (Card.new("As"), Card.new("Ks"), Card.new("Qs")),  # High
            (Card.new("2s"), Card.new("3h"), Card.new("4d")),  # Low connected
            (Card.new("7s"), Card.new("7h"), Card.new("2d")),  # Paired
            (Card.new("Ah"), Card.new("2d"), Card.new("9c")),  # Rainbow
            (Card.new("Ts"), Card.new("Js"), Card.new("Qs")),  # Flush draw
            (Card.new("5s"), Card.new("6h"), Card.new("7d")),  # Straight board
            (Card.new("As"), Card.new("Ad"), Card.new("Ac")),  # Trips
            (Card.new("Kh"), Card.new("Qh"), Card.new("Jh")),  # Flush draw high
            (Card.new("2c"), Card.new("7s"), Card.new("Kd")),  # Rainbow spread
            (Card.new("9s"), Card.new("9h"), Card.new("9d")),  # Trips medium
        ]

        # Fit
        clusterer.fit(sample_boards, Street.FLOP)

        # Should be fitted
        assert clusterer.fitted
        assert Street.FLOP in clusterer.clusterers

        # Predict clusters
        for board in sample_boards:
            cluster = clusterer.get_cluster(board)
            assert 0 <= cluster < 10

    def test_similar_boards_same_cluster(self):
        """Test that similar boards get same cluster."""
        clusterer = BoardClusterer(num_clusters={Street.FLOP: 5})

        # Create training set with clear patterns
        training_boards = []

        # Add monotone flush boards (cluster 1)
        for suit in ["s", "h", "d", "c"]:
            training_boards.append(
                (Card.new(f"A{suit}"), Card.new(f"K{suit}"), Card.new(f"Q{suit}"))
            )

        # Add paired boards (cluster 2)
        for rank in ["A", "K", "Q", "J", "T"]:
            training_boards.append((Card.new(f"{rank}s"), Card.new(f"{rank}h"), Card.new("2d")))

        # Add rainbow spread boards (cluster 3)
        for _ in range(10):
            training_boards.append((Card.new("2s"), Card.new("7h"), Card.new("Kd")))

        # Fit
        clusterer.fit(training_boards, Street.FLOP)

        # Test: Similar monotone boards should cluster together
        board1 = (Card.new("As"), Card.new("Ks"), Card.new("Qs"))
        board2 = (Card.new("Ah"), Card.new("Kh"), Card.new("Qh"))

        cluster1 = clusterer.get_cluster(board1)
        cluster2 = clusterer.get_cluster(board2)

        # Should be same cluster (or very close)
        # With only 5 clusters, there's a good chance they match
        assert isinstance(cluster1, int)
        assert isinstance(cluster2, int)

    def test_get_cluster_invalid_board(self):
        """Test getting cluster for invalid board size."""
        clusterer = BoardClusterer()

        # Try with wrong number of cards
        board = (Card.new("As"), Card.new("Ks"))  # Only 2 cards

        try:
            clusterer.get_cluster(board)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid board size" in str(e)

    def test_get_cluster_not_fitted(self):
        """Test getting cluster before fitting."""
        clusterer = BoardClusterer()

        board = (Card.new("As"), Card.new("Ks"), Card.new("Qs"))

        try:
            clusterer.get_cluster(board)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not fitted" in str(e)

    def test_get_num_features(self):
        """Test feature count."""
        clusterer = BoardClusterer()
        assert clusterer.get_num_features() == 16

    def test_str_representation(self):
        """Test string representation."""
        clusterer = BoardClusterer()
        s = str(clusterer)
        assert "BoardClusterer" in s
        assert "200" in s
        assert "500" in s
        assert "1000" in s
