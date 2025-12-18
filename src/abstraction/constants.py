"""
Constants for poker abstraction system.

Centralizes magic numbers and tunable parameters for easy experimentation.
Note: Street-keyed dictionaries are created at runtime to avoid circular imports.
"""

# =============================================================================
# Card Bucketing Constants
# =============================================================================

# Number of equity buckets per street
# These determine the granularity of hand strength abstraction
# Note: Callers should use get_default_num_buckets() to get Street-keyed dict
DEFAULT_FLOP_BUCKETS = 50
DEFAULT_TURN_BUCKETS = 100
DEFAULT_RIVER_BUCKETS = 200

# Number of board clusters per street
# These reduce the state space for precomputation
DEFAULT_FLOP_BOARD_CLUSTERS = 200
DEFAULT_TURN_BOARD_CLUSTERS = 500
DEFAULT_RIVER_BOARD_CLUSTERS = 1000

# Small values for testing (faster computation, less accurate)
TEST_FLOP_BUCKETS = 10
TEST_TURN_BUCKETS = 20
TEST_RIVER_BUCKETS = 30

TEST_FLOP_BOARD_CLUSTERS = 10
TEST_TURN_BOARD_CLUSTERS = 10
TEST_RIVER_BOARD_CLUSTERS = 10

# =============================================================================
# Equity Calculator Constants
# =============================================================================

# Default number of Monte Carlo samples for equity calculation
DEFAULT_EQUITY_SAMPLES = 1000

# Number of samples for testing (faster but less accurate)
TEST_EQUITY_SAMPLES = 100

# =============================================================================
# SPR (Stack-to-Pot Ratio) Constants
# =============================================================================

# SPR bucket thresholds
# Shallow: SPR < 4 (push/fold decisions)
# Medium: 4 <= SPR < 13 (standard play)
# Deep: SPR >= 13 (complex postflop play)
SPR_SHALLOW_THRESHOLD = 4.0
SPR_DEEP_THRESHOLD = 13.0

# =============================================================================
# Board Clustering Constants
# =============================================================================

# Default number of samples per board cluster for equity matrix computation
DEFAULT_SAMPLES_PER_CLUSTER = 10

# =============================================================================
# Preflop Constants
# =============================================================================

# Total number of canonical preflop hands (169)
# 13 pairs + 78 suited + 78 offsuit
NUM_PREFLOP_HANDS = 169

# Number of ranks and suits in standard deck
NUM_RANKS = 13
NUM_SUITS = 4
DECK_SIZE = 52

# =============================================================================
# K-means Clustering Constants
# =============================================================================

# Maximum iterations for K-means convergence
KMEANS_MAX_ITER = 300

# Random seed for reproducible clustering
KMEANS_RANDOM_STATE = 42
