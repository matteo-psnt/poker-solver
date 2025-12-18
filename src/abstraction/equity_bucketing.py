"""
Equity-based card abstraction using Monte Carlo sampling.

This module implements equity bucketing, which clusters hands based on
their equity (win probability) against a random opponent hand.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from treys import Deck

from src.abstraction.card_abstraction import CardAbstraction
from src.abstraction.clustering import KMeansClustering
from src.game.evaluator import get_evaluator
from src.game.state import Card, Street


class EquityBucketing(CardAbstraction):
    """
    Equity-based card abstraction using Monte Carlo rollouts.

    For each hand, we:
    1. Compute equity via Monte Carlo simulation (1000+ rollouts)
    2. Cluster hands with similar equities using k-means
    3. Assign bucket based on nearest cluster center

    This is precomputed offline and saved to disk.
    """

    def __init__(
        self,
        num_buckets_per_street: Optional[Dict[Street, int]] = None,
        num_rollouts: int = 1000,
    ):
        """
        Initialize equity bucketing.

        Args:
            num_buckets_per_street: Number of buckets for each street
                                   {Street.PREFLOP: 8, Street.FLOP: 50, ...}
            num_rollouts: Number of Monte Carlo rollouts for equity calculation
        """
        self.num_rollouts = num_rollouts
        self.evaluator = get_evaluator()

        if num_buckets_per_street is None:
            # Default configuration
            self.num_buckets_per_street = {
                Street.PREFLOP: 8,
                Street.FLOP: 50,
                Street.TURN: 50,
                Street.RIVER: 50,
            }
        else:
            self.num_buckets_per_street = num_buckets_per_street

        # Cluster centers (set via precomputation)
        self.bucket_centers: Dict[Street, np.ndarray] = {}

        # Whether bucketing has been precomputed
        self.is_fitted = False

    def get_bucket(
        self, hole_cards: Tuple[Card, Card], board: Tuple[Card, ...], street: Street
    ) -> int:
        """
        Get bucket for a hand.

        Args:
            hole_cards: Player's hole cards
            board: Community cards
            street: Current street

        Returns:
            Bucket ID (0 to num_buckets-1)
        """
        if not self.is_fitted:
            raise ValueError("Bucketing not precomputed. Call precompute_buckets() first.")

        if street not in self.bucket_centers:
            raise ValueError(f"No bucket centers for street {street}")

        # Compute equity
        equity = self.compute_equity(hole_cards, board, street)

        # Find nearest cluster center
        centers = self.bucket_centers[street]
        distances = np.abs(centers - equity)
        bucket = int(np.argmin(distances))

        return bucket

    def compute_equity(
        self,
        hole_cards: Tuple[Card, Card],
        board: Tuple[Card, ...],
        street: Street,
        num_rollouts: Optional[int] = None,
    ) -> float:
        """
        Compute hand equity via Monte Carlo simulation.

        Args:
            hole_cards: Player's hole cards
            board: Community cards
            street: Current street
            num_rollouts: Number of simulations (default: self.num_rollouts)

        Returns:
            Equity (win probability) between 0 and 1
        """
        if num_rollouts is None:
            num_rollouts = self.num_rollouts

        # Convert cards to treys format
        hero_cards = [c.card_int for c in hole_cards]
        board_cards = [c.card_int for c in board]

        # Create deck and remove known cards
        deck = Deck()
        deck.cards = [c for c in deck.cards if c not in hero_cards and c not in board_cards]

        wins = 0
        ties = 0

        for _ in range(num_rollouts):
            # Shuffle deck
            random.shuffle(deck.cards)

            # Deal opponent cards
            villain_cards = deck.cards[:2]

            # Deal remaining board cards
            remaining_board = board_cards.copy()
            board_idx = 2

            if street == Street.PREFLOP:
                # Deal all 5 board cards
                remaining_board.extend(deck.cards[board_idx : board_idx + 5])
            elif street == Street.FLOP:
                # Deal turn and river
                remaining_board.extend(deck.cards[board_idx : board_idx + 2])
            elif street == Street.TURN:
                # Deal river
                remaining_board.append(deck.cards[board_idx])
            # else RIVER: board is complete

            # Evaluate hands
            hero_rank = self.evaluator.evaluator.evaluate(remaining_board, hero_cards)
            villain_rank = self.evaluator.evaluator.evaluate(remaining_board, villain_cards)

            # Lower rank = better hand
            if hero_rank < villain_rank:
                wins += 1
            elif hero_rank == villain_rank:
                ties += 1

        # Equity = (wins + 0.5 * ties) / total
        equity = (wins + 0.5 * ties) / num_rollouts

        return equity

    def precompute_buckets(
        self,
        num_samples_per_street: Optional[Dict[Street, int]] = None,
        seed: int = 42,
    ):
        """
        Precompute bucket centers via k-means clustering.

        This samples random hands, computes equities, and clusters them.

        Args:
            num_samples_per_street: Number of hands to sample per street
                                   {Street.PREFLOP: 10000, ...}
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

        if num_samples_per_street is None:
            # Default: sample 10K hands per street
            num_samples_per_street = {
                Street.PREFLOP: 10000,
                Street.FLOP: 5000,
                Street.TURN: 3000,
                Street.RIVER: 2000,
            }

        print("Precomputing equity buckets...")

        for street in [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]:
            num_buckets = self.num_buckets_per_street[street]
            num_samples = num_samples_per_street[street]

            print(f"\n{street.name}: Sampling {num_samples} hands into {num_buckets} buckets...")

            # Sample random hands and compute equities
            equities = []
            for i in range(num_samples):
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{num_samples} hands...")

                # Sample random hand and board
                hole_cards, board = self._sample_random_hand(street)

                # Compute equity
                equity = self.compute_equity(hole_cards, board, street)
                equities.append(equity)

            equities = np.array(equities).reshape(-1, 1)  # Shape: [n_samples, 1]

            # Cluster equities
            print(f"  Clustering {num_samples} hands into {num_buckets} buckets...")
            kmeans = KMeansClustering(n_clusters=num_buckets)
            kmeans.fit(equities, seed=seed)

            # Store cluster centers
            self.bucket_centers[street] = kmeans.cluster_centers.flatten()

            print(f"  {street.name} bucketing complete!")
            print(f"  Bucket centers: {self.bucket_centers[street]}")

        self.is_fitted = True
        print("\nAll buckets precomputed successfully!")

    def _sample_random_hand(self, street: Street) -> Tuple[Tuple[Card, Card], Tuple[Card, ...]]:
        """
        Sample a random hand and board for a given street.

        Args:
            street: Street to sample for

        Returns:
            (hole_cards, board)
        """
        deck = Deck()
        random.shuffle(deck.cards)

        # Deal hole cards
        hole_cards = (Card(deck.cards[0]), Card(deck.cards[1]))

        # Deal board based on street
        if street == Street.PREFLOP:
            board = tuple()
        elif street == Street.FLOP:
            board = tuple(Card(c) for c in deck.cards[2:5])
        elif street == Street.TURN:
            board = tuple(Card(c) for c in deck.cards[2:6])
        elif street == Street.RIVER:
            board = tuple(Card(c) for c in deck.cards[2:7])
        else:
            raise ValueError(f"Unknown street: {street}")

        return hole_cards, board

    def save(self, filepath: Path):
        """
        Save precomputed buckets to disk.

        Args:
            filepath: Path to save file (.pkl)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted bucketing")

        data = {
            "num_buckets_per_street": self.num_buckets_per_street,
            "num_rollouts": self.num_rollouts,
            "bucket_centers": self.bucket_centers,
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved equity bucketing to {filepath}")

    def load(self, filepath: Path):
        """
        Load precomputed buckets from disk.

        Args:
            filepath: Path to load from (.pkl)
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.num_buckets_per_street = data["num_buckets_per_street"]
        self.num_rollouts = data["num_rollouts"]
        self.bucket_centers = data["bucket_centers"]
        self.is_fitted = True

        print(f"Loaded equity bucketing from {filepath}")

    def num_buckets(self, street: Street) -> int:
        """Get number of buckets for a street."""
        return self.num_buckets_per_street[street]

    def __str__(self) -> str:
        buckets_str = ", ".join([f"{s.name}={self.num_buckets_per_street[s]}" for s in Street])
        return f"EquityBucketing({buckets_str}, rollouts={self.num_rollouts})"
