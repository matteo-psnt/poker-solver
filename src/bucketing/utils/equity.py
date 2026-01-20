"""
Monte Carlo equity calculator for poker hands.

Computes hand equity via random rollouts to the river.
"""

import random

import numpy as np

from src.game.evaluator import get_evaluator
from src.game.state import Card, Street


class EquityCalculator:
    """
    Monte Carlo equity calculator.

    Computes equity by sampling random opponent hands and board runouts.
    """

    def __init__(self, num_samples: int = 1000, seed: int | None = None):
        """
        Initialize equity calculator.

        Args:
            num_samples: Number of Monte Carlo samples per calculation
            seed: Random seed for reproducibility (None = random)
        """
        self.num_samples = num_samples
        self.evaluator = get_evaluator()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Cache the full deck for performance (huge speedup!)
        self.full_deck = Card.get_full_deck()

        # Pre-compute card mask set for faster filtering
        self._full_deck_card_masks = {c.mask for c in self.full_deck}

    def calculate_equity(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> float:
        """
        Calculate equity (win %) for a hand.

        Args:
            hole_cards: Player's two hole cards
            board: Community cards (can be empty for preflop)
            street: Current street

        Returns:
            Equity as float between 0 and 1

        Example:
            equity_calc = EquityCalculator(num_samples=1000)
            equity = equity_calc.calculate_equity(
                hole_cards=(Card.new("As"), Card.new("Ks")),
                board=(Card.new("Ah"), Card.new("Kh"), Card.new("2c")),
                street=Street.FLOP
            )
            # equity ≈ 0.95 (two pair is very strong)
        """
        wins = 0
        ties = 0

        for _ in range(self.num_samples):
            # Sample opponent hand and complete board
            opp_hand, full_board = self._sample_runout(hole_cards, board, street)

            # Evaluate hands
            result = self._evaluate_showdown(hole_cards, opp_hand, full_board)

            if result < 0:  # Win (compare_hands returns -1 for hand1 win)
                wins += 1
            elif result == 0:  # Tie
                ties += 1

        # Equity = (wins + 0.5 * ties) / total
        return (wins + 0.5 * ties) / self.num_samples

    def calculate_equity_distribution(
        self,
        hole_cards: tuple[Card, Card],
        board: tuple[Card, ...],
        street: Street,
        num_buckets: int = 10,
    ) -> np.ndarray:
        """
        Calculate equity distribution histogram for potential-aware clustering.

        This captures not just current equity, but also hand potential (draws).

        Args:
            hole_cards: Player's two hole cards
            board: Community cards
            street: Current street
            num_buckets: Number of buckets for histogram (default: 10 for 0-10%, 10-20%, etc.)

        Returns:
            Histogram of equity distribution (sums to 1.0)

        Example:
            # Made hand (top pair)
            dist = calculate_equity_distribution((As, Kc), (Ah, 7s, 2d), Street.FLOP)
            # dist ≈ [0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.5, 0.2]
            # Most mass in 70-90% range (strong but not nutted)

            # Draw (flush draw)
            dist = calculate_equity_distribution((As, Ks), (Ah, 7s, 2s), Street.FLOP)
            # dist ≈ [0.3, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5]
            # Bimodal: either miss (0-10%) or hit (80-100%)
        """
        histogram = np.zeros(num_buckets)

        for _ in range(self.num_samples):
            # Sample opponent hand and complete board
            opp_hand, full_board = self._sample_runout(hole_cards, board, street)

            # Compute final equity (1 if win, 0.5 if tie, 0 if loss)
            result = self._evaluate_showdown(hole_cards, opp_hand, full_board)

            if result < 0:  # Win (compare_hands returns -1 for hand1 win)
                equity = 1.0
            elif result == 0:  # Tie
                equity = 0.5
            else:  # Loss
                equity = 0.0

            # Add to histogram
            bucket_idx = min(int(equity * num_buckets), num_buckets - 1)
            histogram[bucket_idx] += 1

        # Normalize to sum to 1
        return histogram / self.num_samples

    def _sample_runout(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> tuple[tuple[Card, Card], tuple[Card, ...]]:
        """
        Sample opponent hand and complete board to river.

        Optimized to minimize list comprehensions and sampling operations.

        Args:
            hole_cards: Player's hole cards
            board: Current board
            street: Current street

        Returns:
            (opponent_hand, completed_board)
        """
        # Create set of used card masks for fast lookup
        used_card_masks = {c.mask for c in hole_cards}
        used_card_masks.update(c.mask for c in board)

        # Filter deck once using card mask (faster than Card equality)
        # Use list comp with direct card mask check (avoids set lookups in tight loop)
        available = [c for c in self.full_deck if c.mask not in used_card_masks]

        # Calculate total cards needed (opponent + remaining board)
        cards_needed_board = 5 - len(board)
        total_cards_needed = 2 + cards_needed_board  # 2 for opponent + N for board

        # Sample all cards at once (more efficient than multiple samples)
        sampled_cards = random.sample(available, total_cards_needed)

        # Split into opponent hand and board cards
        opp_hand: tuple[Card, Card] = (sampled_cards[0], sampled_cards[1])

        if cards_needed_board > 0:
            new_board_cards = tuple(sampled_cards[2:])
            full_board = board + new_board_cards
        else:
            full_board = board

        return opp_hand, full_board

    def _evaluate_showdown(
        self,
        hand1: tuple[Card, Card],
        hand2: tuple[Card, Card],
        board: tuple[Card, ...],
    ) -> int:
        """
        Evaluate showdown between two hands.

        Args:
            hand1: First player's hand
            hand2: Second player's hand
            board: Board cards (must be 5 cards)

        Returns:
            -1 if hand1 loses, 0 if tie, +1 if hand1 wins
        """
        return self.evaluator.compare_hands(hand1, hand2, board)

    def batch_calculate_equity(
        self,
        hands: list[tuple[Card, Card]],
        board: tuple[Card, ...],
        street: Street,
    ) -> np.ndarray:
        """
        Calculate equity for multiple hands efficiently.

        Uses the same opponent samples for all hands for faster computation.

        Args:
            hands: List of hole card pairs
            board: Community cards
            street: Current street

        Returns:
            Array of equities (one per hand)
        """
        num_hands = len(hands)
        wins = np.zeros(num_hands)
        ties = np.zeros(num_hands)

        for _ in range(self.num_samples):
            # Sample one opponent hand and board
            # Create deck excluding board
            used_cards = set(board)
            deck = [c for c in Card.get_full_deck() if c not in used_cards]

            # Sample opponent hand
            opp_cards = random.sample(deck, 2)
            opp_hand = tuple(opp_cards)

            # Complete board
            deck = [c for c in deck if c not in opp_cards]
            cards_needed = 5 - len(board)
            if cards_needed > 0:
                new_cards = random.sample(deck, cards_needed)
                full_board = board + tuple(new_cards)
            else:
                full_board = board

            # Evaluate all hands against this opponent
            for i, hand in enumerate(hands):
                # Skip if hand uses opponent's cards or board cards
                if any(c in opp_hand or c in full_board for c in hand):
                    continue

                result = self._evaluate_showdown(hand, (opp_hand[0], opp_hand[1]), full_board)
                if result < 0:  # Win (compare_hands returns -1 for hand1 win)
                    wins[i] += 1
                elif result == 0:
                    ties[i] += 1

        # Calculate equities
        equities = (wins + 0.5 * ties) / self.num_samples
        return equities

    def __str__(self) -> str:
        """String representation."""
        return f"EquityCalculator(samples={self.num_samples})"
