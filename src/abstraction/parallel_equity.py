"""
Parallel equity computation for faster precomputation.

Uses multiprocessing to compute equity for multiple hands simultaneously.
"""

import logging
import multiprocessing as mp
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.abstraction.equity_calculator import EquityCalculator
from src.abstraction.preflop_hands import PreflopHandMapper
from src.game.state import Card, Street

logger = logging.getLogger(__name__)


def _compute_hand_equity_row(args):
    """
    Worker function to compute equity for one hand across all clusters.

    This runs in a separate process.

    Args:
        args: Tuple of (hand_idx, hand_string, cluster_representatives,
                       num_equity_samples, street, seed)

    Returns:
        Tuple of (hand_idx, equity_row, conflicts_info)
    """
    (
        hand_idx,
        hand_string,
        cluster_representatives,
        num_equity_samples,
        street,
        seed,
    ) = args

    # Create equity calculator in worker process
    equity_calc = EquityCalculator(num_samples=num_equity_samples, seed=seed + hand_idx)

    # Get example hand
    hole_cards = _get_example_hand(hand_string)

    # Pre-compute hole card set for faster conflict detection
    hole_card_ints = {c.card_int for c in hole_cards}

    # Compute equity for each cluster
    num_clusters = len(cluster_representatives)
    equity_row = np.zeros(num_clusters)

    empty_clusters = 0
    conflict_defaults = 0

    for cluster_id in range(num_clusters):
        boards = cluster_representatives[cluster_id]

        if len(boards) == 0:
            equity_row[cluster_id] = 0.5
            empty_clusters += 1
            continue

        # Compute average equity across representative boards
        equities = []
        conflicts = 0

        for board in boards:
            # Skip if hole cards conflict with board (optimized check)
            board_ints = {c.card_int for c in board}
            if hole_card_ints & board_ints:  # Set intersection
                conflicts += 1
                continue

            equity = equity_calc.calculate_equity(hole_cards, board, street)
            equities.append(equity)

        if len(equities) == 0:
            # All boards conflicted - use default
            equity_row[cluster_id] = 0.5
            conflict_defaults += 1
        else:
            equity_row[cluster_id] = np.mean(equities)

    return hand_idx, equity_row, (empty_clusters, conflict_defaults)


def _get_example_hand(hand_string: str) -> Tuple[Card, Card]:
    """Get a concrete example of a hand string."""
    if len(hand_string) == 2:
        # Pair
        rank = hand_string[0]
        return (Card.new(f"{rank}h"), Card.new(f"{rank}d"))
    else:
        # Suited or offsuit
        high_rank = hand_string[0]
        low_rank = hand_string[1]
        suited = hand_string[2] == "s"

        if suited:
            return (Card.new(f"{high_rank}s"), Card.new(f"{low_rank}s"))
        else:
            return (Card.new(f"{high_rank}h"), Card.new(f"{low_rank}d"))


def _cards_conflict(
    hole_cards: Tuple[Card, Card],
    board: Tuple[Card, ...],
) -> bool:
    """
    Check if hole cards conflict with board cards.

    Optimized to use card_int comparison for speed.
    """
    hole_ints = {c.card_int for c in hole_cards}
    board_ints = {c.card_int for c in board}
    return bool(hole_ints & board_ints)  # Returns True if any overlap


def compute_equity_matrix_parallel(
    cluster_representatives: Dict[int, List],
    street: Street,
    num_equity_samples: int,
    seed: int,
    num_workers: int = None,
) -> np.ndarray:
    """
    Compute equity matrix using multiprocessing.

    Args:
        cluster_representatives: Dict mapping cluster_id -> list of boards
        street: Which street
        num_equity_samples: MC samples per equity calculation
        seed: Random seed
        num_workers: Number of worker processes (None = auto)

    Returns:
        equity_matrix: shape [169 hands, num_board_clusters]
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    num_clusters = len(cluster_representatives)
    equity_matrix = np.zeros((169, num_clusters))

    # Get all 169 hand strings
    all_hands = PreflopHandMapper.get_all_hands()

    # Prepare arguments for workers
    work_args = [
        (
            hand_idx,
            all_hands[hand_idx],
            cluster_representatives,
            num_equity_samples,
            street,
            seed,
        )
        for hand_idx in range(169)
    ]

    # Run parallel computation
    logger.info(f"Computing {street.name} equity matrix with {num_workers} workers...")

    empty_clusters_total = 0
    conflict_defaults_total = 0

    # Use multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for progress tracking
        results = list(
            tqdm(
                pool.imap_unordered(_compute_hand_equity_row, work_args),
                total=169,
                desc=f"{street.name} equity matrix (parallel)",
                unit="hand",
            )
        )

    # Collect results
    for hand_idx, equity_row, (empty_clusters, conflict_defaults) in results:
        equity_matrix[hand_idx, :] = equity_row
        empty_clusters_total += empty_clusters
        conflict_defaults_total += conflict_defaults

    # Log summary
    total_cells = 169 * num_clusters
    if empty_clusters_total > 0 or conflict_defaults_total > 0:
        logger.warning(
            f"{street.name}: {empty_clusters_total + conflict_defaults_total} cells "
            f"used default equity 0.5 ({100 * (empty_clusters_total + conflict_defaults_total) / total_cells:.2f}%)"
        )

    return equity_matrix
