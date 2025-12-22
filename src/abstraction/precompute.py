"""
Equity bucketing precomputation module.

Provides optimized functions for precomputing equity-based card abstraction.
Supports parallelization, checkpointing, and resumable computation.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from src.abstraction.equity_bucketing import EquityBucketing
from src.abstraction.equity_calculator import EquityCalculator
from src.game.state import Card, Street

logger = logging.getLogger(__name__)


@dataclass
class PrecomputeConfig:
    """Configuration for equity bucketing precomputation."""

    # Board sampling
    num_samples_per_street: Dict[Street, int]

    # Bucketing parameters
    num_buckets: Dict[Street, int]
    num_board_clusters: Dict[Street, int]

    # Equity calculation
    num_equity_samples: int
    num_samples_per_cluster: int

    # Output
    output_file: Path

    # Optional
    seed: int = 42
    num_workers: int = 1  # Number of parallel workers (1 = sequential, >1 = parallel)
    aliases: List[str] = field(default_factory=list)  # User-defined aliases for this abstraction
    config_name: Optional[str] = None  # Name of the config (e.g., "production", "test")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PrecomputeConfig":
        """
        Create PrecomputeConfig from dictionary (e.g., loaded from YAML).

        Args:
            config_dict: Configuration dictionary

        Returns:
            PrecomputeConfig instance
        """

        # Convert string keys to Street enums
        def convert_street_dict(d: dict) -> Dict[Street, int]:
            result = {}
            for key, value in d.items():
                if isinstance(key, str):
                    # Convert string to Street enum
                    result[Street[key]] = value
                else:
                    result[key] = value
            return result

        return cls(
            num_samples_per_street=convert_street_dict(config_dict["num_samples_per_street"]),
            num_buckets=convert_street_dict(config_dict["num_buckets"]),
            num_board_clusters=convert_street_dict(config_dict["num_board_clusters"]),
            num_equity_samples=config_dict["num_equity_samples"],
            num_samples_per_cluster=config_dict["num_samples_per_cluster"],
            output_file=Path(
                config_dict.get("output_file", "data/abstractions/equity_buckets.pkl")
            ),
            seed=config_dict.get("seed", 42),
            num_workers=config_dict.get("num_workers", 1),
            aliases=config_dict.get("aliases", []),
            config_name=config_dict.get("config_name"),
        )

    @classmethod
    def default(cls) -> "PrecomputeConfig":
        """Get default production configuration."""
        return cls(
            num_samples_per_street={
                Street.FLOP: 5000,
                Street.TURN: 3000,
                Street.RIVER: 2000,
            },
            num_buckets={
                Street.FLOP: 50,
                Street.TURN: 100,
                Street.RIVER: 200,
            },
            num_board_clusters={
                Street.FLOP: 200,
                Street.TURN: 500,
                Street.RIVER: 1000,
            },
            num_equity_samples=1000,
            num_samples_per_cluster=5,
            output_file=Path("data/abstractions/equity_buckets.pkl"),
            config_name="production",
            aliases=["default"],  # Additional alias
        )

    @classmethod
    def fast_test(cls) -> "PrecomputeConfig":
        """Get fast configuration for testing (~2-5 minutes)."""
        return cls(
            num_samples_per_street={
                Street.FLOP: 500,
                Street.TURN: 300,
                Street.RIVER: 200,
            },
            num_buckets={
                Street.FLOP: 10,
                Street.TURN: 20,
                Street.RIVER: 30,
            },
            num_board_clusters={
                Street.FLOP: 20,
                Street.TURN: 30,
                Street.RIVER: 40,
            },
            num_equity_samples=100,
            num_samples_per_cluster=3,
            output_file=Path("data/abstractions/equity_buckets_test.pkl"),
            config_name="fast_test",
            aliases=["test"],  # Additional alias
        )


def generate_boards_optimized(
    street: Street,
    num_samples: int,
    seed: int = 42,
    show_progress: bool = True,
) -> List[Tuple[Card, ...]]:
    """
    Generate random sample boards efficiently.

    Optimized version that doesn't shuffle entire deck each time.

    Args:
        street: Which street to generate for
        num_samples: Number of boards to sample
        seed: Random seed
        show_progress: Show progress bar

    Returns:
        List of board tuples
    """
    rng = np.random.RandomState(seed)
    boards = []

    # Board size based on street
    board_size = {
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
    }[street]

    # Use tqdm if requested
    iterator: Union[range, tqdm] = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Sampling {street.name} boards", unit="board")

    # Generate boards using efficient sampling
    for _ in iterator:
        # Sample without replacement (faster than shuffle)
        board_indices = rng.choice(52, size=board_size, replace=False)

        # Convert indices to Card objects
        board = tuple(Card.get_full_deck()[i] for i in board_indices)
        boards.append(board)

    return boards


def precompute_equity_bucketing(
    config: PrecomputeConfig,
    checkpoint_file: Optional[Path] = None,
    resume: bool = False,
    save_with_metadata: bool = True,
) -> EquityBucketing:
    """
    Main precomputation pipeline.

    Args:
        config: Precomputation configuration
        checkpoint_file: Optional checkpoint file for resume
        resume: Whether to resume from checkpoint

    Returns:
        Fitted EquityBucketing object
    """
    logger.info("Starting equity bucketing precomputation")
    logger.info(f"Configuration: {config}")

    start_time = time.time()

    # Check for resume
    if resume and checkpoint_file and checkpoint_file.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        return EquityBucketing.load(checkpoint_file)

    # Step 1: Generate sample boards
    logger.info("Step 1: Generating sample boards")
    sample_boards = {}

    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_samples = config.num_samples_per_street[street]
        logger.info(f"  {street.name}: {num_samples:,} boards")

        boards = generate_boards_optimized(
            street=street,
            num_samples=num_samples,
            seed=config.seed,
            show_progress=True,
        )
        sample_boards[street] = boards

    boards_time = time.time() - start_time
    logger.info(f"Board generation completed in {boards_time:.1f}s")

    # Step 2: Create equity bucketing system
    logger.info("Step 2: Creating equity bucketing system")

    equity_calc = EquityCalculator(
        num_samples=config.num_equity_samples,
        seed=config.seed,
    )

    bucketing = EquityBucketing(
        num_buckets=config.num_buckets,
        num_board_clusters=config.num_board_clusters,
        equity_calculator=equity_calc,
    )

    logger.info(f"  Buckets: {config.num_buckets}")
    logger.info(f"  Board clusters: {config.num_board_clusters}")
    logger.info(f"  Equity samples: {config.num_equity_samples}")

    # Step 3: Fit bucketing system
    logger.info("Step 3: Fitting bucketing system")

    if config.num_workers > 1:
        logger.info(f"  Using {config.num_workers} parallel workers")
        logger.info("  This significantly speeds up computation!")
    else:
        logger.info("  Using sequential computation (1 worker)")

    logger.info("  This is the slow part - computing equity matrices...")

    # Calculate total work
    total_calcs = sum(
        169 * config.num_board_clusters[street] * config.num_samples_per_cluster
        for street in [Street.FLOP, Street.TURN, Street.RIVER]
    )
    logger.info(
        f"  Total: {total_calcs:,} equity calculations "
        f"({total_calcs * config.num_equity_samples:,} MC rollouts)"
    )

    fit_start = time.time()
    bucketing.fit(
        sample_boards,
        num_samples_per_cluster=config.num_samples_per_cluster,
        num_workers=config.num_workers if config.num_workers > 1 else None,
    )
    fit_time = time.time() - fit_start

    logger.info(f"Fitting completed in {fit_time:.1f}s ({fit_time / 60:.1f} minutes)")

    # Step 4: Save to disk
    logger.info("Step 4: Saving to disk")
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    bucketing.save(config.output_file)

    file_size = config.output_file.stat().st_size
    logger.info(f"  File: {config.output_file}")
    logger.info(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    # Summary
    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    # Step 5: Save with metadata if requested
    if save_with_metadata:
        from datetime import datetime

        from src.abstraction.abstraction_metadata import AbstractionManager, AbstractionMetadata

        logger.info("Step 5: Saving with metadata")

        # Create metadata
        metadata = AbstractionMetadata(
            name="",  # Will be auto-generated from config
            created_at=datetime.now().isoformat(),
            abstraction_type="equity_bucketing",
            num_buckets={street.name: config.num_buckets[street] for street in config.num_buckets},
            num_board_clusters={
                street.name: config.num_board_clusters[street]
                for street in config.num_board_clusters
            },
            num_equity_samples=config.num_equity_samples,
            num_samples_per_cluster=config.num_samples_per_cluster,
            seed=config.seed,
        )

        # Save with metadata (auto-generate name from config)
        manager = AbstractionManager()

        # Build aliases: config name + explicit aliases from config
        aliases = []

        # Add config name as automatic alias (if provided and not "default")
        if config.config_name and config.config_name != "default":
            aliases.append(config.config_name)

        # Add explicit aliases from config
        aliases.extend(config.aliases)

        abstraction_dir = manager.save_abstraction(
            config.output_file,
            metadata,
            aliases=aliases,
            auto_name=True,  # Generate name from config
        )

        logger.info(f"  Saved abstraction with metadata to: {abstraction_dir}")
        logger.info(f"  Name: {metadata.name}")
        if aliases:
            logger.info(f"  Aliases: {', '.join(aliases)}")
        else:
            logger.info("  No aliases set (use config_name or aliases field to add)")
        logger.info("  Use AbstractionManager().list_abstractions() to view all")

    return bucketing


def validate_bucketing(bucketing: EquityBucketing) -> None:
    """
    Validate a bucketing by testing a few cases.

    Args:
        bucketing: Bucketing to validate
    """
    logger.info("Validating bucketing...")

    test_cases = [
        # Premium hand on dry board
        (
            (Card.new("As"), Card.new("Ah")),
            (Card.new("Ks"), Card.new("Qs"), Card.new("Jh")),
            Street.FLOP,
            "Pocket Aces on dry board",
        ),
        # Weak hand on same board
        (
            (Card.new("7d"), Card.new("2c")),
            (Card.new("Ks"), Card.new("Qs"), Card.new("Jh")),
            Street.FLOP,
            "72o on same board",
        ),
        # Strong draw
        (
            (Card.new("Ts"), Card.new("9s")),
            (Card.new("8s"), Card.new("7s"), Card.new("2h")),
            Street.FLOP,
            "Straight flush draw",
        ),
    ]

    for hole_cards, board, street, description in test_cases:
        bucket = bucketing.get_bucket(hole_cards, board, street)
        logger.info(f"  {description}: Bucket {bucket}")

    logger.info("Validation complete")


def print_summary(config: PrecomputeConfig) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 80)
    print("EQUITY BUCKET PRECOMPUTATION - CONFIGURATION")
    print("=" * 80)
    print()
    print("Board Sampling:")
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_samples = config.num_samples_per_street[street]
        print(f"  {street.name:6s}: {num_samples:,} boards")

    print()
    print("Abstraction Size:")
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        clusters = config.num_board_clusters[street]
        buckets = config.num_buckets[street]
        print(f"  {street.name:6s}: {clusters:4d} board clusters → {buckets:3d} buckets")

    print()
    print("Equity Calculation:")
    print(f"  MC samples per calculation: {config.num_equity_samples}")
    print(f"  Boards per cluster: {config.num_samples_per_cluster}")

    # Calculate total work
    total_calcs = sum(
        169 * config.num_board_clusters[street] * config.num_samples_per_cluster
        for street in [Street.FLOP, Street.TURN, Street.RIVER]
    )
    total_rollouts = total_calcs * config.num_equity_samples

    print()
    print("Computational Cost:")
    print(f"  Total equity calculations: {total_calcs:,}")
    print(f"  Total MC rollouts: {total_rollouts:,}")

    # Estimate time
    calcs_per_second = 1000  # Rough estimate
    estimated_seconds = total_calcs / calcs_per_second

    print()
    print("Estimated Time:")
    print(f"  ~{estimated_seconds / 60:.0f} minutes ({estimated_seconds / 3600:.1f} hours)")

    print()
    print("Output:")
    print(f"  File: {config.output_file}")
    print("  Expected size: ~300 KB")
    print()
