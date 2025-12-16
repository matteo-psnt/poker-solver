#!/usr/bin/env python3
"""
Main training script for HUNLHE solver.

Usage:
    python scripts/train.py                              # Use default config
    python scripts/train.py --config config/custom.yaml  # Use custom config
    python scripts/train.py --iterations 10000           # Override iterations
    python scripts/train.py --resume                     # Resume from checkpoint
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import Trainer
from src.utils.config import Config, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train HUNLHE poker solver using Monte Carlo CFR"
    )

    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to YAML configuration file (default: use built-in defaults)",
    )

    # Training parameters
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=None,
        help="Number of training iterations (overrides config)",
    )

    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume from latest checkpoint",
    )

    # Storage
    parser.add_argument(
        "--storage",
        choices=["memory", "disk"],
        default=None,
        help="Storage backend (overrides config)",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (overrides config)",
    )

    # System
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Disable progress output",
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Load configuration
    print("Loading configuration...")
    if args.config:
        print(f"  Using config file: {args.config}")
        config = Config.from_file(args.config)
    else:
        print("  Using default configuration")
        config = Config.default()

    # Apply command-line overrides
    if args.storage:
        config.set("storage.backend", args.storage)

    if args.checkpoint_dir:
        config.set("training.checkpoint_dir", str(args.checkpoint_dir))

    if args.seed is not None:
        config.set("system.seed", args.seed)

    if args.quiet:
        config.set("training.verbose", False)

    # Validate config
    config.validate()

    # Print config summary
    print("\nConfiguration Summary:")
    print(f"  Game: {config.get('game.starting_stack')}BB stacks")
    print(f"  Card Abstraction: {config.get('card_abstraction.type')}")
    print(f"  Storage: {config.get('storage.backend')}")
    print(f"  Checkpoint Dir: {config.get('training.checkpoint_dir')}")

    if args.seed is not None:
        print(f"  Seed: {args.seed}")

    # Build trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config)

    # Run training
    iterations = args.iterations or config.get("training.num_iterations")
    print(f"\nStarting training for {iterations:,} iterations...")

    try:
        results = trainer.train(
            num_iterations=iterations,
            resume=args.resume,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total Iterations: {results['total_iterations']:,}")
        print(f"Final Infosets: {results['final_infosets']:,}")
        print(f"Average Utility: {results['avg_utility']:+.2f}")
        print(f"Elapsed Time: {results['elapsed_time']:.1f}s")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Creating checkpoint...")
        trainer.checkpoint_manager.save(trainer.solver, trainer.solver.iteration)
        print("Checkpoint saved. You can resume with --resume flag.")
        return 1

    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
