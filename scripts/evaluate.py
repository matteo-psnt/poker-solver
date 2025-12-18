#!/usr/bin/env python3
"""
Evaluation script for HUNLHE solver.

Evaluates trained solvers through head-to-head matches and statistical analysis.

Usage:
    python scripts/evaluate.py --checkpoint data/checkpoints/run1  # Evaluate checkpoint
    python scripts/evaluate.py --compare checkpoint1 checkpoint2   # Compare two checkpoints
    python scripts/evaluate.py --self-play --hands 1000            # Self-play evaluation
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.abstraction.action_abstraction import ActionAbstraction
from src.abstraction.card_abstraction import RankBasedBucketing
from src.evaluation.head_to_head import HeadToHeadEvaluator, MatchStatistics
from src.evaluation.statistics import MatchStatisticsAnalyzer
from src.game.rules import GameRules
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import DiskBackedStorage, InMemoryStorage
from src.utils.config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HUNLHE poker solver performance")

    # Checkpoint selection
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint directory to evaluate",
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("CHECKPOINT1", "CHECKPOINT2"),
        help="Compare two checkpoints head-to-head",
    )

    # Evaluation parameters
    parser.add_argument(
        "--hands",
        "-n",
        type=int,
        default=1000,
        help="Number of hands to play (default: 1000)",
    )

    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Evaluate via self-play (same strategy vs itself)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )

    # Display
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def load_solver(checkpoint_dir: Path, config: Config) -> MCCFRSolver:
    """
    Load solver from checkpoint.

    Args:
        checkpoint_dir: Checkpoint directory
        config: Configuration

    Returns:
        Loaded solver
    """
    # Build components
    action_abs = ActionAbstraction(config.get_section("action_abstraction"))
    card_abs = RankBasedBucketing()

    # Load storage
    if checkpoint_dir.exists():
        storage = DiskBackedStorage(checkpoint_dir, cache_size=10000)
        print(f"  Loaded checkpoint with {storage.num_infosets()} infosets")
    else:
        storage = InMemoryStorage()
        print("  Warning: Checkpoint not found, using empty solver")

    # Build solver
    solver = MCCFRSolver(
        action_abstraction=action_abs,
        card_abstraction=card_abs,
        storage=storage,
        config=config.get_section("game"),
    )

    return solver


def evaluate_checkpoint(
    checkpoint_dir: Path,
    num_hands: int,
    seed: Optional[int],
    verbose: bool,
    confidence: float,
):
    """Evaluate single checkpoint via self-play."""
    print(f"\nEvaluating checkpoint: {checkpoint_dir}")
    print("=" * 60)

    # Load config and solver
    config = Config.default()
    print("\nLoading solver...")
    solver = load_solver(checkpoint_dir, config)

    # Setup evaluator
    rules = GameRules(
        small_blind=config.get("game.small_blind", 1),
        big_blind=config.get("game.big_blind", 2),
    )
    action_abs = ActionAbstraction(config.get_section("action_abstraction"))
    card_abs = RankBasedBucketing()

    evaluator = HeadToHeadEvaluator(
        rules=rules,
        action_abstraction=action_abs,
        card_abstraction=card_abs,
        starting_stack=config.get("game.starting_stack", 200),
    )

    # Run self-play match
    print(f"\nRunning self-play evaluation ({num_hands} hands)...")
    stats = evaluator.play_match(
        solver0=solver,
        solver1=solver,
        num_hands=num_hands,
        seed=seed,
    )

    # Print results
    print_match_results(stats, "Self-Play", verbose)

    # Statistical analysis
    analyzer = MatchStatisticsAnalyzer(confidence_level=confidence)
    payoffs = [r.player0_payoff for r in stats.results]
    analysis = analyzer.analyze_payoffs(payoffs, big_blind=rules.big_blind)

    print(f"\nStatistical Analysis ({confidence*100:.0f}% confidence):")
    print(f"  Mean: {analysis['bb_per_hand']:+.3f} bb/hand")
    print(f"  95% CI: [{analysis['ci_lower']:+.3f}, {analysis['ci_upper']:+.3f}] bb/hand")
    print(f"  Std Dev: {analysis['std_dev']:.3f} bb/hand")


def compare_checkpoints(
    checkpoint_dirs: List[Path],
    num_hands: int,
    seed: Optional[int],
    verbose: bool,
    confidence: float,
):
    """Compare two checkpoints head-to-head."""
    print("\nComparing checkpoints:")
    print(f"  Checkpoint 1: {checkpoint_dirs[0]}")
    print(f"  Checkpoint 2: {checkpoint_dirs[1]}")
    print("=" * 60)

    # Load config and solvers
    config = Config.default()

    print("\nLoading solvers...")
    print("Solver 1:")
    solver1 = load_solver(checkpoint_dirs[0], config)
    print("Solver 2:")
    solver2 = load_solver(checkpoint_dirs[1], config)

    # Setup evaluator
    rules = GameRules(
        small_blind=config.get("game.small_blind", 1),
        big_blind=config.get("game.big_blind", 2),
    )
    action_abs = ActionAbstraction(config.get_section("action_abstraction"))
    card_abs = RankBasedBucketing()

    evaluator = HeadToHeadEvaluator(
        rules=rules,
        action_abstraction=action_abs,
        card_abstraction=card_abs,
        starting_stack=config.get("game.starting_stack", 200),
    )

    # Run match
    print(f"\nRunning head-to-head match ({num_hands} hands)...")
    stats = evaluator.play_match(
        solver0=solver1,
        solver1=solver2,
        num_hands=num_hands,
        seed=seed,
    )

    # Print results
    print_match_results(stats, "Head-to-Head", verbose)

    # Statistical significance
    analyzer = MatchStatisticsAnalyzer(confidence_level=confidence)
    payoffs1 = [r.player0_payoff for r in stats.results]
    payoffs2 = [r.player1_payoff for r in stats.results]

    comparison = analyzer.compare_strategies(payoffs1, payoffs2)

    print("\nStatistical Comparison:")
    print(f"  Mean Difference: {comparison['mean_difference']/rules.big_blind:+.3f} bb/hand")
    print(f"  t-statistic: {comparison['t_statistic']:.3f}")
    print(f"  p-value: {comparison['p_value']:.4f}")
    print(
        f"  Significant at {confidence*100:.0f}%: {'Yes' if comparison['is_significant'] else 'No'}"
    )


def print_match_results(stats: MatchStatistics, title: str, verbose: bool):
    """Print formatted match results."""
    print(f"\n{title} Results:")
    print("=" * 60)
    print(f"Hands Played: {stats.num_hands}")
    print(
        f"Player 0: {stats.player0_wins} wins ({stats.player0_wins/stats.num_hands*100:.1f}%), "
        f"{stats.player0_bb_per_hand:+.2f} bb/hand"
    )
    print(
        f"Player 1: {stats.player1_wins} wins ({stats.player1_wins/stats.num_hands*100:.1f}%), "
        f"{stats.player1_bb_per_hand:+.2f} bb/hand"
    )
    print(f"Showdown Rate: {stats.showdown_pct:.1f}%")

    if verbose:
        print("\nDetailed Results:")
        for i, result in enumerate(stats.results[:10]):  # Show first 10
            print(
                f"  Hand {i+1}: P0={result.player0_payoff:+4d}, "
                f"P1={result.player1_payoff:+4d}, "
                f"Street={result.final_street.name}"
            )
        if len(stats.results) > 10:
            print(f"  ... ({len(stats.results) - 10} more hands)")


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Validate arguments
    if not args.checkpoint and not args.compare:
        print("Error: Must specify --checkpoint or --compare")
        return 1

    try:
        if args.compare:
            compare_checkpoints(
                checkpoint_dirs=args.compare,
                num_hands=args.hands,
                seed=args.seed,
                verbose=args.verbose,
                confidence=args.confidence,
            )
        else:
            evaluate_checkpoint(
                checkpoint_dir=args.checkpoint,
                num_hands=args.hands,
                seed=args.seed,
                verbose=args.verbose,
                confidence=args.confidence,
            )

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
